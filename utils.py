from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

from prompt_builder import render_chat_prompt


SCRIPT_DIR = Path(__file__).resolve().parent


# --- rank / ddp helpers ---

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_rank0() -> bool:
    return get_rank() == 0


def ddp_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


# --- dataset construction ---

def get_report(pkl_input):
    # person column can be a pickle filepath (str) or an already-loaded object;
    # extract the behavior report from index/key 3 if possible.
    try:
        if isinstance(pkl_input, str):
            pkl_path = Path(pkl_input)
            if not pkl_path.is_absolute():
                candidates = (SCRIPT_DIR / pkl_path, SCRIPT_DIR.parent / pkl_path)
                pkl_path = next(
                    (candidate for candidate in candidates if candidate.is_file()),
                    candidates[0],
                )
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
        else:
            data = pkl_input

        if isinstance(data, dict):
            return str(data.get(3, data.get("3", "No report")))

        if isinstance(data, (list, tuple)):
            return str(data[3]) if len(data) > 3 else "No report"

        return str(data)
    except Exception:
        return "Report extraction failed"


def normalize_date(x):
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return str(x)


def normalize_weekend(x):
    if isinstance(x, (int, float)):
        return "1" if int(x) == 1 else "0"
    s = str(x).strip().lower()
    if s in {"true", "t", "yes", "y", "weekend", "1"}:
        return "1"
    if s in {"false", "f", "no", "n", "weekday", "0"}:
        return "0"
    return str(x)


def apply_report_dropout(df: pd.DataFrame, dropout: float, seed: int = 3407) -> pd.DataFrame:
    if dropout <= 0:
        return df

    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < dropout

    df = df.copy()
    df.loc[mask, "behavior_report"] = "No behavior pattern available."

    if is_rank0():
        print(f"[Report Dropout] dropout={dropout}, dropped={mask.sum()}/{len(df)}")

    return df


# --- tokenization & collation ---

def tokenize_with_assistant_mask(dataset: Dataset, tokenizer: AutoTokenizer, max_seq_len: int) -> Dataset:
    eos = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    def _tok(batch):
        sys_list = batch["system_text"]
        usr_list = batch["user_text"]
        ans_list = batch["answer_raw"]

        input_ids_list, attn_list, labels_list = [], [], []

        for sys_t, usr_t, ans_raw in zip(sys_list, usr_list, ans_list):
            prompt = render_chat_prompt(tokenizer, sys_t, usr_t)
            answer = f"{str(ans_raw).strip()}{eos}"

            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            input_ids = (p_ids + a_ids)[:max_seq_len]
            attention_mask = [1] * len(input_ids)

            labels = ([-100] * len(p_ids) + a_ids)[:max_seq_len]
            if len(labels) < len(input_ids):
                labels += [-100] * (len(input_ids) - len(labels))
            else:
                labels = labels[: len(input_ids)]

            input_ids_list.append(input_ids)
            attn_list.append(attention_mask)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "attention_mask": attn_list, "labels": labels_list}

    return dataset.map(_tok, batched=True, remove_columns=["system_text", "user_text", "answer_raw"])


def data_collator(features, pad_token_id: int):
    max_len = max(len(f["input_ids"]) for f in features)
    batch_input_ids, batch_attention, batch_labels = [], [], []

    for f in features:
        ids, attn, lab = f["input_ids"], f["attention_mask"], f["labels"]
        pad_len = max_len - len(ids)
        batch_input_ids.append(ids + [pad_token_id] * pad_len)
        batch_attention.append(attn + [0] * pad_len)
        batch_labels.append(lab + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


# --- model / LoRA setup ---

def build_model_and_tokenizer(model_name: str):
    hf_token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        # many causal LMs don't have pad_token; use eos for padding
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Training requires GPU.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device_id = torch.cuda.current_device()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device_id},  # critical for 4bit + DDP
        quantization_config=bnb_config,
        torch_dtype=dtype,
        token=hf_token,
    )
    model.config.use_cache = False

    # LoRA (works for Llama/Qwen2.5)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # older Trainer sometimes tries to move; mark as parallelizable
    model.is_parallelizable = True
    model.model_parallel = True

    if is_rank0():
        print(f"[Model] {model_name}")
        print(f"[DDP] world_size={int(os.environ.get('WORLD_SIZE','1'))} local_rank={local_rank} device_id={device_id}")
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return model, tokenizer


def build_model_and_tokenizer_cont(model_name: str, phase1_lora_path: str):
    hf_token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Training requires GPU.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device_id = torch.cuda.current_device()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device_id},
        quantization_config=bnb_config,
        torch_dtype=dtype,
        token=hf_token,
    )
    base.config.use_cache = False

    # Continue training SAME adapter from phase1
    model = PeftModel.from_pretrained(base, phase1_lora_path, is_trainable=True)

    model.is_parallelizable = True
    model.model_parallel = True

    if is_rank0():
        print(f"[Model] {model_name}")
        print(f"[DDP] world_size={int(os.environ.get('WORLD_SIZE','1'))} local_rank={local_rank} device_id={device_id}")
        print(f"[LoRA] continue from: {phase1_lora_path}")

    return model, tokenizer
