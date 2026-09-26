import os
import re
import pickle
import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel

from prompt_builder import (
    make_recent_k,
    render_chat_prompt,
    rewrite_behavior_report,
    state_system_text,
    state_user_text,
)
from utils import (
    apply_report_dropout,
    build_model_and_tokenizer,
    build_model_and_tokenizer_cont,
    data_collator,
    ddp_barrier,
    get_rank,
    get_report,
    is_rank0,
    normalize_date,
    normalize_weekend,
    tokenize_with_assistant_mask,
)

# ─────────────────────────────────────────────────────────
# Global config
# ─────────────────────────────────────────────────────────
SEED = 3407

# matches the exact phase1 answer format: "<Location>#<ID> at HH:MM:SS"
TARGET_PATTERN = re.compile(r"^.+#\d+\s+at\s+\d{2}:\d{2}:\d{2}$")

def invert_dict(d):
    return {value: key for key, value in d.items()}

MOTIVATION_LABELS = [
    "Return",
    "Personal Service",
    "Entertainment",
    "Dining",
    "Transit",
    "Religious",
    "Education",
    "Work",
    "Social",
    "Shopping",
    "Healthcare",
]

mot2id = {name: idx for idx, name in enumerate(MOTIVATION_LABELS)}

id2mot = invert_dict(mot2id)

def is_valid_target(tgt: str) -> bool:
    if tgt is None:
        return False
    if "\n" in str(tgt) or "\r" in str(tgt):
        return False
    t = str(tgt).strip()
    return bool(TARGET_PATTERN.match(t))


# ─────────────────────────────────────────────────────────
# Dataset assembly
# ─────────────────────────────────────────────────────────
def load_dataset(csv_path: str, history_k: int, report_dropout: float = 0.0) -> Dataset:
    df = pd.read_csv(csv_path)
    print(csv_path)
    # required CSV columns: person->pkl path for behavior report, motivation_id->id2mot
    # label id, llm_text->generation target, history->recent_k source, date/is_weekend->context fields
    required_cols = [
    "person", "original", "motivation_id", "llm_text",
    "date", "is_weekend", "history"
]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in CSV: {c}")

    df["person"] = df["person"].str.replace("/workspaces/TrajGen/", "", regex=False)

    df["behavior_report"] = df["person"].apply(get_report)
    df["behavior_report"] = df["behavior_report"].apply(rewrite_behavior_report)
    df = apply_report_dropout(df, report_dropout)

    before = len(df)
    df = df[
        df["behavior_report"].notna()
        & df["original"].notna()
        & df["motivation_id"].notna()
        & df["llm_text"].notna()
        & df["date"].notna()
        & df["is_weekend"].notna()
    ].copy()

    df["date_str"] = df["date"].apply(normalize_date)
    df["weekend_str"] = df["is_weekend"].apply(normalize_weekend)

    # df["context"] = df["context"].fillna("").astype(str)
    df["history"] = df["history"].fillna("").astype(str)
    df["recent_k"] = df["history"].apply(lambda x: make_recent_k(x, k=history_k))

    df["llm_text"] = df["llm_text"].astype(str)
    df = df[df["llm_text"].apply(is_valid_target)].copy()
    after = len(df)

    if is_rank0():
        print(f"[Data] kept {after}/{before} rows after strict target-format filtering.")
        print(f"[Data] train size (final): {after}")

    # store raw fields; tokenize step will render with chat_template
    df["system_text"] = state_system_text()

    df["user_text"] = [
        state_user_text(r, h, o, id2mot[m], d, w)
        for r, h, o, m, d, w in zip(
            df["behavior_report"],
            df["recent_k"],
            # df["context"],
            df["original"],
            df["motivation_id"],
            df["date_str"],
            df["weekend_str"],
        )
    ]
    df["answer_raw"] = df["llm_text"].astype(str)

    return Dataset.from_pandas(df[["system_text", "user_text", "answer_raw"]], preserve_index=False)


# ─────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="llm_predictions_combined.csv")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--phase1_lora", type=str, default="None")
    ap.add_argument("--output_dir", type=str, default="outputs_phase1_llama_8B")
    ap.add_argument("--save_path", type=str, default="ft_out/phase1_next_state/llama_8B_phase1")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--per_device_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--history_k", type=int, default=30)
    ap.add_argument("--report_dropout", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    dataset = load_dataset(args.csv, args.history_k, args.report_dropout)
    if args.phase1_lora == "None":
        model, tokenizer = build_model_and_tokenizer(args.model)
    else:
        model, tokenizer = build_model_and_tokenizer_cont(args.model, args.phase1_lora)
    dataset = tokenize_with_assistant_mask(dataset, tokenizer, args.max_seq_len)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=200,
        max_steps=args.max_steps,
        learning_rate=1e-4,
        logging_steps=10,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        bf16=True,
        fp16=False,
        report_to="none",
        save_steps=1000,
        save_total_limit=5,
        ddp_find_unused_parameters=os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "false").lower() in {"1", "true", "yes"},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=lambda feats: data_collator(feats, pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()

    ddp_barrier()
    if is_rank0():
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print("Phase 1 training done. Saved to:", args.save_path)

if __name__ == "__main__":
    main()
