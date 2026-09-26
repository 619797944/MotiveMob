from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

import transformer_predictor as tp
from transformer_predictor import (
    CAT_COLS,
    TIME_CAT_COLS,
    USER_FAV_COLS,
    CAT_FAV_COLS,
    PKL_DIR,
    GLOBAL_LOC_MAP_PATH,
    GRID_MAP_PATH,
    LOCATION_ACTIVITY_MAP_PATH,
    id2mot,
    load_reference_data,
    construct_df,
    collect_transformer_candidates,
    sanity_check_transformer_output,
    parse_user_pkl,
    extract_date_and_steps,
    get_lat_lng,
)
from prompt_builder import (
    judge_system_text,
    judge_user_text,
    make_earlier_today,
    render_chat_prompt,
    rewrite_behavior_report,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *_args, **_kwargs):
        return iterable


def resolve_sample_paths(pkl_dir: Path) -> List[Path]:
    # Must return the same set as construct_df() - load_person_contexts() keys
    # its lookups by these same paths.
    return sorted(Path(pkl_dir).rglob("*.pkl"))

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lng1 = a
    lat2, lng2 = b
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    h = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(h))

def candidate_coord(subcat_id: int, branch_id: int) -> Optional[Tuple[float, float]]:
    # id2subcat/map_loc are reassigned by tp.load_reference_data(), so they must
    # be read via the module (tp.xxx) rather than imported by name at load time.
    subcategory = tp.id2subcat.get(int(subcat_id))
    if subcategory is None:
        return None
    loc_name = tp.map_loc.get(f"{subcategory}#{int(branch_id)}")
    if loc_name is None:
        return None
    try:
        return get_lat_lng(loc_name)
    except (IndexError, TypeError, ValueError):
        return None

def candidate_texts_from_triples(
    triple_candidates,
    cur_time_bin: int,
    max_candidates: int,
    current_lat: Optional[float] = None,
    current_lng: Optional[float] = None,
    include_location: bool = False,
    allow_equal_time: bool = False,
) -> List[Dict]:
    candidates = []
    seen = set()
    sorted_candidates = sorted(triple_candidates, key=lambda x: float(x[1]), reverse=True)
    for combo, score in sorted_candidates:
        try:
            subcat_id, branch_id, time_bin = combo
        except Exception:
            continue
        ok, _err, text = sanity_check_transformer_output(
            subcat_id=subcat_id,
            branch_id=branch_id,
            time_bin=time_bin,
            cur_time_bin=cur_time_bin,
            allow_equal_time=allow_equal_time,
        )
        if not ok or text in seen:
            continue
        seen.add(text)
        candidate = {
            "text": text,
            "score": float(score),
            "subcat_id": int(subcat_id),
            "branch_id": int(branch_id),
            "time_bin": int(time_bin),
        }
        if include_location:
            coord = candidate_coord(int(subcat_id), int(branch_id))
            candidate["current_lat"] = current_lat
            candidate["current_lng"] = current_lng
            candidate["candidate_lat"] = coord[0] if coord is not None else None
            candidate["candidate_lng"] = coord[1] if coord is not None else None
            if coord is not None and current_lat is not None and current_lng is not None:
                candidate["distance_km"] = haversine_km((current_lat, current_lng), coord)
            else:
                candidate["distance_km"] = None
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates

def normalize_weekend_to_01(x) -> str:
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "weekend"}:
        return "1"
    if s in {"0", "false", "f", "no", "n", "weekday"}:
        return "0"
    try:
        return "1" if int(float(s)) == 1 else "0"
    except Exception:
        return "0"

def load_person_contexts(pkl_paths: List[Path], history_days: int) -> Dict[str, Dict]:
    contexts = {}
    for p in pkl_paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        trajectories, _motivations, report = parse_user_pkl(obj)

        days = []
        by_date = {}
        for traj in trajectories:
            date_str, steps = extract_date_and_steps(traj)
            if not date_str or not steps:
                continue
            date_key = pd.to_datetime(date_str).strftime("%Y-%m-%d")
            text = f"Activities at {date_key}: " + ", ".join(steps)
            days.append((pd.to_datetime(date_key).date(), date_key, steps, text))
            by_date[date_key] = steps

        days = sorted(days, key=lambda x: x[0])
        histories = {}
        for idx, (date_obj, date_key, _steps, _text) in enumerate(days):
            prev = [x[3] for x in days[:idx] if x[0] < date_obj]
            histories[date_key] = "\n".join(prev[-history_days:])

        contexts[str(p)] = {
            "report": report,
            "histories": histories,
            "steps_by_date": by_date,
        }
        contexts[str(p.resolve())] = contexts[str(p)]
    return contexts

@dataclass
class JudgeGenConfig:
    max_new_tokens: int = 8
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9

def load_judge_model(model_name: str, lora_path: Optional[str] = None, load_4bit: bool = True):
    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Please run Llama judge on GPU.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "token": hf_token,
    }
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if lora_path and lora_path != "None":
        if PeftModel is None:
            raise RuntimeError("peft is not available, cannot load --judge-lora.")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def generate_one_line(model, tokenizer, system_text: str, user_text: str, cfg: "JudgeGenConfig") -> str:
    prompt = render_chat_prompt(tokenizer, system_text, user_text)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    kwargs = {
        **inputs,
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        kwargs["temperature"] = cfg.temperature
        kwargs["top_p"] = cfg.top_p

    out = model.generate(**kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip().splitlines()[0].strip() if text.strip() else ""

def parse_choice(text: str, n_candidates: int) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", str(text))
    if not m:
        return None
    choice = int(m.group(1))
    return choice if 1 <= choice <= n_candidates else None

def run_llm_judge_sampling(args):
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")

    random.seed(args.seed)
    load_reference_data(args.global_loc_map, args.grid_map, args.location_activity_map)

    if args.input_samples_csv is not None:
        df = pd.read_csv(args.input_samples_csv).reset_index(drop=True)
        pkl_paths = [Path(p) for p in df["person"].dropna().astype(str).unique()]
    else:
        df = construct_df(pkl_dir=args.pkl_dir).reset_index(drop=True)
        pkl_paths = resolve_sample_paths(args.pkl_dir)

    cols = CAT_COLS + TIME_CAT_COLS + USER_FAV_COLS + CAT_FAV_COLS
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    for c in ["weekday_total_km", "weekend_total_km", "cur_lat", "cur_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float")
    for c in ["next_subcat_id", "next_grid_id", "next_time_bin", "next_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[c])
        df[c] = df[c].astype("int64")
    df = df.reset_index(drop=True)
    if "row_id" not in df.columns:
        df["row_id"] = df.index
    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").fillna(-1).astype("int64")

    # --prepare-samples-only stops here: save the cleaned candidate dataframe and
    # skip context loading / LLM judging entirely (used to fan out shards afterward).
    if args.prepare_samples_only:
        prepared_path = args.prepared_samples_csv or args.output_csv
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(prepared_path, index=False)
        print(f"Saved prepared alternative samples to {prepared_path}")
        print(f"Rows: {len(df):,}")
        return

    contexts = load_person_contexts(pkl_paths, history_days=args.history_days)

    # Shard partition by row_id % num_shards so merge_llm_judge_shards can reassemble rows in order later.
    work_df = df[df["row_id"] % args.num_shards == args.shard_id].copy()
    if args.limit is not None:
        work_df = work_df.head(args.limit).copy()
    work_df = work_df.reset_index(drop=True)

    print(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(work_df):,} rows out of {len(df):,} total rows"
    )

    _pair_cands, triple_cands = collect_transformer_candidates(
        valid_df=work_df,
        grid_map_path=args.grid_map,
        model_ckpt_path=args.transformer_ckpt,
        K_sub=args.k_sub,
        K_grid=args.k_grid,
        K_time=args.k_time,
        K_pair_out=args.candidate_count,
        K_triple_out=args.transformer_triple_count,
        alpha_sub=args.alpha_sub,
        alpha_grid=args.alpha_grid,
        alpha_time=args.alpha_time,
        alpha_user=args.alpha_user,
        lambda_d=args.lambda_d,
    )

    model, tokenizer = load_judge_model(
        args.judge_model,
        lora_path=args.judge_lora,
        load_4bit=not args.no_4bit,
    )
    gen_cfg = JudgeGenConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    chosen_texts = []
    chosen_indices = []
    chosen_scores = []
    judge_raw = []
    candidate_jsons = []
    fallback_reasons = []
    reports = []
    histories = []
    earlier_todays = []
    printed_first_prompt = False

    for i, row in tqdm(work_df.iterrows(), total=len(work_df), desc="LLM judging"):
        cur_time_bin = int(row.get("cur_time_bin", -1))
        is_start_row = str(row.get("original", "")).strip() == "START"
        current_lat = None
        current_lng = None
        if args.include_location_in_prompt and not is_start_row:
            try:
                current_lat = float(row["cur_lat"])
                current_lng = float(row["cur_lng"])
            except Exception:
                current_lat = None
                current_lng = None
        candidates = candidate_texts_from_triples(
            triple_cands[i],
            cur_time_bin=cur_time_bin,
            max_candidates=args.candidate_count,
            current_lat=current_lat,
            current_lng=current_lng,
            include_location=args.include_location_in_prompt,
            allow_equal_time=is_start_row,
        )

        person = str(row["person"])
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        context = contexts.get(person) or contexts.get(str(Path(person).resolve()), {})
        report = rewrite_behavior_report(context.get("report", ""))
        history = context.get("histories", {}).get(date_str, "")
        steps_today = context.get("steps_by_date", {}).get(date_str, [])
        earlier_today = make_earlier_today(steps_today, row["original"])

        reports.append(report)
        histories.append(history)
        earlier_todays.append(earlier_today)
        candidate_jsons.append(candidates)

        if not candidates:
            chosen_texts.append(None)
            chosen_indices.append(None)
            chosen_scores.append(None)
            judge_raw.append("")
            fallback_reasons.append("no_valid_transformer_candidates")
            continue

        motivation = id2mot.get(int(row["motivation_id"]), str(row["motivation_id"]))
        user_text = judge_user_text(
            report=report,
            recent_history=history,
            earlier_today=earlier_today,
            current_state=row["original"],
            current_lat=current_lat,
            current_lng=current_lng,
            motivation=motivation,
            date_str=date_str,
            is_weekend=normalize_weekend_to_01(row["is_weekend"]),
            candidates=candidates,
            include_location=args.include_location_in_prompt,
        )

        if args.print_first_prompt and not printed_first_prompt:
            print("=" * 80)
            print("FIRST SAMPLE SYSTEM PROMPT")
            print("=" * 80)
            print(judge_system_text())
            print("=" * 80)
            print("FIRST SAMPLE USER PROMPT")
            print("=" * 80)
            print(user_text)
            print("=" * 80)
            print("FIRST SAMPLE SOURCE FIELDS")
            print("=" * 80)
            print(f"row_index={i}")
            print(f"person={person}")
            print(f"date={date_str}")
            print(f"current_state={row['original']}")
            if args.include_location_in_prompt:
                print(f"current_lat={current_lat}")
                print(f"current_lng={current_lng}")
            print(f"motivation_id={row['motivation_id']}")
            print(f"motivation={motivation}")
            print(f"num_candidates={len(candidates)}")
            print("=" * 80)
            printed_first_prompt = True
            if args.print_first_prompt_only:
                return

        raw = generate_one_line(model, tokenizer, judge_system_text(), user_text, gen_cfg)
        choice = parse_choice(raw, len(candidates))
        fallback = None
        if choice is None:
            choice = 1
            fallback = f"invalid_llm_choice: {raw!r}"

        selected = candidates[choice - 1]
        chosen_texts.append(selected["text"])
        chosen_indices.append(choice)
        chosen_scores.append(selected["score"])
        judge_raw.append(raw)
        fallback_reasons.append(fallback)

    out = work_df.copy()
    out["behavior_report_for_judge"] = reports
    out["recent_history_for_judge"] = histories
    out["earlier_today_for_judge"] = earlier_todays
    out["candidate_set_json"] = [json.dumps(x, ensure_ascii=False) for x in candidate_jsons]
    out["llm_judge_raw"] = judge_raw
    out["llm_choice_index"] = chosen_indices
    out["llm_choice_score"] = chosen_scores
    out["llm_judge_text"] = chosen_texts
    out["llm_judge_fallback"] = fallback_reasons

    if "person" in out.columns:
        out["person"] = out["person"].astype(str)

    output_csv = args.output_csv
    output_parquet = args.output_parquet
    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv / f"shard_{args.shard_id}.csv"
    if output_parquet.suffix.lower() != ".parquet":
        output_parquet = output_parquet / f"shard_{args.shard_id}.parquet"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    out.to_parquet(output_parquet, index=False)
    print(f"Saved LLM-judged alternatives to {output_csv} and {output_parquet}")

def merge_llm_judge_shards(args):
    csv_paths = sorted(args.shard_dir.glob(args.shard_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No shard files found: {args.shard_dir / args.shard_glob}")

    parts = []
    for path in csv_paths:
        print(f"Reading shard: {path}")
        parts.append(pd.read_csv(path))

    merged = pd.concat(parts, ignore_index=True)
    if "row_id" in merged.columns:
        merged = merged.sort_values("row_id").reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    merged.to_parquet(args.output_parquet, index=False)
    print(f"Merged {len(csv_paths)} shards, rows={len(merged):,}")
    print(f"Saved merged outputs to {args.output_csv} and {args.output_parquet}")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Use Transformer candidate sets plus an LLM judge for alternative sampling."
    )
    parser.add_argument("--pkl-dir", type=Path, default=PKL_DIR)
    parser.add_argument("--global-loc-map", type=Path, default=GLOBAL_LOC_MAP_PATH)
    parser.add_argument("--grid-map", type=Path, default=GRID_MAP_PATH)
    parser.add_argument("--location-activity-map", type=Path, default=LOCATION_ACTIVITY_MAP_PATH)
    parser.add_argument("--transformer-ckpt", type=Path, required=True)
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--judge-lora", type=str, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("llm_judged_transformer_samples.csv"))
    parser.add_argument("--output-parquet", type=Path, default=Path("llm_judged_transformer_samples.parquet"))
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--transformer-triple-count", type=int, default=50)
    parser.add_argument("--k-sub", type=int, default=8)
    parser.add_argument("--k-grid", type=int, default=8)
    parser.add_argument("--k-time", type=int, default=8)
    parser.add_argument("--history-days", type=int, default=3)
    parser.add_argument("--alpha-sub", type=float, default=0.8)
    parser.add_argument("--alpha-grid", type=float, default=0.9)
    parser.add_argument("--alpha-time", type=float, default=0.55)
    parser.add_argument("--alpha-user", type=float, default=1.0)
    parser.add_argument("--lambda-d", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--input-samples-csv", type=Path, default=None)
    parser.add_argument("--prepare-samples-only", action="store_true")
    parser.add_argument("--prepared-samples-csv", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--shard-dir", type=Path, default=Path("."))
    parser.add_argument("--shard-glob", type=str, default="shard_*.csv")
    parser.add_argument("--print-first-prompt", action="store_true")
    parser.add_argument("--print-first-prompt-only", action="store_true")
    parser.add_argument(
        "--include-location-in-prompt",
        action="store_true",
        help=(
            "Add current/candidate lat/lng and distance_km to the LLM prompt "
            "and candidate_set_json."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    # --merge-shards selects the merge-shards path; otherwise run_llm_judge_sampling
    # handles both the --prepare-samples-only and per-shard inference modes internally.
    if args.merge_shards:
        merge_llm_judge_shards(args)
    else:
        run_llm_judge_sampling(args)


if __name__ == "__main__":
    main()
