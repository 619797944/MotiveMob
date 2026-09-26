#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from location_utils import (
    broad_category,
    coordinates,
    geodistance,
    parse_history_events,
    parse_report_events,
    time_slot,
)
from prompt_builder import (
    make_recent_k,
    motivation_system_text,
    motivation_user_text,
    render_chat_prompt,
    rewrite_behavior_report,
    state_system_text,
    state_user_text,
)

# ─────────────────────────────────────────────────────────
# Global config
# ─────────────────────────────────────────────────────────
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
    "END",
]
MOT_SET = set(MOTIVATION_LABELS)

TARGET_RE = re.compile(r"^(?P<loc>.+)#(?P<id>\d+)\s+at\s+(?P<t>\d{2}:\d{2}:\d{2})$")

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Provenance only: labels output rows with the test-set period; nothing in the generation loop reads it.
DATASET_PROFILE_PERIOD = {
    "seen_seen": "2019_train",
    "seen_unseen_normal": "2019_train",
    "unseen_seen": "2019_train",
    "unseen_unseen_normal": "2019_train",
    "seen_unseen_abnormal": "2020_03",
    "unseen_unseen_abnormal": "2020_03",
}
PROFILE_PERIODS = tuple(sorted(set(DATASET_PROFILE_PERIOD.values())))


# ─────────────────────────────────────────────────────────
# Text process
# ─────────────────────────────────────────────────────────
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


def hms_to_seconds(hms: str) -> int:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def day_offset_seconds(hms: str) -> int:
    """Map HH:MM:SS into [0, 86399] w.r.t. day starting at 04:00:00."""
    t = hms_to_seconds(hms)
    boundary = 4 * 3600
    if t >= boundary:
        return t - boundary
    return t + (24 * 3600 - boundary)


def crossed_into_next_day(prev_hms: str, next_hms: str) -> bool:
    """Stop if prev < 04:00 and next >= 04:00, i.e. crosses into next shifted day."""
    boundary = 4 * 3600
    return (hms_to_seconds(prev_hms) < boundary) and (hms_to_seconds(next_hms) >= boundary)


def merge_recent_with_generated(
    seed_history: str,
    generated_states: List[str],
    k: int = 30,
) -> str:
    """Recent-K block. The deployed loop passes [] -- history only, no rollout."""
    seed_recent = make_recent_k(seed_history, k=k)
    parts = []
    if seed_recent.strip():
        parts.extend([x.strip() for x in seed_recent.splitlines() if x.strip()])
    parts.extend([x.strip() for x in generated_states if str(x).strip()])
    return "\n".join(parts[-k:])


def load_valid_locations(global_loc_map_path: str) -> Optional[set]:
    if not global_loc_map_path:
        return None
    with open(global_loc_map_path, "rb") as f:
        loc_map = pickle.load(f)
    return {str(v).strip() for v in loc_map.values()}


def parse_state(line: str, valid_locations: Optional[set] = None) -> Optional[Tuple[str, str]]:
    m = TARGET_RE.match(str(line).strip())
    if not m:
        return None
    state_str = str(line).strip()
    if valid_locations is not None:
        loc_part = state_str.split(" at ")[0].strip()
        if loc_part not in valid_locations:
            return None
    return state_str, m.group("t")


def parse_motivation(line: str) -> Optional[str]:
    m = str(line).strip()
    return m if m in MOT_SET else None


def split_generated_state(state: str) -> tuple[str, str]:
    """Return the (location, timestamp) identity of one generated check-in."""
    try:
        location, timestamp = str(state).rsplit(" at ", 1)
    except ValueError as exc:
        raise ValueError(f"Generated state has no ' at ' separator: {state!r}") from exc
    return location.strip(), timestamp.strip()


def is_private_home(location: str) -> bool:
    """True only for the Foursquare private-home pseudo-category."""
    return str(location).split("#", 1)[0].strip().casefold() == "home (private)"


# ─────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────
@dataclass
class GenConfig:
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


def load_base_and_lora(model_name: str, lora_path: str | None = None):
    hf_token = os.environ.get("HF_TOKEN", None)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Please run on GPU.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=dtype,
        token=hf_token,
    )
    base.eval()

    if lora_path and lora_path != "None":
        model = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
        model.eval()
        return model, tok
    return base, tok


@torch.no_grad()
def generate_one_line(model, tokenizer, system_text: str, user_text: str, gen: GenConfig) -> str:
    prompt = render_chat_prompt(tokenizer, system_text, user_text)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    kwargs = dict(
        **inputs,
        max_new_tokens=gen.max_new_tokens,
        do_sample=gen.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if gen.do_sample:
        kwargs["temperature"] = gen.temperature
        kwargs["top_p"] = gen.top_p

    out = model.generate(**kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def build_regrounding_candidates(
    history: str,
    report: Optional[str] = None,
    *,
    include_history: bool = True,
):
    """POIs the user is actually known to visit, bucketed by activity category.
    This pool is consulted ONLY when the current step's resampling is
    exhausted. It never touches a step that was accepted.
    """
    candidates = defaultdict(list)
    events = parse_history_events(history) if include_history else []
    if report is not None:
        events.extend(parse_report_events(report))
    for location, time_text in events:
        if is_private_home(location):
            continue
        try:
            point = coordinates(location)
            slot = time_slot(time_text)
        except (KeyError, ValueError, IndexError):
            continue
        candidates[broad_category(location)].append((location, slot, point))
    return candidates


def repeated_short_cycle_lengths(
    locations: List[str],
    *,
    max_block_length: int = 5,
    min_rounds: int = 3,
) -> List[int]:
    """Find short location-block lengths repeated consecutively enough times."""
    if max_block_length < 1:
        raise ValueError("max_block_length must be at least 1")
    if min_rounds < 2:
        raise ValueError("min_rounds must be at least 2")

    found = set()
    count = len(locations)
    for block_length in range(1, min(max_block_length, count // min_rounds) + 1):
        span = block_length * min_rounds
        for start in range(count - span + 1):
            block = locations[start: start + block_length]
            if all(
                locations[
                    start + round_index * block_length:
                    start + (round_index + 1) * block_length
                ]
                == block
                for round_index in range(1, min_rounds)
            ):
                found.add(block_length)
                break
    return sorted(found)


# ─────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────
@dataclass
class TrajGenConfig:
    max_transitions: int = 25
    retries_per_step: int = 8
    recent_k: int = 30
    phase1_do_sample: bool = False
    phase1_temperature: float = 0.7
    phase1_top_p: float = 0.9
    validate_locations: bool = True
    loc_map: str = "../data/global_loc_map.pkl"
    include_history_candidates: bool = True
    include_report_candidates: bool = True
    # Constraint (d): speed feasibility. None disables the test entirely.
    speed_vmax_kmh: Optional[float] = None
    speed_retry_temperature_start: float = 0.3
    speed_retry_temperature_end: float = 0.9
    speed_reground_on_exhaustion: bool = True
    speed_reground_rule: str = "timeslot"
    # Stop rule: end the day instead of extending a short repeating pattern.
    stop_on_repetition_after: int = 0
    stop_repetition_max_block_length: int = 5
    stop_repetition_min_rounds: int = 3


class OnlineConstrainedTrajectoryGenerator:
    """One day of check-ins, generated under per-step constraints only."""

    def __init__(self, model_name: str, phase1_lora: str, phase2_lora: str, cfg: TrajGenConfig):
        self.cfg = cfg
        self.phase1_model, self.tok1 = load_base_and_lora(model_name, phase1_lora)
        self.phase2_model, self.tok2 = load_base_and_lora(model_name, phase2_lora)
        self.valid_locations = (
            load_valid_locations(cfg.loc_map) if cfg.validate_locations else None
        )

        self.candidate_frequency: dict[str, int] = defaultdict(int)
        self.last_location_changes: list[dict] = []
        self.last_speed_rejections = 0
        self.last_speed_steps_with_rejection = 0
        self.last_speed_resample_accepted = 0
        self.last_speed_retry_exhausted = 0
        self.last_speed_reground_count = 0
        self.last_speed_reground_unfixable = 0
        self.last_stopped_on_repetition = False
        self.last_hit_max_transitions = False

    # ---- constraint (d) -------------------------------------------------
    def speed_kmh(self, previous_location, location, prev_time, timestamp):
        """Implied speed of one step, or None when the step is not spatial."""
        if previous_location is None or is_private_home(location):
            return None
        try:
            here = coordinates(location)
            there = coordinates(previous_location)
        except (KeyError, ValueError, IndexError):
            return None
        gap_slots = max(time_slot(timestamp) - time_slot(prev_time), 1)
        return geodistance(there, here) / (gap_slots * 10.0 / 60.0)

    def speed_feasible(self, state_str, previous_location, prev_time, timestamp) -> bool:
        if self.cfg.speed_vmax_kmh is None:
            return True
        location = state_str.rsplit(" at ", 1)[0].strip()
        speed = self.speed_kmh(previous_location, location, prev_time, timestamp)
        if speed is None:
            return True
        return speed <= self.cfg.speed_vmax_kmh

    # ---- re-grounding ----------------------------------------------------
    def reground_infeasible(self, state_str, previous_spatial_location, prev_time, candidates):
        """Only for steps whose every resample stayed speed-infeasible.

        Keeps the activity category, keeps only candidates that satisfy the
        speed constraint, and ranks the survivors by time-of-day similarity
        with a history-frequency tie-break.  Distance never enters the ranking.
        """
        location, timestamp = split_generated_state(state_str)

        if is_private_home(location):
            return state_str, previous_spatial_location, None
        try:
            coordinates(location)
        except (KeyError, ValueError, IndexError):
            return state_str, location, None
        if previous_spatial_location is None:
            return state_str, location, None
        try:
            previous_point = coordinates(previous_spatial_location)
        except (KeyError, ValueError, IndexError):
            return state_str, location, None

        target_slot = time_slot(timestamp)
        gap_hours = max(target_slot - time_slot(prev_time), 1) * 10.0 / 60.0
        same_category = candidates.get(broad_category(location), [])
        feasible = [
            item
            for item in same_category
            if geodistance(previous_point, item[2]) / gap_hours <= self.cfg.speed_vmax_kmh
        ]
        if not feasible:
            self.last_speed_reground_unfixable += 1
            return state_str, location, None

        if self.cfg.speed_reground_rule == "frequency":
            rank_key = lambda item: (
                -self.candidate_frequency.get(item[0], 0),
                abs(item[1] - target_slot),
            )
        else:
            rank_key = lambda item: (
                abs(item[1] - target_slot),
                -self.candidate_frequency.get(item[0], 0),
            )
        candidate_location, candidate_slot, candidate_point = min(feasible, key=rank_key)
        if candidate_location == location:
            return state_str, location, None

        self.last_speed_reground_count += 1
        change = {
            "time": timestamp,
            "old_location": location,
            "new_location": candidate_location,
            "old_jump_km": geodistance(previous_point, coordinates(location)),
            "new_jump_km": geodistance(previous_point, candidate_point),
            "category": broad_category(location),
            "candidate_time_slot": candidate_slot,
            "reason": f"speed_retry_exhausted:{self.cfg.speed_reground_rule}",
        }
        return f"{candidate_location} at {timestamp}", candidate_location, change

    # ---- the rollout -----------------------------------------------------
    def generate_states(
        self,
        report: str,
        history: str,
        date_str: str,
        is_weekend: str,
        candidate_report: Optional[str] = None,
    ) -> List[str]:
        current_state = "START"
        previous_spatial_location = None
        prev_time = None
        prev_offset = None
        out_states: List[str] = []

        self.last_location_changes = []
        self.last_speed_rejections = 0
        self.last_speed_steps_with_rejection = 0
        self.last_speed_resample_accepted = 0
        self.last_speed_retry_exhausted = 0
        self.last_speed_reground_count = 0
        self.last_speed_reground_unfixable = 0
        self.last_stopped_on_repetition = False
        self.last_hit_max_transitions = False

        candidates = build_regrounding_candidates(
            history,
            report=(candidate_report if self.cfg.include_report_candidates else None),
            include_history=self.cfg.include_history_candidates,
        )
        self.candidate_frequency = defaultdict(int)
        for bucket in candidates.values():
            for candidate_location, _slot, _point in bucket:
                self.candidate_frequency[candidate_location] += 1

        gen_mot = GenConfig(max_new_tokens=8, do_sample=False)
        gen_state = GenConfig(
            max_new_tokens=32,
            do_sample=self.cfg.phase1_do_sample,
            temperature=self.cfg.phase1_temperature,
            top_p=self.cfg.phase1_top_p,
        )
        legacy_retry_state = GenConfig(
            max_new_tokens=32,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

        def retry_config(attempt: int) -> GenConfig:
            """Rising temperature ladder; legacy 0.2 when no speed cap is set.

            The legacy retry temperature was tuned to repair formatting and
            time-order violations and barely moves the sampled POI, which is
            not enough to escape an infeasible jump.
            """
            if self.cfg.speed_vmax_kmh is None or self.cfg.retries_per_step < 3:
                return legacy_retry_state
            span = self.cfg.retries_per_step - 2
            ratio = min(max((attempt - 1) / span, 0.0), 1.0)
            temperature = self.cfg.speed_retry_temperature_start + ratio * (
                self.cfg.speed_retry_temperature_end
                - self.cfg.speed_retry_temperature_start
            )
            return GenConfig(
                max_new_tokens=32,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )

        for _step in range(self.cfg.max_transitions):
            # The deployed prompt carries the seed history only; generated check-ins are NOT appended to the Recent-K block.
            recent_text = merge_recent_with_generated(history, [], k=self.cfg.recent_k)

            # --- 1. motivation -------------------------------------------
            motivation_system = motivation_system_text()
            motivation_user = motivation_user_text(
                report,
                recent_text,
                current_state,
                date_str,
                is_weekend,
            )
            motivation_line = generate_one_line(
                self.phase2_model,
                self.tok2,
                motivation_system,
                motivation_user,
                gen_mot,
            )
            motivation = parse_motivation(motivation_line)
            if motivation is None or motivation == "END":
                break

            # --- 2. state proposal + acceptance test ----------------------
            accepted = None
            day_ended = False
            last_proposal = None
            step_speed_rejections = 0

            for attempt in range(self.cfg.retries_per_step):
                state_user = state_user_text(
                    report,
                    recent_text,
                    current_state,
                    motivation,
                    date_str,
                    is_weekend,
                )
                state_line = generate_one_line(
                    self.phase1_model,
                    self.tok1,
                    state_system_text(),
                    state_user,
                    gen_state if attempt == 0 else retry_config(attempt),
                )

                # (a) format + POI validity
                parsed = parse_state(state_line, self.valid_locations)
                if parsed is None:
                    continue
                state_str, timestamp = parsed
                offset = day_offset_seconds(timestamp)

                # (b) time order / (c) day boundary
                if prev_offset is not None:
                    if offset < prev_offset:
                        continue
                    if crossed_into_next_day(prev_time, timestamp):
                        accepted = None
                        day_ended = True
                        break

                last_proposal = (state_str, timestamp, offset)

                # (d) speed feasibility
                if not self.speed_feasible(
                    state_str,
                    previous_spatial_location,
                    prev_time,
                    timestamp,
                ):
                    self.last_speed_rejections += 1
                    step_speed_rejections += 1
                    continue

                accepted = (state_str, timestamp, offset)
                if step_speed_rejections:
                    self.last_speed_resample_accepted += 1
                break

            if step_speed_rejections:
                self.last_speed_steps_with_rejection += 1

            retry_exhausted = False
            if accepted is None and not day_ended and last_proposal is not None:
                # Never truncate the day for an unmet constraint: keep the last proposal and let re-grounding decide if a feasible same-category POI exists.
                accepted = last_proposal
                retry_exhausted = True
                if step_speed_rejections:
                    self.last_speed_retry_exhausted += 1
            if accepted is None:
                break

            raw_state, timestamp, offset = accepted

            # --- 3. re-grounding (same step, only when resampling failed) --
            if (
                self.cfg.speed_vmax_kmh is not None
                and retry_exhausted
                and self.cfg.speed_reground_on_exhaustion
            ):
                next_state, previous_spatial_location, change = self.reground_infeasible(
                    raw_state,
                    previous_spatial_location,
                    prev_time,
                    candidates,
                )
            else:
                next_state = raw_state
                change = None
                location = raw_state.rsplit(" at ", 1)[0].strip()
                if not is_private_home(location):
                    previous_spatial_location = location

            # --- 4. stop rule --------------------------------------------
            if (
                self.cfg.stop_on_repetition_after > 0
                and len(out_states) >= self.cfg.stop_on_repetition_after
                and repeated_short_cycle_lengths(
                    [split_generated_state(state)[0] for state in out_states]
                    + [split_generated_state(next_state)[0]],
                    max_block_length=self.cfg.stop_repetition_max_block_length,
                    min_rounds=self.cfg.stop_repetition_min_rounds,
                )
            ):
                # Ends the rollout instead of extending a short repeating pattern; equivalent to the model emitting END here, so no generated prefix changes.
                self.last_stopped_on_repetition = True
                break

            if change is not None:
                change["step"] = len(out_states)
                self.last_location_changes.append(change)
            out_states.append(next_state)
            current_state = next_state
            prev_time = timestamp
            prev_offset = offset

        # A day that reaches max_transitions is written out as generated; nothing repairs it afterward.
        self.last_hit_max_transitions = len(out_states) == self.cfg.max_transitions
        return out_states

    @staticmethod
    def format_traj(date_str: str, states: List[str]) -> str:
        return f"Activities at {date_str}: " + ", ".join(states)


# ---- Sharded CSV runner ----
def infer_dataset_name(csv_path: str) -> str:
    stem = Path(csv_path).stem
    for dataset_name in sorted(DATASET_PROFILE_PERIOD, key=len, reverse=True):
        if stem.endswith(dataset_name):
            return dataset_name
    raise ValueError(
        f"Cannot infer a period label from {csv_path!r}. Use --profile_period explicitly."
    )


def resolve_profile_period(csv_path: str, requested: str) -> tuple[str, str]:
    if requested != "auto":
        try:
            dataset_name = infer_dataset_name(csv_path)
        except ValueError:
            dataset_name = Path(csv_path).stem
        return dataset_name, requested
    dataset_name = infer_dataset_name(csv_path)
    return dataset_name, DATASET_PROFILE_PERIOD[dataset_name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online per-step-constrained inference",
    )
    parser.add_argument("--csv_in", type=str, required=True)
    parser.add_argument("--csv_out", type=str, default=None)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--phase1_lora", type=str, required=True)
    parser.add_argument("--phase2_lora", type=str, required=True)
    parser.add_argument("--max_transitions", type=int, default=25)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--history_k", type=int, default=30)
    parser.add_argument("--phase1_sample", action="store_true")
    parser.add_argument("--phase1_temperature", type=float, default=0.7)
    parser.add_argument("--phase1_top_p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--loc_map",
        type=str,
        default=str(PROJECT_DIR / "data/global_loc_map.pkl"),
    )
    parser.add_argument("--no_validate_locations", action="store_true")

    # Speed feasibility (constraint d)
    parser.add_argument(
        "--speed_vmax",
        type=float,
        default=None,
        help="Reject a step whose implied speed exceeds this cap in km/h. "
             "The deployed value is 50.",
    )
    parser.add_argument("--speed_retry_temperature_start", type=float, default=0.3)
    parser.add_argument("--speed_retry_temperature_end", type=float, default=0.9)
    parser.add_argument(
        "--speed_no_reground",
        action="store_true",
        help="Keep the model POI when every resample stays infeasible.",
    )
    parser.add_argument(
        "--speed_reground_rule",
        choices=("timeslot", "frequency"),
        default="timeslot",
    )

    # Stop rule
    parser.add_argument(
        "--stop_on_repetition_after",
        type=int,
        default=0,
        help="End the day once it holds this many events and the locations "
             "have entered a short repeating pattern. 0 disables the rule. "
             "The deployed value is 15.",
    )
    parser.add_argument("--stop_repetition_max_block_length", type=int, default=5)
    parser.add_argument("--stop_repetition_min_rounds", type=int, default=3)

    # Re-grounding candidate pool
    parser.add_argument("--exclude_history_candidates", action="store_true")
    parser.add_argument("--exclude_report_candidates", action="store_true")
    parser.add_argument(
        "--history_only_candidates",
        action="store_true",
        help="Alias of --exclude_report_candidates.",
    )

    # Row provenance label only; read by nothing in the generation loop.
    parser.add_argument(
        "--profile_period",
        choices=("auto", *PROFILE_PERIODS),
        default="auto",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    if args.max_transitions < 1 or args.retries < 1:
        raise ValueError("max_transitions and retries must be at least 1")
    if args.phase1_temperature <= 0:
        raise ValueError("phase1_temperature must be positive")
    if not (0 < args.phase1_top_p <= 1):
        raise ValueError("phase1_top_p must be in (0, 1]")

    if args.speed_vmax is not None:
        if args.speed_vmax <= 0:
            raise ValueError("speed_vmax must be positive")
        if args.speed_retry_temperature_start <= 0:
            raise ValueError("speed_retry_temperature_start must be positive")
        if args.speed_retry_temperature_end < args.speed_retry_temperature_start:
            raise ValueError(
                "speed_retry_temperature_end must not be below the start value"
            )
    elif args.speed_no_reground:
        raise ValueError("--speed_no_reground requires --speed_vmax")

    if args.stop_on_repetition_after < 0:
        raise ValueError("stop_on_repetition_after must be non-negative")
    if args.stop_repetition_max_block_length < 1:
        raise ValueError("stop_repetition_max_block_length must be at least 1")
    if args.stop_repetition_min_rounds < 2:
        raise ValueError("stop_repetition_min_rounds must be at least 2")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    dataset_name, profile_period = resolve_profile_period(args.csv_in, args.profile_period)
    print(f"Dataset: {dataset_name}; period label: {profile_period} (provenance only)")
    print(
        "Per-step constraints: "
        f"speed_vmax={args.speed_vmax}; "
        f"speed_reground={'none' if args.speed_no_reground else args.speed_reground_rule}; "
        f"retry_temperature={args.speed_retry_temperature_start}->{args.speed_retry_temperature_end}; "
        f"stop_on_repetition_after={args.stop_on_repetition_after}"
    )
    print(
        "Phase-1 decoding: "
        f"phase1_sample={args.phase1_sample}; "
        f"phase1_temperature={args.phase1_temperature}; "
        f"phase1_top_p={args.phase1_top_p}"
    )

    if args.csv_out is None:
        output_path = f"out_shard{args.shard_id}.csv"
    else:
        os.makedirs(args.csv_out, exist_ok=True)
        output_path = os.path.join(args.csv_out, f"shard_{args.shard_id}.csv")

    frame_full = pd.read_csv(args.csv_in).reset_index(drop=True)
    frame_full["row_id"] = frame_full.index
    for column in ["person", "date", "is_weekend", "report", "history", "row_id"]:
        if column not in frame_full.columns:
            raise ValueError(f"CSV missing column: {column}")
    frame = frame_full[frame_full["row_id"] % args.num_shards == args.shard_id].copy()
    if args.limit is not None:
        frame = frame.head(args.limit).copy()

    cfg = TrajGenConfig(
        max_transitions=args.max_transitions,
        retries_per_step=args.retries,
        recent_k=args.history_k,
        phase1_do_sample=args.phase1_sample,
        phase1_temperature=args.phase1_temperature,
        phase1_top_p=args.phase1_top_p,
        validate_locations=not args.no_validate_locations,
        loc_map=args.loc_map,
        include_history_candidates=not args.exclude_history_candidates,
        include_report_candidates=not (
            args.history_only_candidates or args.exclude_report_candidates
        ),
        speed_vmax_kmh=args.speed_vmax,
        speed_retry_temperature_start=args.speed_retry_temperature_start,
        speed_retry_temperature_end=args.speed_retry_temperature_end,
        speed_reground_on_exhaustion=not args.speed_no_reground,
        speed_reground_rule=args.speed_reground_rule,
        stop_on_repetition_after=args.stop_on_repetition_after,
        stop_repetition_max_block_length=args.stop_repetition_max_block_length,
        stop_repetition_min_rounds=args.stop_repetition_min_rounds,
    )
    generator = OnlineConstrainedTrajectoryGenerator(
        args.model,
        args.phase1_lora,
        args.phase2_lora,
        cfg,
    )

    generated_trajectories = []
    metadata_rows = []
    for _, row in tqdm(
        frame.iterrows(),
        total=len(frame),
        desc=f"shard {args.shard_id}/{args.num_shards}",
    ):
        history = "" if pd.isna(row["history"]) else str(row["history"])
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        is_weekend = normalize_weekend_to_01(row["is_weekend"])
        raw_report = "" if pd.isna(row["report"]) else str(row["report"])
        prompt_report = rewrite_behavior_report(raw_report, keep_covid_context=True)

        error = ""
        try:
            states = generator.generate_states(
                prompt_report,
                history,
                date_str,
                is_weekend,
                candidate_report=raw_report,
            )
            generated_trajectories.append(generator.format_traj(date_str, states))
        except Exception as exc:
            print(f"[WARN] row_id={row['row_id']} failed: {repr(exc)}")
            generated_trajectories.append(None)
            error = repr(exc)

        metadata_rows.append(
            {
                "profile_period": profile_period,
                "speed_vmax_kmh": (
                    cfg.speed_vmax_kmh if cfg.speed_vmax_kmh is not None else ""
                ),
                "speed_reground_on_exhaustion": cfg.speed_reground_on_exhaustion,
                "speed_reground_rule": (
                    cfg.speed_reground_rule if cfg.speed_reground_on_exhaustion else "none"
                ),
                "speed_rejection_count": generator.last_speed_rejections,
                "speed_steps_with_rejection": generator.last_speed_steps_with_rejection,
                "speed_resample_accepted_count": generator.last_speed_resample_accepted,
                "speed_retry_exhausted_count": generator.last_speed_retry_exhausted,
                "speed_reground_count": generator.last_speed_reground_count,
                "speed_reground_unfixable_count": generator.last_speed_reground_unfixable,
                "reground_change_count": len(generator.last_location_changes),
                "stop_on_repetition_after": cfg.stop_on_repetition_after,
                "stopped_on_repetition": generator.last_stopped_on_repetition,
                "hit_max_transitions": generator.last_hit_max_transitions,
                "inference_error": error,
            }
        )

    frame["gen_traj"] = generated_trajectories
    metadata = pd.DataFrame(metadata_rows, index=frame.index)
    for column in metadata:
        frame[column] = metadata[column]
    frame.to_csv(output_path, index=False)
    print("Saved:", output_path)

    if cfg.speed_vmax_kmh is not None:
        print(
            f"Speed feasibility (vmax={cfg.speed_vmax_kmh} km/h): "
            f"rejected_proposals={int(frame['speed_rejection_count'].sum())}, "
            f"steps_with_rejection={int(frame['speed_steps_with_rejection'].sum())}, "
            f"resolved_by_resampling={int(frame['speed_resample_accepted_count'].sum())}, "
            f"retry_exhausted={int(frame['speed_retry_exhausted_count'].sum())}, "
            f"regrounded={int(frame['speed_reground_count'].sum())}, "
            f"unfixable={int(frame['speed_reground_unfixable_count'].sum())}"
        )
    print(
        "Rollout end: "
        f"stopped_on_repetition={int(frame['stopped_on_repetition'].sum())}, "
        f"hit_max_transitions={int(frame['hit_max_transitions'].sum())}"
    )


if __name__ == "__main__":
    main()
