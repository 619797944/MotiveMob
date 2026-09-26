#!/usr/bin/env python3
"""Replace implausible long jumps with same-category POIs from user history.

Only the location token in gen_traj may change. Event timestamps, event count,
event order, and broad activity category are preserved.
"""

import argparse
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

EVENT_RE = re.compile(
    r"(?:^|,\s*|\n)(?:Activities at [^:\n]+:\s*)?"
    r"(.+?)\s+at\s+(\d{1,2}:\d{2}(?::\d{2})?)\.?(?=,|\n|$)"
)
REPORT_EVENT_RE = re.compile(
    r"([A-Za-z0-9&'() /.-]+#\d+)\s+at\s+(\d{1,2}:\d{2}(?::\d{2})?)"
)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


GLOBAL_LOC_MAP = load_pickle(DATA_DIR / "global_loc_map.pkl")
LOCATION_ACTIVITY_MAP = load_pickle(DATA_DIR / "location_activity_map.pkl")
MAP_LOC = {location_id: location_key for location_key, location_id in GLOBAL_LOC_MAP.items()}


def broad_category(location):
    base_name = location.split("#", 1)[0].strip()
    return LOCATION_ACTIVITY_MAP.get(base_name, "Unknown")


def coordinates(location):
    location_key = MAP_LOC[location.strip()]
    flattened = location_key.replace(" (", ", ").replace(")", "")
    fields = flattened.split(", ")
    return float(fields[-2]), float(fields[-1])


def time_slot(time_text, start_hour=4):
    fields = [int(value) for value in time_text.strip().strip(".").split(":")]
    hour, minute = fields[:2]
    seconds = hour * 3600 + minute * 60 + (fields[2] if len(fields) == 3 else 0)
    start_seconds = start_hour * 3600
    if seconds < start_seconds:
        seconds += 24 * 3600
    return (seconds - start_seconds) // 600


def geodistance(first, second):
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(value)) * 6371


def parse_history_events(trajectory):
    return [(match.group(1).strip(), match.group(2)) for match in EVENT_RE.finditer(str(trajectory))]


def parse_generated_events(trajectory):
    """Parse gen_traj with the same comma-buffer behavior as evaluation.py."""
    body = str(trajectory).split(": ")[-1]
    buffer = ""
    steps = []
    for piece in body.split(","):
        piece = piece.strip()
        if "00" in piece:
            steps.append((buffer + piece).strip())
            buffer = ""
        else:
            buffer += piece + ", "

    events = []
    for step in steps:
        fields = step.replace(".", "").split(" at ")
        if len(fields) < 2:
            continue
        location = fields[0].strip()
        time_text = fields[1].split(" ")[0]
        if ":" in time_text:
            events.append((location, time_text))
    return events


def generated_trajectory_is_safe_to_rewrite(trajectory):
    """Reject malformed model outputs that evaluation.py parses ambiguously."""
    body = str(trajectory).split(": ")[-1]
    buffer = ""
    for piece in body.split(","):
        piece = piece.strip()
        if "00" in piece:
            step = (buffer + piece).strip()
            buffer = ""
            if step.count(" at ") != 1:
                return False
        else:
            buffer += piece + ", "
    return not buffer.strip()


def parse_report_events(report):
    events = []
    for match in REPORT_EVENT_RE.finditer(str(report)):
        raw_location = match.group(1).strip()
        words = raw_location.split()
        location = None
        for start in range(len(words)):
            candidate = " ".join(words[start:])
            if candidate in MAP_LOC:
                location = candidate
                break
        if location is not None:
            events.append((location, match.group(2)))
    return events


def format_generated_trajectory(original, events):
    header = str(original).split(":", 1)[0]
    body = ", ".join(f"{location} at {time_text}" for location, time_text in events)
    return f"{header}: {body}"


def build_history_candidates(history, report=None):
    candidates = defaultdict(list)
    source_events = parse_history_events(history)
    if report is not None:
        source_events.extend(parse_report_events(report))
    for location, time_text in source_events:
        # evaluation.py drops any location containing the substring "Home" (e.g. "Home Appliance Store"); never introduce one here.
        if "home" in location.lower():
            continue
        try:
            point = coordinates(location)
        except (KeyError, ValueError, IndexError):
            continue
        candidates[broad_category(location)].append(
            (location, time_slot(time_text), point)
        )
    return candidates


def process_trajectory(gen_traj, history, threshold_km, time_weight, report=None):
    if not generated_trajectory_is_safe_to_rewrite(gen_traj):
        return gen_traj, 0, []
    events = parse_generated_events(gen_traj)
    if len(events) < 2:
        return gen_traj, 0, []

    history_candidates = build_history_candidates(history, report=report)
    processed = []
    changes = []
    previous_point = None

    for location, time_text in events:
        chosen_location = location
        if "home" in location.lower():
            processed.append((location, time_text))
            continue
        try:
            original_point = coordinates(location)
        except (KeyError, ValueError, IndexError):
            processed.append((location, time_text))
            previous_point = None
            continue

        if previous_point is not None:
            original_distance = geodistance(previous_point, original_point)
            category = broad_category(location)
            candidates = history_candidates.get(category, [])

            if original_distance > threshold_km and candidates:
                target_slot = time_slot(time_text)
                candidate_location, _, candidate_point = min(
                    candidates,
                    key=lambda item: (
                        geodistance(previous_point, item[2])
                        + time_weight * abs(item[1] - target_slot)
                    ),
                )
                candidate_distance = geodistance(previous_point, candidate_point)

                if candidate_distance < original_distance:
                    chosen_location = candidate_location
                    original_point = candidate_point
                    changes.append(
                        {
                            "time": time_text,
                            "old_location": location,
                            "new_location": candidate_location,
                            "old_jump_km": original_distance,
                            "new_jump_km": candidate_distance,
                            "category": category,
                        }
                    )

        processed.append((chosen_location, time_text))
        previous_point = original_point

    if not changes:
        return gen_traj, 0, []
    return format_generated_trajectory(gen_traj, processed), len(changes), changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-km", type=float, default=30.0)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional allowance: keep at most the first N generated events before post-processing.",
    )
    parser.add_argument(
        "--time-weight",
        type=float,
        default=0.01,
        help="Distance-cost km added per 10-minute difference from the historical visit time.",
    )
    parser.add_argument(
        "--include-report",
        action="store_true",
        help="Also use the usual POI/time pairs stated in the report as candidates.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"gen_traj", "history"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    changed_rows = 0
    changed_events = 0
    truncated_rows = 0
    removed_events = 0
    examples = []
    output_trajectories = []

    for row_index, row in frame.iterrows():
        gen_traj = row["gen_traj"]
        if not generated_trajectory_is_safe_to_rewrite(gen_traj):
            output_trajectories.append(gen_traj)
            continue
        if args.max_events is not None:
            if args.max_events < 1:
                raise ValueError("--max-events must be at least 1")
            parsed = parse_generated_events(gen_traj)
            if len(parsed) > args.max_events:
                truncated_rows += 1
                removed_events += len(parsed) - args.max_events
                gen_traj = format_generated_trajectory(gen_traj, parsed[: args.max_events])

        processed, count, changes = process_trajectory(
            gen_traj,
            row["history"],
            args.threshold_km,
            args.time_weight,
            report=row.get("report") if args.include_report else None,
        )
        output_trajectories.append(processed)
        if count:
            changed_rows += 1
            changed_events += count
            if len(examples) < 5:
                examples.append({"row": int(row_index), **changes[0]})

    frame["gen_traj"] = output_trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    print(f"rows={len(frame)}")
    print(f"changed_rows={changed_rows}")
    print(f"changed_events={changed_events}")
    print(f"truncated_rows={truncated_rows}")
    print(f"removed_events={removed_events}")
    print(f"output={output_path}")
    print("examples:")
    for example in examples:
        print(example)


if __name__ == "__main__":
    main()
