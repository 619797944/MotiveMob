"""Merge inference.py's per-shard output CSVs into one file for evaluation.py.

inference.py writes one CSV per shard (default: out_shard{shard_id}.csv in the
cwd, or shard_{shard_id}.csv inside --csv_out if that was passed) and never
merges them itself - evaluation.py expects a single --input file (merged.csv
by default). Run this after all shards have finished.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_inference_shards(args: argparse.Namespace) -> None:
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
    merged.to_csv(args.output_csv, index=False)
    print(f"Merged {len(csv_paths)} shards, rows={len(merged):,}")
    print(f"Saved merged output to {args.output_csv}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dir", type=Path, default=Path("."),
                         help="directory holding inference.py's shard CSVs")
    parser.add_argument("--shard_glob", type=str, default="out_shard*.csv",
                         help="glob for shard files; use 'shard_*.csv' if inference.py "
                              "was run with --csv_out <dir>")
    parser.add_argument("--output_csv", type=Path, default=Path("merged.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    merge_inference_shards(_parse_args())
