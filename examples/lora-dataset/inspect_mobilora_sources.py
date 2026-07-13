#!/usr/bin/env python3
"""Inspect the three local datasets used to build MobiLoRA-style workloads."""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Remove ShareGPT HTML wrappers and collapse whitespace."""
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {"p10": 0, "p50": 0, "p90": 0, "max": 0}
    return {
        "p10": int(np.percentile(values, 10)),
        "p50": int(np.percentile(values, 50)),
        "p90": int(np.percentile(values, 90)),
        "max": int(max(values)),
    }


def inspect_xsum(path: Path, sample_size: int) -> None:
    dataset = load_from_disk(str(path))
    indices = np.linspace(0, len(dataset) - 1, min(sample_size, len(dataset)), dtype=int)
    lengths = [len(dataset[int(index)]["document"]) for index in indices]

    print("\n===== XSum =====")
    print(dataset)
    print("columns:", dataset.column_names)
    print("document character lengths:", percentiles(lengths))

    row = dataset[0]
    print("sample id:", row["id"])
    print("document:", row["document"][:500].replace("\n", " "))
    print("summary:", row["summary"])


def inspect_sharegpt(path: Path, sample_size: int) -> None:
    dataset = load_from_disk(str(path))
    indices = np.linspace(0, len(dataset) - 1, min(sample_size, len(dataset)), dtype=int)

    turn_counts: list[int] = []
    character_counts: list[int] = []
    valid = 0

    for index in indices:
        conversations = dataset[int(index)].get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue

        cleaned = [
            clean_text(turn.get("value", ""))
            for turn in conversations
            if isinstance(turn, dict)
        ]
        cleaned = [text for text in cleaned if text]
        if not cleaned:
            continue

        valid += 1
        turn_counts.append(len(cleaned))
        character_counts.append(sum(map(len, cleaned)))

    print("\n===== ShareGPT52K =====")
    print(dataset)
    print("columns:", dataset.column_names)
    print(f"valid sampled conversations: {valid}/{len(indices)}")
    print("turn counts:", percentiles(turn_counts))
    print("conversation character lengths:", percentiles(character_counts))

    row = dataset[0]
    print("sample id:", row["id"])
    for turn in row["conversations"][:4]:
        speaker = turn.get("from", "unknown")
        print(f"{speaker}: {clean_text(turn.get('value', ''))[:300]}")


def inspect_lsapp(path: Path) -> None:
    csv_path = path / "df_start.csv"
    dataset = pd.read_csv(csv_path)

    print("\n===== LSApp =====")
    print("rows:", len(dataset))
    print("columns:", list(dataset.columns))
    print("users:", dataset["user_id"].nunique())
    print("sessions:", dataset["session_id"].nunique())
    print("apps:", dataset["app_name"].nunique())
    print("time range:", dataset["timestamp"].min(), "->", dataset["timestamp"].max())
    print("event types:")
    print(dataset["event_type"].value_counts().head(10).to_string())
    print("top apps:")
    print(dataset["app_name"].value_counts().head(25).to_string())
    print("sample rows:")
    print(dataset.head(8).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\ecnu_experiment\datasets"),
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    # ShareGPT contains multilingual text and occasional unusual Unicode
    # characters. Avoid failures under the default Windows console encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    inspect_xsum(args.dataset_root / "xsum_dataset", args.sample_size)
    inspect_sharegpt(args.dataset_root / "sharegpt52k_dataset", args.sample_size)
    inspect_lsapp(args.dataset_root / "lsapp")


if __name__ == "__main__":
    main()
