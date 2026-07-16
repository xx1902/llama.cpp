#!/usr/bin/env python3
"""Validate that original/5min/10min workloads share one frozen mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_mapping(directory: Path) -> list[dict]:
    return json.loads((directory / "app_lora_mapping.json").read_text(encoding="utf-8"))


def validate_directory(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    mapping = load_mapping(directory)
    groups = json.loads((directory / "lora_groups.json").read_text(encoding="utf-8"))
    entries = [entry for group in groups.values() for entry in group]
    errors: list[str] = []
    if len(mapping) != 87:
        errors.append(f"mapping count={len(mapping)}")
    if len(entries) != 87:
        errors.append(f"group adapter count={len(entries)}")
    if len({int(entry['lora_id']) for entry in entries}) != 87:
        errors.append("group LoRA IDs are not unique")
    if len({row['app_name'] for row in mapping}) != 87:
        errors.append("App names are not unique")
    if len({int(row['lora_id']) for row in mapping}) != 87:
        errors.append("App mapping is not one-to-one")
    for entry in entries:
        if not Path(entry["actual_adapter_path"]).is_file():
            errors.append(f"missing GGUF: {entry['actual_adapter_path']}")
    for row in read_jsonl(directory / "grouped" / "grouped_requests.jsonl"):
        expected = next(item for item in mapping if item["app_name"] == row["app_name"])
        if int(row["lora_id"]) != int(expected["lora_id"]):
            errors.append(f"grouped mapping mismatch: {row['app_name']}")
            break
    return {"summary": summary, "mapping": mapping, "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [validate_directory(directory) for directory in args.directories]
    reference = [
        (row["app_name"], int(row["lora_id"]), row["group_name"])
        for row in results[0]["mapping"]
    ]
    for directory, result in zip(args.directories, results):
        current = [
            (row["app_name"], int(row["lora_id"]), row["group_name"])
            for row in result["mapping"]
        ]
        if current != reference:
            result["errors"].append("mapping/grouping differs from the original workload")
        print(
            f"{directory}: slice={result['summary']['duration_slice_min']} "
            f"gru_events={result['summary']['gru_events']} "
            f"gru_samples={result['summary']['gru_samples']} "
            f"errors={len(result['errors'])}"
        )
        for error in result["errors"]:
            print(f"  ERROR: {error}")
    if any(result["errors"] for result in results):
        raise SystemExit(2)
    print("validation passed: three workloads share one 87-App/87-LoRA mapping")


if __name__ == "__main__":
    main()
