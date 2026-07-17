#!/usr/bin/env python3
"""Convert GRU Top-k output to the JSONL format consumed by lora-base-test.

Accepted JSONL fields per row:
  request_id
  predictions: [{"lora_id": 12, "probability": 0.7}, ...]

Accepted compact fields:
  request_id, topk_lora_ids, topk_probabilities

CSV input accepts one row per candidate with columns:
  request_id, lora_id, probability
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize_row(row: dict) -> dict | None:
    request_id = row.get("request_id", row.get("sample_id"))
    if request_id is None:
        return None

    predictions = row.get("predictions")
    if isinstance(predictions, list):
        normalized = []
        for item in predictions:
            if not isinstance(item, dict):
                continue
            if "lora_id" not in item:
                continue
            normalized.append(
                {
                    "lora_id": int(item["lora_id"]),
                    "probability": float(item.get("probability", 0.0)),
                }
            )
        if normalized:
            return {"request_id": int(request_id), "predictions": normalized}

    ids = row.get("topk_lora_ids")
    probabilities = row.get("topk_probabilities", [])
    if isinstance(ids, str):
        ids = json.loads(ids)
    if isinstance(probabilities, str) and probabilities:
        probabilities = json.loads(probabilities)
    if isinstance(ids, list):
        result = []
        for index, lora_id in enumerate(ids):
            probability = probabilities[index] if index < len(probabilities) else 0.0
            result.append({"lora_id": int(lora_id), "probability": float(probability)})
        if result:
            return {"request_id": int(request_id), "predictions": result}

    compact_lora_id = row.get("predicted_lora_id", row.get("lora_id"))
    if compact_lora_id is not None:
        return {
            "request_id": int(request_id),
            "predictions": [
                {
                    "lora_id": int(compact_lora_id),
                    "probability": float(row.get("probability", 1.0)),
                }
            ],
        }
    return None


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    grouped: dict[int, list[dict]] = {}
    for row in read_rows(args.input):
        normalized = normalize_row(row)
        if normalized is not None:
            grouped.setdefault(normalized["request_id"], []).extend(normalized["predictions"])
    rows = [
        {
            "request_id": request_id,
            "predictions": sorted(
                predictions,
                key=lambda item: item["probability"],
                reverse=True,
            ),
        }
        for request_id, predictions in sorted(grouped.items())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"converted_rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
