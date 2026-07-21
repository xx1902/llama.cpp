#!/usr/bin/env python3
"""Summarize online GRU and prefetch metrics emitted by lora-base-test2."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(number(row, key) for row in rows) if rows else 0.0


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * probability))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    csv_path = args.output_dir / "online_request_results.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    baseline = [row for row in rows if row.get("benchmark") == "baseline"]
    online = [row for row in rows if row.get("benchmark") == "online"]
    service = [row for row in online if row.get("prediction_source") == "service"]
    successful = [row for row in service if number(row, "prediction_service_ok") == 1]
    ready = [row for row in successful if number(row, "prediction_ready") == 1]
    query_ms = [number(row, "prediction_query_ms") for row in successful]
    inference_ms = [number(row, "prediction_inference_ms") for row in ready]
    transport_ms = [
        max(0.0, number(row, "prediction_query_ms") -
            number(row, "prediction_inference_ms"))
        for row in ready
    ]

    baseline_by_id = {row["request_id"]: row for row in baseline}
    paired = [row for row in online if row.get("request_id") in baseline_by_id]
    paired_baseline_ms = statistics.fmean(
        number(baseline_by_id[row["request_id"]], "ttft_ms") for row in paired
    ) if paired else 0.0
    paired_online_ms = mean(paired, "ttft_ms")

    summary = {
        "online_requests": len(online),
        "service_requests": len(service),
        "service_success_rate": len(successful) / len(service) if service else 0.0,
        "prediction_ready_requests": len(ready),
        "prediction_top1_accuracy": mean(ready, "prediction_top1_hit"),
        "prediction_topk_accuracy": mean(ready, "prediction_topk_hit"),
        "prediction_query_mean_ms": statistics.fmean(query_ms) if query_ms else 0.0,
        "prediction_query_p95_ms": percentile(query_ms, 0.95),
        "gru_inference_mean_ms": statistics.fmean(inference_ms) if inference_ms else 0.0,
        "gru_inference_p95_ms": percentile(inference_ms, 0.95),
        "http_and_serialization_mean_ms": (
            statistics.fmean(transport_ms) if transport_ms else 0.0
        ),
        "paired_baseline_ttft_ms": paired_baseline_ms,
        "paired_online_ttft_ms": paired_online_ms,
        "paired_ttft_change_rate": (
            paired_online_ms / paired_baseline_ms - 1.0 if paired_baseline_ms else 0.0
        ),
        "exact_prefix_hit_rate": mean(online, "exact_prefix_hit"),
        "prefetch_built": int(sum(number(row, "prefetch_built") for row in online)),
        "prefetch_work_ms": sum(number(row, "prefetch_ms") for row in online),
        "delta_background_work_ms": sum(
            number(row, "delta_background_ms") for row in online
        ),
    }
    output = args.output_dir / "gru_online_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={output}")


if __name__ == "__main__":
    main()
