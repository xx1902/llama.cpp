#!/usr/bin/env python3
"""Build one comparison table from lora-base-test2 ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        result = float(row.get(key, default))
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(value(row, key) for row in rows) if rows else 0.0


def total(rows: list[dict[str, str]], key: str) -> float:
    return sum(value(row, key) for row in rows)


KNOWN_CASES = {
    "prefix_only", "deferred_delta",
    "no_prediction", "gru_file", "gru_online", "oracle",
    "gpu_exact_only", "gpu_plus_host_full", "host_anchor_delta", "gru_gpu_ready",
    "recency", "frequency", "frequency_recency", "frequency_prediction",
    "full_latency", "full_density",
    "random", "semantic", "transition", "hybrid",
}


def parse_name(directory: Path) -> tuple[str, str, str, int]:
    prefix = "output_ablation_"
    name = directory.name[len(prefix):] if directory.name.startswith(prefix) else directory.name
    parts = name.split("_", 1)
    suite = parts[0]
    remainder = parts[1] if len(parts) > 1 else "unknown"
    case = remainder
    suffix = ""
    for candidate in sorted(KNOWN_CASES, key=len, reverse=True):
        if remainder == candidate or remainder.startswith(candidate + "_"):
            case = candidate
            suffix = remainder[len(candidate):].lstrip("_")
            break
    repeat = 0
    match = re.search(r"(?:^|_)r(\d+)$", suffix)
    if match:
        repeat = int(match.group(1))
        suffix = suffix[:match.start()].rstrip("_")
    return suite, case, suffix, repeat


def summarize(directory: Path) -> dict[str, object] | None:
    rows = read_csv(directory / "online_request_results.csv")
    deltas = read_csv(directory / "delta_prefix_probe.csv")
    family_events = read_csv(directory / "family_cache_events.csv")
    if not rows and not deltas:
        return None

    suite, case, run_tag, repeat = parse_name(directory)
    baseline = [row for row in rows if row.get("benchmark") == "baseline"]
    online = [row for row in rows if row.get("benchmark") == "online"]
    baseline_by_id = {row.get("request_id"): row for row in baseline}
    paired = [row for row in online if row.get("request_id") in baseline_by_id]
    baseline_ttft = statistics.fmean(
        value(baseline_by_id[row["request_id"]], "ttft_ms") for row in paired
    ) if paired else 0.0
    online_ttft = mean(paired, "ttft_ms")
    ready = [row for row in online if value(row, "prediction_ready") == 1]
    valid_delta = [row for row in deltas if row.get("status") == "ok"]

    return {
        "suite": suite,
        "case": case,
        "run_tag": run_tag,
        "repeat": repeat,
        "output_dir": str(directory),
        "requests": len(online),
        "baseline_ttft_ms": baseline_ttft,
        "online_ttft_ms": online_ttft,
        "ttft_change_rate": online_ttft / baseline_ttft - 1.0 if baseline_ttft else 0.0,
        "full_prefix_hit_rate": mean(online, "exact_prefix_hit"),
        "any_chunk_prefix_hit_rate": (
            sum(value(row, "chunk_hit_tokens") > 0 for row in online) / len(online)
            if online else 0.0
        ),
        "mean_chunk_hit_tokens": mean(online, "chunk_hit_tokens"),
        "total_chunk_hit_tokens": total(online, "chunk_hit_tokens"),
        "peak_host_full_mb": max(
            (value(row, "host_full_bytes") for row in online), default=0.0
        ) / 1024.0 / 1024.0,
        "peak_host_delta_mb": max(
            (value(row, "host_delta_bytes") for row in online), default=0.0
        ) / 1024.0 / 1024.0,
        "final_delta_compression_rate": (
            value(online[-1], "host_delta_compression_rate") if online else 0.0
        ),
        "delta_candidates": int(total(online, "delta_candidates_marked")),
        "delta_compressed": int(total(online, "delta_compressed_background")),
        "delta_background_s": total(online, "delta_background_ms") / 1000.0,
        "prefetch_built": int(total(online, "prefetch_built")),
        "prefetch_work_s": total(online, "prefetch_ms") / 1000.0,
        "prediction_ready": len(ready),
        "prediction_top1_accuracy": mean(ready, "prediction_top1_hit"),
        "prediction_topk_accuracy": mean(ready, "prediction_topk_hit"),
        "prediction_query_ms": mean(ready, "prediction_query_ms"),
        "gru_inference_ms": mean(ready, "prediction_inference_ms"),
        "delta_pairs_ok": len(valid_delta),
        "delta_kv_cos": mean(valid_delta, "kv_cos"),
        "delta_kv_l2": mean(valid_delta, "kv_l2"),
        "delta_saved_rate": mean(valid_delta, "logical_saved_rate"),
        "family_events": len(family_events),
        "family_evictions": sum(
            row.get("action") in {"evict", "release", "demote"}
            for row in family_events
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2"),
    )
    parser.add_argument("--pattern", default="output_ablation_*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = [
        result
        for directory in sorted(args.root.glob(args.pattern))
        if directory.is_dir() and (result := summarize(directory)) is not None
    ]
    if not summaries:
        raise SystemExit("no completed ablation outputs found")

    output = args.output or args.root / "system_ablation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"experiments={len(summaries)}")
    print(f"csv={output}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
