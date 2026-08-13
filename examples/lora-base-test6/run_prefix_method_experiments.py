#!/usr/bin/env python3
"""Run online TTFT experiments for the five prefix-boundary methods."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_results import (
    DATASET_LABELS,
    PREFIX_METHODS,
    bootstrap_mean_ci,
    read_jsonl,
    split_prefix,
)
from run_experiments import DEFAULT_BINARY, run_cell


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MATRIX = SCRIPT_DIR / "experiment_matrix.json"
DEFAULT_WORKLOADS_ROOT = SCRIPT_DIR.parent / "lora-base-datasets" / "output" / "real_30"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "real_30_matrix"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_method_workload(
    source_dir: Path,
    destination_dir: Path,
    method_id: str,
    request_limit: int,
) -> None:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)
    source_path = source_dir / "grouped" / "grouped_requests.jsonl"
    destination_path = destination_dir / "grouped" / "grouped_requests.jsonl"
    rows = read_jsonl(source_path)
    rows = rows[:request_limit]
    prefixes_by_context: dict[str, list[str]] = {}
    for row in rows:
        context_id = str(row.get("context_id", ""))
        prefixes_by_context.setdefault(context_id, []).append(
            str(row.get("common_prefix_text", ""))
        )

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        context_id = str(row.get("context_id", ""))
        text = str(row.get("common_prefix_text", ""))
        reuse_points = {
            len(prefix)
            for prefix in prefixes_by_context.get(context_id, [])
            if len(prefix) < len(text) and text.startswith(prefix)
        }
        chunks = split_prefix(row, method_id, reuse_points)
        updated = dict(row)
        # A large runtime step makes each precomputed boundary one node.
        updated["prefix_segments"] = [
            {"type": "prefix_method", "text": chunk} for chunk in chunks
        ]
        updated["prefix_segment_types"] = ["prefix_method"] * len(chunks)
        updated["prefix_method"] = method_id
        updated["prefix_method_chunk_count"] = len(chunks)
        output_rows.append(updated)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def summarize_method_run(manifest: dict[str, Any], method_id: str, method_label: str) -> dict[str, Any]:
    run_dir = Path(manifest["run_dir"])
    frame = pd.read_csv(run_dir / "online_request_results.csv")
    baseline = frame[frame["benchmark"] == "baseline"].copy()
    online = frame[frame["benchmark"] == "online"].copy()
    baseline_ttft = baseline[["request_id", "ttft_ms"]].rename(
        columns={"ttft_ms": "baseline_ttft_ms"}
    )
    requests = online.merge(baseline_ttft, on="request_id", how="inner")
    base = numeric(requests, "baseline_ttft_ms").to_numpy(dtype=float)
    current = numeric(requests, "ttft_ms").to_numpy(dtype=float)
    relative_speedup = np.where(base > 0, 100.0 * (base - current) / base, 0.0)
    ci_low, ci_high = bootstrap_mean_ci(
        relative_speedup,
        f"{manifest['dataset']}:{method_id}",
    )
    same_lora = numeric(online, "same_lora_variant_hit") > 0
    legacy_tokens = np.where(same_lora, numeric(online, "prefix_tokens"), 0.0)
    chunk_tokens = numeric(online, "chunk_hit_tokens").to_numpy(dtype=float)
    reused_tokens = np.maximum(chunk_tokens, legacy_tokens)
    prefix_tokens = numeric(online, "prefix_tokens").to_numpy(dtype=float)
    event_path = run_dir / "family_cache_events.csv"
    evictions = 0
    if event_path.is_file() and event_path.stat().st_size:
        events = pd.read_csv(event_path)
        if "action" in events:
            evictions = int((events["action"] == "family_evict").sum())

    return {
        "dataset": manifest["dataset"],
        "dataset_label": DATASET_LABELS.get(manifest["dataset"], manifest["dataset"]),
        "method_id": method_id,
        "method_label": method_label,
        "runtime_requests": int(len(online)),
        "runtime_request_hit_rate": float(np.mean(reused_tokens > 0)) if len(online) else 0.0,
        "runtime_token_hit_rate": float(reused_tokens.sum() / max(1.0, prefix_tokens.sum())),
        "runtime_ttft_speedup_percent": float(
            100.0 * (numeric(baseline, "ttft_ms").mean() - numeric(online, "ttft_ms").mean())
            / max(1e-9, numeric(baseline, "ttft_ms").mean())
        ),
        "runtime_ttft_ci95_low_percent": ci_low,
        "runtime_ttft_ci95_high_percent": ci_high,
        "runtime_ttft_ci95_half_percent": 0.5 * (ci_high - ci_low),
        "runtime_host_peak_mb": float(
            (numeric(online, "host_delta_bytes") + numeric(online, "host_full_bytes")).max()
            / (1024.0 * 1024.0)
        ),
        "runtime_peak_nodes": float(numeric(online, "cache_nodes").max()),
        "runtime_family_evictions": evictions,
        "runtime_reusable_prefix_tokens": float(reused_tokens.sum()),
        "run_elapsed_seconds": float(manifest.get("elapsed_seconds", 0.0)),
        "status": manifest.get("status", "unknown"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--workloads-root", type=Path, default=DEFAULT_WORKLOADS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-limit", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = read_json(args.matrix)
    datasets = list(matrix["datasets"])
    derived_root = args.output_dir / "prefix_method_workloads"
    run_root = args.output_dir
    rows: list[dict[str, Any]] = []
    common_options = dict(matrix["common_options"])
    strategy_options = {
        "system-v2": 1,
        "prefix-chunk-tokens": 4096,
        "system-chunk-tokens": 4096,
        "context-chunk-tokens": 4096,
        "cross-lora-policy": "deferred",
        "max-host-delta-mb": 256,
        "max-host-full-mb": 512,
        "prefetch-policy": "oracle",
        "prefetch-storage": "auto",
        "delta-store-policy": "none",
        "background-policy": "unlimited",
    }

    for method_id, method_label in PREFIX_METHODS:
        for dataset in datasets:
            source_dir = args.workloads_root / dataset
            destination_dir = derived_root / method_id / dataset
            build_method_workload(source_dir, destination_dir, method_id, args.request_limit)
            strategy = {
                "id": f"prefix_method_{method_id}",
                "label": f"Prefix method / {method_label}",
                "category": "prefix_method",
                "options": strategy_options,
            }
            manifest = run_cell(
                binary=args.binary,
                workload_dir=destination_dir,
                output_root=run_root,
                dataset=dataset,
                strategy=strategy,
                common_options=common_options,
                request_limit=args.request_limit,
                timeout_seconds=args.timeout_seconds,
                force=args.force,
            )
            if manifest.get("status") != "ok":
                raise RuntimeError(
                    f"prefix method cell failed: {method_id}/{dataset}: {manifest.get('error', '')}"
                )
            rows.append(summarize_method_run(manifest, method_id, method_label))

    output_path = args.output_dir / "prefix_method_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"prefix method experiments complete: rows={len(rows)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
