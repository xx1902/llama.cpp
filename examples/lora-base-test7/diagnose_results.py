#!/usr/bin/env python3
"""Explain TTFT regressions in a completed test7 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)


def inspect_cell(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    baseline = frame[frame["benchmark"] == "baseline"].copy()
    online = frame[frame["benchmark"] == "online"].copy()
    base = baseline[["request_id", "ttft_ms"]].rename(columns={"ttft_ms": "baseline_ttft_ms"})
    joined = online.merge(base, on="request_id", how="inner")
    speed = 100.0 * (joined["baseline_ttft_ms"] - joined["ttft_ms"]) / joined["baseline_ttft_ms"].clip(lower=1e-9)
    prefix = num(online, "prefix_tokens")
    chunk = num(online, "chunk_hit_tokens")
    same = num(online, "same_lora_variant_hit") > 0
    valid = num(online, "ttft_ms") > 0
    return {
        "requests": int(len(online)),
        "valid_requests": int(valid.sum()),
        "invalid_requests": int((~valid).sum()),
        "baseline_ttft_ms": float(num(baseline, "ttft_ms").mean()),
        "online_ttft_ms": float(num(online, "ttft_ms").mean()),
        "speedup_percent": float(speed.mean()),
        "p95_speedup_percent": float(np.percentile(speed, 95)) if len(speed) else 0.0,
        "request_hit_rate": float(np.mean((chunk > 0) | (same & (prefix > 0)))) if len(online) else 0.0,
        "token_hit_rate": float(chunk.sum() / max(1.0, prefix.sum())),
        "mean_prefix_tokens": float(prefix.mean()),
        "mean_chunk_hit_tokens": float(chunk.mean()),
        "max_cache_tokens": float(num(online, "physical_cache_tokens").max()),
        "max_cache_nodes": float(num(online, "cache_nodes").max()),
        "max_host_delta_mb": float(num(online, "host_delta_bytes").max() / (1024 * 1024)),
        "max_host_full_mb": float(num(online, "host_full_bytes").max() / (1024 * 1024)),
        "modes": ", ".join(f"{k}:{v}" for k, v in online["mode"].value_counts().items()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "real_100_matrix")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted((args.output_dir / "runs").glob("*/*/online_request_results.csv")):
        cell = inspect_cell(path)
        cell.update({"dataset": path.parts[-3], "strategy": path.parts[-2]})
        rows.append(cell)
    if not rows:
        raise SystemExit(f"no result files under {args.output_dir / 'runs'}")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "diagnostic_cells.csv", index=False)
    lines = [
        "# Test7 TTFT regression diagnosis",
        "",
        "This report is generated from paired request rows. `speedup_percent` is",
        "`100 * (baseline TTFT - online TTFT) / baseline TTFT`; negative values mean",
        "the cache path is slower than the matched full-prefill request.",
        "",
        "## Evidence by cell",
        "",
        "| Dataset | Strategy | Valid/total | Baseline TTFT (ms) | Online TTFT (ms) | Speedup | Request hit | Token hit | Mean prefix tokens | Mean chunk-hit tokens | Cache peak tokens | Nodes | Host delta MiB | Host full MiB | Modes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | {int(row['valid_requests'])}/{int(row['requests'])} | {row['baseline_ttft_ms']:.2f} | "
            f"{row['online_ttft_ms']:.2f} | {row['speedup_percent']:.1f}% | "
            f"{100*row['request_hit_rate']:.1f}% | {100*row['token_hit_rate']:.1f}% | "
            f"{row['mean_prefix_tokens']:.1f} | {row['mean_chunk_hit_tokens']:.1f} | "
            f"{row['max_cache_tokens']:.0f} | {row['max_cache_nodes']:.0f} | "
            f"{row['max_host_delta_mb']:.1f} | {row['max_host_full_mb']:.1f} | {row['modes']} |"
        )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Low request/token hit means the cache opportunity was not realized.",
        "- High prefix length with a small chunk hit means most of the prompt was still prefetched or rebuilt.",
        "- A negative result with high token hit indicates cache construction, copy, cross-LoRA conversion, or eviction overhead dominated the saved prefill.",
        "- Parallel traces can have exact shared prefixes but still pay for a child-LoRA full-KV build when no usable cross-LoRA delta is available.",
        "- Long XSum prompts amplify every uncached suffix and every eviction; this is a cost diagnosis, not evidence that the source article is invalid.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "D:\\anaconda\\envs\\qwen2.5_vl\\python.exe -B examples/lora-base-test7/diagnose_results.py",
        "```",
    ])
    (args.output_dir / "DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(frame), "output": str(args.output_dir / "DIAGNOSIS.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
