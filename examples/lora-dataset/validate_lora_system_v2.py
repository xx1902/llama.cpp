#!/usr/bin/env python3
"""Validate the materialize, chunk-cache, and prefetch results of system v2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.output_dir / "online_request_results.csv"
    data = pd.read_csv(path)
    online = data[data["benchmark"] == "online"].copy()
    baseline = data[data["benchmark"] == "baseline"][["request_id", "ttft_ms"]]
    baseline = baseline.rename(columns={"ttft_ms": "baseline_ttft_ms"})
    paired = online.merge(baseline, on="request_id", how="left")

    materialized = online[online["materialize_ok"] == 1]
    delta_hits = online[online["mode"] == "chunk_prefix_delta_materialize"]
    reconstructed = online[online["reconstruction_cos"] > 0]
    background = online[(online["prefetch_ms"] > 0) | (online["delta_background_ms"] > 0)]

    print(f"requests={len(online)}")
    print(f"modes=\n{online['mode'].value_counts().to_string()}")
    print(f"mean_ttft_ms={online['ttft_ms'].mean():.3f}")
    print(f"paired_change_pct={(online['ttft_ms'].mean() / paired['baseline_ttft_ms'].mean() - 1) * 100:.3f}")
    print(f"mean_chunk_hit_tokens={online['chunk_hit_tokens'].mean():.3f}")
    print(f"materialize_events={len(materialized)}")
    print(f"delta_hit_requests={len(delta_hits)}")
    print(f"prefetch_built={int(online['prefetch_built'].sum())}")
    if "prefetch_full_built" in online:
        print(f"prefetch_full_built={int(online['prefetch_full_built'].sum())}")
        print(f"prefetch_delta_built={int(online['prefetch_delta_built'].sum())}")
        print(f"prefetch_skipped_no_anchor={int(online['prefetch_skipped_no_anchor'].sum())}")
        print(f"prefetch_dropped_expired={int(online['prefetch_dropped_expired'].sum())}")
    if "delta_store_loaded" in online:
        print(f"delta_store_loaded={int(online['delta_store_loaded'].sum())}")
        print(f"delta_store_saved={int(online['delta_store_saved'].sum())}")
    print(f"max_host_delta_variants={int(online['host_delta_variants'].max())}")
    print(f"max_host_delta_mb={online['host_delta_bytes'].max() / 1024**2:.3f}")
    print(f"max_background_queue={int(online['background_queue_length'].max())}")
    print(f"total_background_overrun_ms={online['background_overrun_ms'].sum():.3f}")

    if len(materialized):
        print(f"mean_materialize_ms={materialized['materialize_ms'].mean():.3f}")
    if len(reconstructed):
        print(f"mean_reconstruction_cos={reconstructed['reconstruction_cos'].mean():.8f}")
        print(f"max_reconstruction_l2={reconstructed['reconstruction_l2'].max():.8f}")
    if len(background):
        work = background["prefetch_ms"] + background["delta_background_ms"]
        enough = background["idle_gap_ms"] >= work
        print(f"background_events={len(background)}")
        print(f"background_fit_rate={enough.mean():.4f}")

    required = {
        "materialize_ok": len(materialized) > 0,
        "host_delta_created": online["host_delta_variants"].max() > 0,
        "finite_reconstruction": len(reconstructed) > 0,
    }
    print("checks=")
    for name, ok in required.items():
        print(f"  {name}: {'PASS' if ok else 'NOT_OBSERVED'}")


if __name__ == "__main__":
    main()
