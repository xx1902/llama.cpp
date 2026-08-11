#!/usr/bin/env python3
"""Aggregate test6 runs and generate reproducible plots and a Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "real_30_matrix"

DATASET_LABELS = {
    "msc_continuous": "MSC",
    "taskmaster_continuous": "Taskmaster",
    "lmsys_continuous": "LMSYS-33K",
    "opus100_parallel": "OPUS-100",
    "xsum_parallel": "XSum",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def percentile(values: pd.Series | np.ndarray, value: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, value)) if array.size else 0.0


def bootstrap_mean_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), float(values[0])
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def event_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "family_cache_events.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "cache_events": 0,
            "family_evictions": 0,
            "gpu_to_host_offloads": 0,
            "delta_demotions": 0,
        }
    frame = pd.read_csv(path)
    if frame.empty or "action" not in frame:
        return {
            "cache_events": 0,
            "family_evictions": 0,
            "gpu_to_host_offloads": 0,
            "delta_demotions": 0,
        }
    counts = frame["action"].value_counts()
    return {
        "cache_events": int(len(frame)),
        "family_evictions": int(counts.get("family_evict", 0)),
        "gpu_to_host_offloads": int(counts.get("gpu_full_to_host", 0)),
        "delta_demotions": int(counts.get("tier3_delta_demote", 0)),
    }


def tree_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "online_prefix_tree.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "final_tree_rows": 0,
            "final_gpu_variants": 0,
            "final_host_full_variants": 0,
            "final_host_delta_variants": 0,
            "final_delta_bytes": 0,
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "final_tree_rows": 0,
            "final_gpu_variants": 0,
            "final_host_full_variants": 0,
            "final_host_delta_variants": 0,
            "final_delta_bytes": 0,
        }
    residency = frame.get("variant_residency", pd.Series(dtype=str)).value_counts()
    return {
        "final_tree_rows": int(len(frame)),
        "final_gpu_variants": int(residency.get("gpu_full", 0)),
        "final_host_full_variants": int(residency.get("host_full", 0)),
        "final_host_delta_variants": int(residency.get("host_delta", 0)),
        "final_delta_bytes": float(numeric(frame, "delta_bytes").sum()),
    }


def summarize_run(
    manifest: dict[str, Any],
    strategy: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    run_dir = Path(manifest["run_dir"])
    frame = pd.read_csv(run_dir / "online_request_results.csv")
    baseline = frame[frame["benchmark"] == "baseline"].copy()
    online = frame[frame["benchmark"] == "online"].copy()
    if baseline.empty or online.empty:
        raise ValueError(f"missing baseline or online rows in {run_dir}")
    invalid_modes = online["mode"].astype(str).str.contains("failed", case=False, na=False)
    invalid_ttft = numeric(online, "ttft_ms") <= 0
    if invalid_modes.any() or invalid_ttft.any():
        invalid = online.loc[invalid_modes | invalid_ttft, ["request_id", "mode", "ttft_ms"]]
        raise ValueError(
            f"invalid online rows in {run_dir}: "
            f"{invalid.head(3).to_dict(orient='records')} (count={len(invalid)})"
        )

    baseline_ttft = baseline[["request_id", "ttft_ms"]].rename(
        columns={"ttft_ms": "baseline_ttft_ms"}
    )
    requests = online.merge(baseline_ttft, on="request_id", how="inner")
    requests["dataset"] = manifest["dataset"]
    requests["strategy_id"] = manifest["strategy_id"]
    requests["strategy_label"] = manifest["strategy_label"]
    requests["ttft_saved_ms"] = (
        numeric(requests, "baseline_ttft_ms") - numeric(requests, "ttft_ms")
    )
    requests["ttft_speedup_percent"] = np.where(
        numeric(requests, "baseline_ttft_ms") > 0,
        100.0 * requests["ttft_saved_ms"] / numeric(requests, "baseline_ttft_ms"),
        0.0,
    )

    chunk_hits = numeric(online, "chunk_hit_tokens")
    same_lora_hits = numeric(online, "same_lora_variant_hit") > 0
    legacy_hit_tokens = np.where(
        same_lora_hits,
        numeric(online, "prefix_tokens"),
        0.0,
    )
    reused_tokens = np.maximum(chunk_hits.to_numpy(), legacy_hit_tokens)
    prefix_tokens = numeric(online, "prefix_tokens").to_numpy()
    token_hit_rate = float(reused_tokens.sum() / max(1.0, prefix_tokens.sum()))
    request_hit_rate = float(np.mean(reused_tokens > 0))

    paired_saved = requests["ttft_saved_ms"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(
        paired_saved,
        f"{manifest['dataset']}:{manifest['strategy_id']}",
    )
    baseline_mean = float(numeric(baseline, "ttft_ms").mean())
    online_mean = float(numeric(online, "ttft_ms").mean())
    speedup = 100.0 * (baseline_mean - online_mean) / baseline_mean if baseline_mean else 0.0

    summary: dict[str, Any] = {
        "dataset": manifest["dataset"],
        "dataset_label": DATASET_LABELS.get(manifest["dataset"], manifest["dataset"]),
        "strategy_id": manifest["strategy_id"],
        "strategy_label": manifest["strategy_label"],
        "category": strategy.get("category", "unknown"),
        "chunk_tokens": strategy.get("chunk_tokens", strategy.get("options", {}).get("prefix-chunk-tokens")),
        "requests": int(len(online)),
        "run_elapsed_seconds": float(manifest.get("elapsed_seconds", 0.0)),
        "baseline_ttft_mean_ms": baseline_mean,
        "ttft_mean_ms": online_mean,
        "ttft_p50_ms": percentile(numeric(online, "ttft_ms"), 50),
        "ttft_p95_ms": percentile(numeric(online, "ttft_ms"), 95),
        "total_mean_ms": float(numeric(online, "total_ms").mean()),
        "prefix_mean_ms": float(numeric(online, "prefix_ms").mean()),
        "paired_ttft_saved_mean_ms": float(paired_saved.mean()),
        "paired_ttft_saved_ci95_low_ms": ci_low,
        "paired_ttft_saved_ci95_high_ms": ci_high,
        "ttft_speedup_percent": speedup,
        "prefix_token_hit_rate": token_hit_rate,
        "request_hit_rate": request_hit_rate,
        "same_lora_hit_rate": float(numeric(online, "same_lora_variant_hit").mean()),
        "cross_lora_match_rate": float(numeric(online, "cross_lora_prefix_match").mean()),
        "max_cache_nodes": float(numeric(online, "cache_nodes").max()),
        "max_cache_variants": float(numeric(online, "cache_variants").max()),
        "max_physical_cache_tokens": float(numeric(online, "physical_cache_tokens").max()),
        "max_host_delta_bytes": float(numeric(online, "host_delta_bytes").max()),
        "max_host_full_bytes": float(numeric(online, "host_full_bytes").max()),
        "max_host_delta_variants": float(numeric(online, "host_delta_variants").max()),
        "max_host_full_variants": float(numeric(online, "host_full_variants").max()),
        "delta_store_loaded": float(numeric(online, "delta_store_loaded").sum()),
        "delta_store_saved": float(numeric(online, "delta_store_saved").sum()),
        "delta_store_load_ms": float(numeric(online, "delta_store_load_ms").sum()),
        "delta_store_bytes_read": float(numeric(online, "delta_store_bytes_read").sum()),
        "delta_compressed_background": float(numeric(online, "delta_compressed_background").sum()),
        "delta_build_ok": float(numeric(online, "delta_build_ok").sum()),
        "materialize_ok": float(numeric(online, "materialize_ok").sum()),
        "reconstruction_checks": float(numeric(online, "reconstruction_checks").sum()),
        "reconstruction_cos_min": float(
            numeric(online, "reconstruction_cos_min").replace(0, np.nan).min()
            if (numeric(online, "reconstruction_cos_min").replace(0, np.nan).notna().any())
            else 0.0
        ),
        "reconstruction_l2_max": float(numeric(online, "reconstruction_l2_max").max()),
        "background_overrun_ms": float(numeric(online, "background_overrun_ms").sum()),
    }
    summary.update(event_metrics(run_dir))
    summary.update(tree_metrics(run_dir))

    baseline_requests = baseline.copy()
    baseline_requests["dataset"] = manifest["dataset"]
    baseline_requests["strategy_id"] = "full_prefill"
    baseline_requests["strategy_label"] = "Full prefill"
    return summary, requests, baseline_requests


def baseline_summary(
    dataset: str,
    baseline_requests: pd.DataFrame,
) -> dict[str, Any]:
    ttft = numeric(baseline_requests, "ttft_ms")
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "strategy_id": "full_prefill",
        "strategy_label": "Full prefill",
        "category": "baseline",
        "chunk_tokens": np.nan,
        "requests": int(len(baseline_requests)),
        "run_elapsed_seconds": float("nan"),
        "baseline_ttft_mean_ms": float(ttft.mean()),
        "ttft_mean_ms": float(ttft.mean()),
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "total_mean_ms": float(numeric(baseline_requests, "total_ms").mean()),
        "prefix_mean_ms": 0.0,
        "paired_ttft_saved_mean_ms": 0.0,
        "paired_ttft_saved_ci95_low_ms": 0.0,
        "paired_ttft_saved_ci95_high_ms": 0.0,
        "ttft_speedup_percent": 0.0,
        "prefix_token_hit_rate": 0.0,
        "request_hit_rate": 0.0,
        "same_lora_hit_rate": 0.0,
        "cross_lora_match_rate": 0.0,
        "max_cache_nodes": 0.0,
        "max_cache_variants": 0.0,
        "max_physical_cache_tokens": 0.0,
        "max_host_delta_bytes": 0.0,
        "max_host_full_bytes": 0.0,
        "max_host_delta_variants": 0.0,
        "max_host_full_variants": 0.0,
        "delta_store_loaded": 0.0,
        "delta_store_saved": 0.0,
        "delta_store_load_ms": 0.0,
        "delta_store_bytes_read": 0.0,
        "delta_compressed_background": 0.0,
        "delta_build_ok": 0.0,
        "materialize_ok": 0.0,
        "reconstruction_checks": 0.0,
        "reconstruction_cos_min": 0.0,
        "reconstruction_l2_max": 0.0,
        "background_overrun_ms": 0.0,
        "cache_events": 0,
        "family_evictions": 0,
        "gpu_to_host_offloads": 0,
        "delta_demotions": 0,
        "final_tree_rows": 0,
        "final_gpu_variants": 0,
        "final_host_full_variants": 0,
        "final_host_delta_variants": 0,
        "final_delta_bytes": 0.0,
    }


def collect_delta_quality(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_datasets: set[str] = set()
    for manifest in manifests:
        dataset = str(manifest["dataset"])
        if dataset in seen_datasets:
            continue
        path = Path(manifest["run_dir"]) / "delta_prefix_probe.csv"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["dataset"] = dataset
        frame["dataset_label"] = DATASET_LABELS.get(dataset, dataset)
        frames.append(frame)
        seen_datasets.add(dataset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_speedup_heatmap(summary: pd.DataFrame, figures_dir: Path) -> None:
    dataset_order = [dataset for dataset in DATASET_LABELS if dataset in set(summary["dataset"])]
    strategy_order = [
        strategy
        for strategy in summary["strategy_id"].drop_duplicates()
        if strategy != "full_prefill"
    ]
    pivot = summary.pivot(index="dataset", columns="strategy_id", values="ttft_speedup_percent")
    values = pivot.reindex(index=dataset_order, columns=strategy_order).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(5.0, float(np.max(np.abs(finite))) if finite.size else 5.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = plt.subplots(figsize=(14, 4.6), constrained_layout=True)
    image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_title("TTFT speedup relative to the matched full-prefill run")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Dataset")
    labels = (
        summary.drop_duplicates("strategy_id")
        .set_index("strategy_id")["strategy_label"]
        .to_dict()
    )
    ax.set_xticks(range(len(strategy_order)), [labels.get(item, item) for item in strategy_order], rotation=35, ha="right")
    ax.set_yticks(range(len(dataset_order)), [DATASET_LABELS[item] for item in dataset_order])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("TTFT speedup (%)")
    fig.savefig(figures_dir / "ttft_speedup_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_chunk_ablation(summary: pd.DataFrame, figures_dir: Path) -> None:
    chunk = summary[summary["category"] == "chunk"].copy()
    if chunk.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    metrics = [
        ("ttft_speedup_percent", "TTFT speedup (%)"),
        ("prefix_token_hit_rate", "Prefix-token hit rate"),
        ("ttft_p95_ms", "TTFT p95 (ms)"),
        ("max_physical_cache_tokens", "Peak physical cache tokens"),
    ]
    colors = plt.get_cmap("tab10").colors
    for dataset_index, (dataset, group) in enumerate(chunk.groupby("dataset", sort=False)):
        group = group.sort_values("chunk_tokens")
        for ax, (metric, label) in zip(axes.flat, metrics):
            values = group[metric].to_numpy(dtype=float)
            if metric == "prefix_token_hit_rate":
                values *= 100.0
                label = "Prefix-token hit rate (%)"
            ax.plot(
                group["chunk_tokens"],
                values,
                marker="o",
                linewidth=1.7,
                label=DATASET_LABELS.get(dataset, dataset),
                color=colors[dataset_index % len(colors)],
            )
            ax.set_xlabel("Fixed chunk size (tokens)")
            ax.set_ylabel(label)
            ax.set_xticks(sorted(chunk["chunk_tokens"].dropna().unique()))
            ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_title("Latency")
    axes[0, 1].set_title("Reuse")
    axes[1, 0].set_title("Tail latency")
    axes[1, 1].set_title("GPU cache pressure")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=min(5, len(labels)), frameon=False)
    fig.savefig(figures_dir / "prefix_chunk_ablation.png", bbox_inches="tight")
    plt.close(fig)


def save_storage_ablation(summary: pd.DataFrame, figures_dir: Path) -> None:
    storage_ids = [
        "full_prefill",
        "legacy_exact",
        "gpu_only_128",
        "gpu_host_full_128",
        "tiered_no_prefetch_128",
        "tiered_oracle_128",
        "tiered_disk_build_128",
        "tiered_disk_warm_128",
    ]
    storage = summary[summary["strategy_id"].isin(storage_ids)].copy()
    if storage.empty:
        return
    order = [item for item in storage_ids if item in set(storage["strategy_id"])]
    labels = (
        storage.drop_duplicates("strategy_id")
        .set_index("strategy_id")["strategy_label"]
        .to_dict()
    )
    datasets = [item for item in DATASET_LABELS if item in set(storage["dataset"])]
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.16
    for index, dataset in enumerate(datasets):
        group = storage[storage["dataset"] == dataset].set_index("strategy_id")
        speedup = [float(group.loc[item, "ttft_speedup_percent"]) if item in group.index else np.nan for item in order]
        hit_rate = [100.0 * float(group.loc[item, "prefix_token_hit_rate"]) if item in group.index else np.nan for item in order]
        offset = (index - (len(datasets) - 1) / 2) * width
        axes[0].bar(x + offset, speedup, width, label=DATASET_LABELS[dataset], color=colors[index])
        axes[1].bar(x + offset, hit_rate, width, label=DATASET_LABELS[dataset], color=colors[index])
    axes[0].axhline(0, color="0.35", linewidth=0.8)
    axes[0].set_ylabel("TTFT speedup (%)")
    axes[0].set_title("Storage-policy latency ablation")
    axes[1].set_ylabel("Prefix-token hit rate (%)")
    axes[1].set_title("Storage-policy reuse ablation")
    for ax in axes:
        ax.set_xticks(x, [labels.get(item, item) for item in order], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=min(5, len(datasets)), frameon=False, loc="best")
    fig.savefig(figures_dir / "storage_ablation.png", bbox_inches="tight")
    plt.close(fig)


def save_resource_tradeoff(summary: pd.DataFrame, figures_dir: Path) -> None:
    cache = summary[summary["strategy_id"] != "full_prefill"].copy()
    if cache.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, (dataset, group) in enumerate(cache.groupby("dataset", sort=False)):
        host_mb = (group["max_host_delta_bytes"] + group["max_host_full_bytes"]) / (1024.0 * 1024.0)
        sizes = 30.0 + np.clip(group["prefix_token_hit_rate"].to_numpy(dtype=float), 0, 1) * 130.0
        ax.scatter(
            group["max_physical_cache_tokens"],
            host_mb,
            s=sizes,
            alpha=0.78,
            label=DATASET_LABELS.get(dataset, dataset),
            color=colors[index % len(colors)],
            edgecolor="0.25",
            linewidth=0.4,
        )
    ax.set_title("Cache resource trade-off (marker size = prefix hit rate)")
    ax.set_xlabel("Peak physical GPU cache tokens")
    ax.set_ylabel("Peak host cache (MiB)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "storage_resource_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def save_delta_quality(delta: pd.DataFrame, figures_dir: Path) -> None:
    if delta.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, (dataset, group) in enumerate(delta.groupby("dataset", sort=False)):
        axes[0].scatter(
            numeric(group, "prefix_tokens"),
            100.0 * numeric(group, "logical_saved_rate"),
            label=DATASET_LABELS.get(dataset, dataset),
            color=colors[index],
            alpha=0.8,
        )
        axes[1].scatter(
            numeric(group, "kv_l2"),
            numeric(group, "kv_cos"),
            label=DATASET_LABELS.get(dataset, dataset),
            color=colors[index],
            alpha=0.8,
        )
    axes[0].set_title("Q8 KV-delta compression")
    axes[0].set_xlabel("Shared prefix tokens")
    axes[0].set_ylabel("Logical bytes saved (%)")
    axes[1].set_title("Cross-LoRA KV similarity")
    axes[1].set_xlabel("KV normalized L2")
    axes[1].set_ylabel("KV cosine similarity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(figures_dir / "delta_quality.png", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> list[str]:
    headers = [label for _, label, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if fmt != "text" else "---" for _, _, fmt in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for key, _, fmt in columns:
            value = row.get(key, "")
            if fmt == "text":
                values.append(str(value).replace("|", "\\|"))
            elif fmt == "int":
                values.append(str(int(round(float(value)))))
            elif fmt == "float" and not math.isfinite(float(value)):
                values.append("n/a")
            elif fmt == "pct":
                values.append(f"{100.0 * float(value):.1f}%")
            elif fmt == "pct_value":
                values.append(f"{float(value):.1f}%")
            elif fmt == "mb":
                values.append(f"{float(value) / (1024.0 * 1024.0):.1f}")
            else:
                values.append(f"{float(value):.2f}")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    snapshot: dict[str, Any],
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> None:
    strategy_meta = {item["id"]: item for item in snapshot["strategies"]}
    best_rows = []
    for dataset, group in summary[summary["strategy_id"] != "full_prefill"].groupby("dataset", sort=False):
        best = group.loc[group["ttft_mean_ms"].idxmin()].copy()
        best_rows.append(best)
    best = pd.DataFrame(best_rows)
    if not best.empty:
        best["saved_ci95_ms"] = best.apply(
            lambda row: (
                f"[{row['paired_ttft_saved_ci95_low_ms']:.2f}, "
                f"{row['paired_ttft_saved_ci95_high_ms']:.2f}]"
            ),
            axis=1,
        )
        best["evidence"] = best.apply(
            lambda row: (
                "改善"
                if row["paired_ttft_saved_ci95_low_ms"] > 0
                else "回退"
                if row["paired_ttft_saved_ci95_high_ms"] < 0
                else "区间跨 0"
            ),
            axis=1,
        )

    storage_ids = [
        "full_prefill",
        "legacy_exact",
        "gpu_only_128",
        "gpu_host_full_128",
        "tiered_no_prefetch_128",
        "tiered_oracle_128",
        "tiered_disk_build_128",
        "tiered_disk_warm_128",
    ]
    storage = summary[summary["strategy_id"].isin(storage_ids)].copy()
    storage_mean = (
        storage.groupby(["strategy_id", "strategy_label"], as_index=False)
        .agg(
            run_elapsed_seconds=("run_elapsed_seconds", "mean"),
            ttft_mean_ms=("ttft_mean_ms", "mean"),
            ttft_p95_ms=("ttft_p95_ms", "mean"),
            ttft_speedup_percent=("ttft_speedup_percent", "mean"),
            prefix_token_hit_rate=("prefix_token_hit_rate", "mean"),
            max_physical_cache_tokens=("max_physical_cache_tokens", "mean"),
            max_host_delta_bytes=("max_host_delta_bytes", "mean"),
            max_host_full_bytes=("max_host_full_bytes", "mean"),
            family_evictions=("family_evictions", "sum"),
            delta_store_loaded=("delta_store_loaded", "sum"),
            delta_store_saved=("delta_store_saved", "sum"),
        )
    )
    storage_mean["order"] = storage_mean["strategy_id"].map(
        {item: index for index, item in enumerate(storage_ids)}
    )
    storage_mean = storage_mean.sort_values("order")

    chunk = summary[summary["category"] == "chunk"].copy()
    chunk = chunk.sort_values(["dataset", "chunk_tokens"])
    disk = summary[summary["category"] == "disk"].copy()

    def one_row(dataset: str, strategy_id: str) -> pd.Series | None:
        rows = summary[
            (summary["dataset"] == dataset)
            & (summary["strategy_id"] == strategy_id)
        ]
        return None if rows.empty else rows.iloc[0]

    interpretation: list[str] = []
    continuous_parts: list[str] = []
    for dataset, label in [
        ("msc_continuous", "MSC"),
        ("taskmaster_continuous", "Taskmaster"),
    ]:
        row = one_row(dataset, "gpu_host_full_128")
        if row is not None:
            continuous_parts.append(
                f"{label} {row['ttft_speedup_percent']:.1f}% "
                f"(95% CI [{row['paired_ttft_saved_ci95_low_ms']:.1f}, "
                f"{row['paired_ttft_saved_ci95_high_ms']:.1f}] ms)"
            )
    if continuous_parts:
        interpretation.append(
            "- 连续增长场景中，简单的 GPU + host-full / 128 路径仍有收益："
            + "；".join(continuous_parts)
            + "。完整 delta 分层并未延续这一收益。"
        )

    lmsys_rows = summary[
        (summary["dataset"] == "lmsys_continuous")
        & (summary["strategy_id"] != "full_prefill")
    ]
    if not lmsys_rows.empty:
        row = lmsys_rows.loc[lmsys_rows["ttft_mean_ms"].idxmin()]
        interpretation.append(
            f"- LMSYS-33K 的最佳非 full-prefill 策略仍回退 "
            f"{abs(row['ttft_speedup_percent']):.1f}%（{row['strategy_label']}）；"
            "普通短聊天中的复用量不足以覆盖缓存管理成本。"
        )

    opus_rows = chunk[chunk["dataset"] == "opus100_parallel"]
    if not opus_rows.empty:
        row = opus_rows.loc[opus_rows["ttft_mean_ms"].idxmin()]
        interpretation.append(
            f"- OPUS-100 是最适合并列复用的 workload：chunk={int(row['chunk_tokens'])} "
            f"时 token 命中率 {100.0 * row['prefix_token_hit_rate']:.1f}%，"
            f"TTFT 改善 {row['ttft_speedup_percent']:.1f}%，且没有 family eviction。"
        )

    xsum_rows = chunk[chunk["dataset"] == "xsum_parallel"]
    if not xsum_rows.empty:
        row = xsum_rows.loc[xsum_rows["ttft_mean_ms"].idxmin()]
        interpretation.append(
            f"- XSum 说明高命中率不是充分条件：表现最好的 chunk={int(row['chunk_tokens'])} "
            f"达到 {100.0 * row['prefix_token_hit_rate']:.1f}% token 命中率，"
            f"但 TTFT 仍回退 {abs(row['ttft_speedup_percent']):.1f}%；"
            "长 prefix 的 materialize、variant 构建和迁移成本抵消了 prefill 节省。"
        )

    no_prefetch = summary[summary["strategy_id"] == "tiered_no_prefetch_128"]
    if not no_prefetch.empty:
        regressions = int((no_prefetch["ttft_speedup_percent"] < 0).sum())
        interpretation.append(
            f"- Tiered no-prefetch / 128 在 {regressions}/{len(no_prefetch)} 个数据集上回退；"
            "oracle 仅表示下一 LoRA 完全已知的上界，不能视为可部署收益。"
        )

    if not disk.empty:
        saved = int(round(disk["delta_store_saved"].sum()))
        loaded = int(round(disk["delta_store_loaded"].sum()))
        load_ms = float(disk["delta_store_load_ms"].sum())
        interpretation.append(
            f"- 磁盘层实际保存 {saved} 个、加载 {loaded} 个 delta，累计加载 "
            f"{load_ms:.1f} ms；warm-load 只在部分数据集降低 cold-build TTFT，"
            "没有形成稳定的端到端优势。OPUS-100 保存/加载均为 0，不能据其结果声称磁盘层有效。"
        )

    if not delta.empty:
        interpretation.append(
            "- OPUS-100/XSum 的离线 KV delta cosine 较高，但 XSum 在线 TTFT 仍回退；"
            "可压缩性或重建精度不能替代端到端延迟测量。"
        )

    lines = [
        "# Test6 真实数据集三级缓存与前缀切分实验",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 实验目的",
        "",
        "本实验在同一套真实请求上比较无缓存 full-prefill、旧版 exact-prefix、GPU-only、GPU + host full，以及包含 GPU full KV、host full/Q8 delta 和磁盘 delta store 的分层策略。连续数据用于观察逐轮增长和 LoRA 返回，并列数据用于观察同一 prefix 下的跨 LoRA 复用。",
        "",
        "> `oracle` 策略使用下一请求的真实 LoRA 作为预取上界，不代表线上可达到的预测精度。物理 GGUF LoRA 仅用于系统切换实验，并未针对这些逻辑任务训练。",
        "",
        "## 实验配置",
        "",
        f"- 数据集：{', '.join(DATASET_LABELS.get(item, item) for item in snapshot['selected_datasets'])}",
        f"- 每个 workload 请求数：{snapshot['request_limit']}",
        f"- `n_ctx`：{snapshot['common_options']['n-ctx']}；GPU cache token 上限：{snapshot['common_options']['max-cache-tokens']}",
        f"- cache node / variant 上限：{snapshot['common_options']['max-cache-nodes']} / {snapshot['common_options']['max-cache-variants']}",
        f"- host delta / full 上限：256 MiB / 512 MiB（分层策略）",
        f"- 生成 token：{snapshot['common_options']['n-predict']}；delta 执行：`{snapshot['common_options']['delta-execution']}`",
        "- 正数 TTFT speedup 表示优于同一次运行内、同请求的 full-prefill；95% 区间由请求级配对 bootstrap 得到。",
        "- 存储汇总中的 Full-prefill 是 legacy cell 内的逐请求参考测量，不是额外运行的 cell；因此其 wall time 标为 n/a。",
        f"- 有效性校验：{len(summary[summary['strategy_id'] != 'full_prefill'])} 个策略 cell 均完成；"
        "任何 `failed` 模式或 TTFT <= 0 的在线行都会使 cell 失败。",
        "",
        "## 策略",
        "",
        "| ID | 策略 | 类别 | 说明 |",
        "|---|---|---|---|",
    ]
    for strategy_id in snapshot["selected_strategies"]:
        item = strategy_meta[strategy_id]
        options = item.get("options", {})
        description = (
            f"system-v2={options.get('system-v2')}, chunk={options.get('prefix-chunk-tokens')}, "
            f"cross-LoRA={options.get('cross-lora-policy')}, prefetch={options.get('prefetch-policy')}, "
            f"disk={options.get('delta-store-policy')}"
        )
        lines.append(
            f"| `{strategy_id}` | {item['label']} | {item.get('category', '')} | {description} |"
        )

    lines.extend(
        [
            "",
            "## 主要结果",
            "",
            "### 各数据集最低平均 TTFT 的非 full-prefill 策略",
            "",
        ]
    )
    if not best.empty:
        lines.extend(
            markdown_table(
                best,
                [
                    ("dataset_label", "数据集", "text"),
                    ("strategy_label", "策略", "text"),
                    ("run_elapsed_seconds", "平均 cell wall time (s)", "float"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("ttft_p95_ms", "p95 TTFT (ms)", "float"),
                    ("ttft_speedup_percent", "相对 full-prefill", "pct_value"),
                    ("prefix_token_hit_rate", "prefix token 命中率", "pct"),
                    ("saved_ci95_ms", "配对节省 95% CI (ms)", "text"),
                    ("evidence", "证据", "text"),
                ],
            )
        )

    lines.extend(["", "![TTFT speedup](figures/ttft_speedup_heatmap.png)", ""])
    lines.extend(["### 存储策略汇总（五个数据集等权平均）", ""])
    if not storage_mean.empty:
        lines.extend(
            markdown_table(
                storage_mean,
                [
                    ("strategy_label", "策略", "text"),
                    ("run_elapsed_seconds", "平均 cell wall time (s)", "float"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("ttft_p95_ms", "平均 p95 (ms)", "float"),
                    ("ttft_speedup_percent", "平均 speedup", "pct_value"),
                    ("prefix_token_hit_rate", "命中率", "pct"),
                    ("max_physical_cache_tokens", "GPU token 峰值", "float"),
                    ("max_host_delta_bytes", "host delta 峰值 (MiB)", "mb"),
                    ("max_host_full_bytes", "host full 峰值 (MiB)", "mb"),
                    ("family_evictions", "淘汰次数", "int"),
                ],
            )
        )
    lines.extend(
        [
            "",
            "![Storage ablation](figures/storage_ablation.png)",
            "",
            "![Storage resource tradeoff](figures/storage_resource_tradeoff.png)",
            "",
            "### Prefix chunk 大小对比",
            "",
        ]
    )
    if not chunk.empty:
        lines.extend(
            markdown_table(
                chunk,
                [
                    ("dataset_label", "数据集", "text"),
                    ("chunk_tokens", "chunk", "int"),
                    ("run_elapsed_seconds", "cell wall time (s)", "float"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("ttft_p95_ms", "p95 (ms)", "float"),
                    ("ttft_speedup_percent", "speedup", "pct_value"),
                    ("prefix_token_hit_rate", "token 命中率", "pct"),
                    ("max_physical_cache_tokens", "GPU token 峰值", "float"),
                    ("family_evictions", "淘汰", "int"),
                ],
            )
        )
    lines.extend(["", "![Prefix chunk ablation](figures/prefix_chunk_ablation.png)", ""])

    lines.extend(["### 磁盘 delta store", ""])
    if disk.empty:
        lines.append("本次结果中没有完成磁盘策略运行。")
    else:
        lines.extend(
            markdown_table(
                disk.sort_values(["dataset", "strategy_id"]),
                [
                    ("dataset_label", "数据集", "text"),
                    ("strategy_label", "策略", "text"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("delta_store_saved", "保存 delta", "int"),
                    ("delta_store_loaded", "加载 delta", "int"),
                    ("delta_store_load_ms", "加载耗时 (ms)", "float"),
                ],
            )
        )

    lines.extend(["", "### 跨 LoRA KV delta 质量", ""])
    if delta.empty:
        lines.append("连续 workload 没有 exact-prefix pair；本次没有可用的离线 delta quality 行。")
    else:
        delta_summary = (
            delta.groupby(["dataset", "dataset_label"], as_index=False)
            .agg(
                pairs=("pair_name", "count"),
                kv_cos=("kv_cos", "mean"),
                kv_l2=("kv_l2", "mean"),
                logical_saved_rate=("logical_saved_rate", "mean"),
                delta_build_ok=("delta_build_ok", "sum"),
            )
        )
        lines.extend(
            markdown_table(
                delta_summary,
                [
                    ("dataset_label", "数据集", "text"),
                    ("pairs", "pair", "int"),
                    ("kv_cos", "平均 cosine", "float"),
                    ("kv_l2", "平均 L2", "float"),
                    ("logical_saved_rate", "逻辑节省率", "pct"),
                    ("delta_build_ok", "构建成功", "int"),
                ],
            )
        )
        lines.extend(["", "![Delta quality](figures/delta_quality.png)"])

    lines.extend(
        [
            "",
            "## 如何解释这些结果",
            "",
            *interpretation,
            "",
            "## 局限性",
            "",
            "1. 每个数据集只有 30 条请求，本报告是小样本系统实验，不是最终统计结论。",
            "2. 每个矩阵 cell 默认只运行一次；请求级 bootstrap 反映 trace 内变化，不包含跨进程、跨温度或跨机器方差。",
            "3. Full-prefill 在每个进程中先执行，首请求可能包含额外 warm-up；比较使用同进程配对 baseline，但仍建议后续增加重复次数和随机化顺序。",
            "4. 当前 GGUF LoRA 与逻辑任务不匹配，因此只能验证缓存、切换和 KV 重建行为，不能据此比较生成质量。",
            "5. XSum 的 QA 和 headline 使用摘要作为代理 reference；本实验未计算 ROUGE、BLEU 或人工质量指标。",
            "6. test6 复用 test4 system-v2 运行时；它的固定切分是在节点建立后再按容量淘汰，尚不是基于边际收益的动态 admission。",
            "",
            "## 原始结果",
            "",
            "- `aggregate_results.csv`：每个 dataset/strategy 的汇总指标。",
            "- `request_results.csv`：请求级 online 与配对 full-prefill 指标。",
            "- `delta_quality.csv`：OPUS-100/XSum exact-prefix pair 的离线 delta 质量。",
            "- `runs/<dataset>/<strategy>/`：运行日志、完整 CSV、缓存树和参数快照。",
        ]
    )
    if failures:
        lines.extend(["", "## 未完成运行", ""])
        for failure in failures:
            lines.append(
                f"- `{failure.get('dataset')}/{failure.get('strategy_id')}`: {failure.get('error', 'unknown error')}"
            )

    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_path = args.output_dir / "matrix_snapshot.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"matrix snapshot not found: {snapshot_path}")
    snapshot = read_json(snapshot_path)
    strategy_by_id = {item["id"]: item for item in snapshot["strategies"]}

    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted((args.output_dir / "runs").glob("*/*/run_manifest.json")):
        manifest = read_json(path)
        if manifest.get("status") == "ok":
            manifests.append(manifest)
        else:
            failures.append(manifest)
    if not manifests:
        raise ValueError(f"no successful test6 runs found under {args.output_dir / 'runs'}")

    summaries: list[dict[str, Any]] = []
    request_frames: list[pd.DataFrame] = []
    baseline_by_dataset: dict[str, pd.DataFrame] = {}
    for manifest in manifests:
        strategy = strategy_by_id[manifest["strategy_id"]]
        summary, requests, baseline = summarize_run(manifest, strategy)
        summaries.append(summary)
        request_frames.append(requests)
        if manifest["strategy_id"] == "legacy_exact":
            baseline_by_dataset[manifest["dataset"]] = baseline

    for dataset, baseline in baseline_by_dataset.items():
        summaries.append(baseline_summary(dataset, baseline))

    summary_frame = pd.DataFrame(summaries)
    dataset_order = {item: index for index, item in enumerate(snapshot["selected_datasets"])}
    strategy_order = {"full_prefill": 0}
    strategy_order.update(
        {item: index + 1 for index, item in enumerate(snapshot["selected_strategies"])}
    )
    summary_frame["dataset_order"] = summary_frame["dataset"].map(dataset_order)
    summary_frame["strategy_order"] = summary_frame["strategy_id"].map(strategy_order)
    summary_frame = summary_frame.sort_values(["dataset_order", "strategy_order"])
    summary_frame.drop(columns=["dataset_order", "strategy_order"]).to_csv(
        args.output_dir / "aggregate_results.csv", index=False
    )

    requests = pd.concat(request_frames, ignore_index=True)
    requests.to_csv(args.output_dir / "request_results.csv", index=False)
    delta = collect_delta_quality(manifests)
    delta.to_csv(args.output_dir / "delta_quality.csv", index=False)

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()
    save_speedup_heatmap(summary_frame, figures_dir)
    save_chunk_ablation(summary_frame, figures_dir)
    save_storage_ablation(summary_frame, figures_dir)
    save_resource_tradeoff(summary_frame, figures_dir)
    save_delta_quality(delta, figures_dir)
    write_report(args.output_dir, snapshot, summary_frame, delta, failures)

    print(
        f"test6 analysis complete: runs={len(manifests)} "
        f"aggregate_rows={len(summary_frame)} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
