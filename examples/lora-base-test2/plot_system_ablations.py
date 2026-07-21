#!/usr/bin/env python3
"""Create colored paper-ready figures from system_ablation_summary.csv."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTE = ["#007C83", "#2D6CDF", "#E66F2D", "#D5A000", "#C94444", "#3A8D5D"]
LABELS = {
    "prefix_only": "Prefix only",
    "deferred_delta": "Deferred delta",
    "no_prediction": "No prediction",
    "gru_file": "GRU file",
    "gru_online": "GRU online",
    "oracle": "Oracle",
    "gpu_exact_only": "GPU exact",
    "gpu_plus_host_full": "GPU + Host full",
    "host_anchor_delta": "Host anchor + delta",
    "gru_gpu_ready": "GRU GPU-ready",
    "recency": "Recency",
    "frequency": "Frequency",
    "frequency_recency": "Freq. + recency",
    "frequency_prediction": "Freq. + prediction",
    "full_latency": "Full latency",
    "full_density": "Full density",
    "random": "Random",
    "semantic": "Semantic",
    "transition": "Transition",
    "hybrid": "Hybrid",
}
ORDER = {
    "PrefixDelta": ["prefix_only", "deferred_delta"],
    "Prediction": ["no_prediction", "gru_file", "gru_online", "oracle"],
    "Storage": ["gpu_exact_only", "gpu_plus_host_full", "host_anchor_delta", "gru_gpu_ready"],
    "Family": [
        "recency", "frequency", "frequency_recency",
        "frequency_prediction", "full_latency", "full_density",
    ],
    "Grouping": ["random", "semantic", "transition", "hybrid"],
}


def number(row: dict[str, str], key: str) -> float:
    try:
        result = float(row.get(key, 0.0))
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["case"])].append(row)

    metrics = [
        "ttft_change_rate", "full_prefix_hit_rate", "any_chunk_prefix_hit_rate",
        "mean_chunk_hit_tokens", "total_chunk_hit_tokens", "peak_host_full_mb",
        "peak_host_delta_mb", "final_delta_compression_rate", "delta_candidates",
        "delta_compressed", "delta_background_s", "prefetch_built", "prefetch_work_s",
        "prediction_top1_accuracy", "prediction_topk_accuracy", "prediction_query_ms",
        "gru_inference_ms", "delta_kv_cos", "delta_kv_l2", "delta_saved_rate",
        "family_events", "family_evictions",
    ]
    result: list[dict[str, float | str | int]] = []
    for (suite, case), samples in groups.items():
        item: dict[str, float | str | int] = {
            "suite": suite,
            "case": case,
            "runs": len(samples),
        }
        for metric in metrics:
            values = [number(row, metric) for row in samples]
            item[f"{metric}_mean"] = statistics.fmean(values) if values else 0.0
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def ordered(aggregated: list[dict], suite: str) -> list[dict]:
    by_case = {row["case"]: row for row in aggregated if row["suite"] == suite}
    return [by_case[case] for case in ORDER.get(suite, []) if case in by_case]


def style_axis(axis: plt.Axes, ylabel: str, percent: bool = False) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    if percent:
        axis.axhline(0, color="#68717C", linewidth=0.8)


def bar_metric(
    axis: plt.Axes,
    rows: list[dict],
    metric: str,
    ylabel: str,
    scale: float = 1.0,
    percent: bool = False,
) -> None:
    x = np.arange(len(rows))
    means = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float) * scale
    stds = np.asarray([row[f"{metric}_std"] for row in rows], dtype=float) * scale
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(rows))]
    axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="none", width=0.68)
    axis.set_xticks(x, [LABELS.get(str(row["case"]), str(row["case"])) for row in rows], rotation=18, ha="right")
    style_axis(axis, ylabel, percent=percent)
    for index, value in enumerate(means):
        offset = max(0.01 * max(float(np.max(np.abs(means))), 1.0), 0.02)
        axis.text(index, value + (offset if value >= 0 else -offset), f"{value:.1f}",
                  ha="center", va="bottom" if value >= 0 else "top", fontsize=8)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prefix_delta(rows: list[dict], output_dir: Path) -> None:
    data = ordered(rows, "PrefixDelta")
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    bar_metric(axes[0], data, "ttft_change_rate", "TTFT change (%)", 100.0, True)
    bar_metric(axes[1], data, "any_chunk_prefix_hit_rate", "Any-chunk hit rate (%)", 100.0)
    bar_metric(axes[2], data, "delta_background_s", "Background delta work (s)")
    save_figure(fig, output_dir, "01_prefix_delta_ablation")


def plot_family(rows: list[dict], output_dir: Path) -> None:
    data = ordered(rows, "Family")
    if not data:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    bar_metric(axes[0, 0], data, "any_chunk_prefix_hit_rate", "Any-chunk hit rate (%)", 100.0)
    bar_metric(axes[0, 1], data, "total_chunk_hit_tokens", "Total hit tokens")
    bar_metric(axes[1, 0], data, "peak_host_full_mb", "Peak Host full KV (MiB)")
    bar_metric(axes[1, 1], data, "family_evictions", "Family eviction/demotion events")
    save_figure(fig, output_dir, "02_family_score_ablation")


def plot_storage(rows: list[dict], output_dir: Path) -> None:
    data = ordered(rows, "Storage")
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    x = np.arange(len(data))
    full = np.asarray([row["peak_host_full_mb_mean"] for row in data])
    delta = np.asarray([row["peak_host_delta_mb_mean"] for row in data])
    axes[0].bar(x, full, color=PALETTE[1], label="Host full")
    axes[0].bar(x, delta, bottom=full, color=PALETTE[2], label="Host delta")
    axes[0].set_xticks(x, [LABELS[str(row["case"])] for row in data], rotation=18, ha="right")
    axes[0].legend(frameon=False)
    style_axis(axes[0], "Peak Host storage (MiB)")
    bar_metric(axes[1], data, "final_delta_compression_rate", "Delta compression rate (%)", 100.0)
    bar_metric(axes[2], data, "ttft_change_rate", "TTFT change (%)", 100.0, True)
    save_figure(fig, output_dir, "03_storage_tier_ablation")


def plot_grouping(rows: list[dict], output_dir: Path) -> None:
    data = ordered(rows, "Grouping")
    if not data:
        return
    transformed = []
    for row in data:
        copy = dict(row)
        copy["cosine_gap_mean"] = (1.0 - float(row["delta_kv_cos_mean"])) * 1000.0
        copy["cosine_gap_std"] = float(row["delta_kv_cos_std"]) * 1000.0
        transformed.append(copy)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    bar_metric(axes[0], transformed, "cosine_gap", "KV cosine gap (1-cos) x 1e3")
    bar_metric(axes[1], data, "delta_kv_l2", "KV relative L2")
    bar_metric(axes[2], data, "delta_saved_rate", "Logical delta saving (%)", 100.0)
    save_figure(fig, output_dir, "04_grouping_ablation")


def plot_prediction(rows: list[dict], output_dir: Path) -> None:
    data = ordered(rows, "Prediction")
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    x = np.arange(len(data))
    top1 = np.asarray([row["prediction_top1_accuracy_mean"] for row in data]) * 100.0
    topk = np.asarray([row["prediction_topk_accuracy_mean"] for row in data]) * 100.0
    width = 0.34
    axes[0].bar(x - width / 2, top1, width, color=PALETTE[0], label="Top-1")
    axes[0].bar(x + width / 2, topk, width, color=PALETTE[2], label="Top-k")
    axes[0].set_xticks(x, [LABELS[str(row["case"])] for row in data], rotation=18, ha="right")
    axes[0].legend(frameon=False)
    style_axis(axes[0], "Prediction accuracy (%)")
    bar_metric(axes[1], data, "ttft_change_rate", "TTFT change (%)", 100.0, True)
    bar_metric(axes[2], data, "prefetch_work_s", "Prefetch work (s)")
    save_figure(fig, output_dir, "05_prediction_ablation")


def write_aggregate(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(r"D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\system_ablation_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\figures_system_ablations"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate(read_rows(args.summary))
    write_aggregate(aggregated, args.output_dir / "system_ablation_aggregate.csv")
    plot_prefix_delta(aggregated, args.output_dir)
    plot_family(aggregated, args.output_dir)
    plot_storage(aggregated, args.output_dir)
    plot_grouping(aggregated, args.output_dir)
    plot_prediction(aggregated, args.output_dir)
    print(f"aggregated_cases={len(aggregated)}")
    print(f"figures={args.output_dir}")


if __name__ == "__main__":
    main()
