"""Create paper-ready figures for the dataset-driven MobiLoRA experiments.

The script reads the four CSV files produced by llama-lora-base-test and
creates both detailed figures and compact summary tables.  It deliberately
separates cold prefix construction, cross-LoRA construction and warm exact
prefix reuse because averaging these paths together can be misleading.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "baseline": "#777777",
    "online": "#147D92",
    "writing_similar": "#D45B35",
    "writing_control": "#3C6EAF",
    "summary_high": "#E4572E",
    "countdown_gradient": "#2E86AB",
    "code_gradient": "#2A9D8F",
    "summary_low": "#264653",
    "anchor": "#D9A441",
    "cross_lora": "#C64A55",
    "reuse": "#2F8A5B",
}

GROUP_LABELS = {
    "writing_similar": "High-similarity LoRA pair",
    "writing_control": "Low-similarity control",
    "summary_high": "Summary: high similarity",
    "countdown_gradient": "Countdown adapters",
    "code_gradient": "Code adapters",
    "summary_low": "Summary: low similarity",
}

PAIR_LABELS = {
    ("summary_sep_prompt", "summary_merged_prompt"): "Summary high",
    ("countdown_search", "countdown_search_react"): "Countdown search/react",
    ("countdown_search", "countdown_optimal"): "Countdown search/optimal",
    ("code_r16v3", "code_r16v2"): "Code v3/v2",
    ("code_r16v3", "code_r16"): "Code v3/v1",
    ("summarization_qlora", "summarization_sft"): "Summary low",
}

PAIR_COLORS = {
    "Summary high": "#E4572E",
    "Countdown search/react": "#2E86AB",
    "Countdown search/optimal": "#5AA9C7",
    "Code v3/v2": "#2A9D8F",
    "Code v3/v1": "#75B79E",
    "Summary low": "#264653",
}

MODE_LABELS = {
    "build_exact_prefix_anchor": "Cold anchor build",
    "build_cross_lora_full_prefix_variant": "Cold cross-LoRA build",
    "reuse_exact_prefix_same_lora": "Warm exact-prefix reuse",
}

MODE_COLORS = {
    "build_exact_prefix_anchor": COLORS["anchor"],
    "build_cross_lora_full_prefix_variant": COLORS["cross_lora"],
    "reuse_exact_prefix_same_lora": COLORS["reuse"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot delta-KV and online prefix-cache experiment results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "D:/ecnu_experiment/LLama.cpp/llama.cpp/"
            "examples/lora-base-test/output"
        ),
        help="Directory containing the four experiment CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory. Defaults to INPUT_DIR/figures.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats, for example png,pdf.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional workload profile containing pair_notes and weight cosine.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_csv(input_dir: Path, filename: str, required: bool = True) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing result file: {path}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if required and frame.empty:
        raise ValueError(f"Result file is empty: {path}")
    return frame


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> None:
    for extension in formats:
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=dpi)
    plt.close(fig)


def sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def add_group_columns(probe: pd.DataFrame) -> pd.DataFrame:
    probe = probe.copy()
    probe["group_label"] = probe["group_name"].map(GROUP_LABELS).fillna(
        probe["group_name"]
    )
    probe["cosine_distance"] = np.maximum(1.0 - probe["kv_cos"], 1e-12)
    probe["full_kv_mib"] = probe["full_kv_bytes"] / 2**20
    probe["delta_total_bytes"] = probe["delta_q8_bytes"] + probe["delta_scale_bytes"]
    probe["delta_total_mib"] = probe["delta_total_bytes"] / 2**20
    probe["logical_saved_mib"] = probe["logical_saved_bytes"] / 2**20
    pair_keys = list(zip(probe["anchor_lora_name"], probe["child_lora_name"]))
    probe["pair_label"] = [
        PAIR_LABELS.get(key, f"{key[0]} / {key[1]}") for key in pair_keys
    ]
    return probe


def load_weight_cosines(profile_path: Path | None) -> dict[tuple[str, str], float]:
    if profile_path is None:
        return {}
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return {
        (str(item["anchor"]), str(item["child"])): float(
            item["effective_update_cosine"]
        )
        for item in profile.get("pair_notes", [])
    }


def pair_summary(
    probe: pd.DataFrame,
    weight_cosines: dict[tuple[str, str], float],
) -> pd.DataFrame:
    summary = (
        probe.groupby(
            ["group_name", "anchor_lora_name", "child_lora_name", "pair_label"],
            sort=False,
        )
        .agg(
            contexts=("pair_name", "size"),
            prefix_tokens_mean=("prefix_tokens", "mean"),
            kv_cos_mean=("kv_cos", "mean"),
            kv_cos_std=("kv_cos", "std"),
            kv_l2_mean=("kv_l2", "mean"),
            kv_l2_std=("kv_l2", "std"),
            probe_ms_mean=("probe_ms", "mean"),
            delta_build_ms_mean=("delta_build_ms", "mean"),
            logical_saved_rate_mean=("logical_saved_rate", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )
    summary["weight_cosine"] = [
        weight_cosines.get((anchor, child), np.nan)
        for anchor, child in zip(
            summary["anchor_lora_name"], summary["child_lora_name"]
        )
    ]
    summary["kv_cosine_distance"] = np.maximum(1.0 - summary["kv_cos_mean"], 1e-12)
    return summary


def summarize_delta(probe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in probe.groupby("group_name", sort=False):
        rows.append(
            {
                "group_name": group_name,
                "group_label": GROUP_LABELS.get(group_name, group_name),
                "pairs": len(group),
                "prefix_tokens_mean": group["prefix_tokens"].mean(),
                "prefix_tokens_median": group["prefix_tokens"].median(),
                "kv_cos_mean": group["kv_cos"].mean(),
                "kv_cos_std": group["kv_cos"].std(ddof=1),
                "kv_l2_mean": group["kv_l2"].mean(),
                "kv_l2_std": group["kv_l2"].std(ddof=1),
                "delta_success_rate": group["delta_build_ok"].mean(),
                "full_kv_mib_mean": group["full_kv_mib"].mean(),
                "delta_total_mib_mean": group["delta_total_mib"].mean(),
                "logical_saved_rate_mean": group["logical_saved_rate"].mean(),
                "anchor_eval_ms_mean": group["anchor_eval_ms"].mean(),
                "child_eval_ms_mean": group["child_eval_ms"].mean(),
                "probe_ms_mean": group["probe_ms"].mean(),
                "delta_build_ms_mean": group["delta_build_ms"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_delta_group_comparison(
    probe: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    groups = list(probe["group_name"].drop_duplicates())
    labels = [GROUP_LABELS.get(group, group) for group in groups]
    colors = [COLORS.get(group, "#777777") for group in groups]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    metrics = [
        ("cosine_distance", "KV cosine distance (1 - cosine)", True),
        ("kv_l2", "KV L2 difference", False),
    ]
    rng = np.random.default_rng(42)

    for axis, (column, ylabel, use_log) in zip(axes, metrics):
        values = [probe.loc[probe["group_name"] == group, column] for group in groups]
        box = axis.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.4},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for index, series in enumerate(values, start=1):
            x = rng.normal(index, 0.045, len(series))
            axis.scatter(x, series, s=10, color=colors[index - 1], alpha=0.32)
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=12, ha="right")
        axis.set_ylabel(ylabel)
        if use_log and (probe[column] > 0).all():
            axis.set_yscale("log")

    axes[0].set_title("Direction difference")
    axes[1].set_title("Magnitude difference")
    fig.suptitle("Cross-LoRA KV similarity on identical token prefixes", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "01_delta_group_comparison", formats, dpi)


def plot_delta_pair_comparison(
    probe: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    pairs = list(probe["pair_label"].drop_duplicates())
    if len(pairs) <= 1:
        return
    colors = [PAIR_COLORS.get(pair, "#777777") for pair in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    metrics = [
        ("cosine_distance", "KV cosine distance (1 - cosine)"),
        ("kv_l2", "KV L2 difference"),
    ]
    for axis, (column, ylabel) in zip(axes, metrics):
        values = [probe.loc[probe["pair_label"] == pair, column] for pair in pairs]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        axis.set_xticks(range(1, len(pairs) + 1), pairs, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
    axes[0].set_title("Adapter-pair direction difference")
    axes[1].set_title("Adapter-pair magnitude difference")
    fig.suptitle("KV delta distribution for six LoRA adapter pairs", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "01b_delta_pair_comparison", formats, dpi)


def plot_weight_kv_correlation(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    summary = summary.dropna(subset=["weight_cosine"]).sort_values("weight_cosine")
    if len(summary) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    targets = [
        ("kv_cosine_distance", "KV cosine distance (1 - cosine)"),
        ("kv_l2_mean", "Mean KV L2 difference"),
    ]
    for axis, (target, ylabel) in zip(axes, targets):
        for _, row in summary.iterrows():
            color = PAIR_COLORS.get(row["pair_label"], "#777777")
            axis.scatter(
                row["weight_cosine"],
                row[target],
                color=color,
                s=70,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            axis.annotate(
                row["pair_label"],
                (row["weight_cosine"], row[target]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
            )
        pearson = summary["weight_cosine"].corr(summary[target], method="pearson")
        spearman = summary["weight_cosine"].corr(summary[target], method="spearman")
        axis.text(
            0.03,
            0.04,
            f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}",
            transform=axis.transAxes,
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("LoRA effective-update cosine")
        axis.set_ylabel(ylabel)
    axes[0].set_title("Weight similarity vs. KV direction")
    axes[1].set_title("Weight similarity vs. KV magnitude")
    fig.tight_layout()
    save_figure(fig, output_dir, "09_weight_kv_correlation", formats, dpi)


def plot_delta_storage(
    probe: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    grouped = (
        probe.groupby("group_name", sort=False)
        .agg(
            full_kv_mib=("full_kv_mib", "mean"),
            delta_total_mib=("delta_total_mib", "mean"),
            saved_rate=("logical_saved_rate", "mean"),
        )
        .reset_index()
    )
    x = np.arange(len(grouped))
    width = 0.34

    fig, axis = plt.subplots(figsize=(7.4, 3.9))
    full_bars = axis.bar(
        x - width / 2,
        grouped["full_kv_mib"],
        width,
        label="Full child KV",
        color="#8A8A8A",
    )
    delta_bars = axis.bar(
        x + width / 2,
        grouped["delta_total_mib"],
        width,
        label="Q8 delta + scales",
        color="#147D92",
    )
    axis.set_xticks(
        x,
        [GROUP_LABELS.get(group, group) for group in grouped["group_name"]],
        rotation=10,
        ha="right",
    )
    axis.set_ylabel("Mean storage per child branch (MiB)")
    axis.set_title("Logical child-branch storage")
    axis.legend(ncols=2, loc="upper right")
    axis.bar_label(full_bars, fmt="%.2f", padding=2, fontsize=8)
    axis.bar_label(delta_bars, fmt="%.2f", padding=2, fontsize=8)

    for index, row in grouped.iterrows():
        top = max(row["full_kv_mib"], row["delta_total_mib"])
        axis.text(
            index,
            top * 1.10,
            f"child saving: {row['saved_rate'] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.margins(y=0.22)
    fig.tight_layout()
    save_figure(fig, output_dir, "02_delta_storage", formats, dpi)


def plot_layerwise_similarity(
    layers: pd.DataFrame,
    probe: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    pair_to_group = probe.set_index("pair_name")["group_name"].to_dict()
    layers = layers.copy()
    layers["group_name"] = layers["pair_name"].map(pair_to_group)
    layers = layers.dropna(subset=["group_name"])
    layers["cosine_distance"] = np.maximum(1.0 - layers["kv_cos"], 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for group_name, group in layers.groupby("group_name", sort=False):
        stats = (
            group.groupby("layer_id")
            .agg(
                cos_mean=("cosine_distance", "mean"),
                cos_std=("cosine_distance", "std"),
                l2_mean=("kv_l2", "mean"),
                l2_std=("kv_l2", "std"),
            )
            .reset_index()
            .fillna(0.0)
        )
        color = COLORS.get(group_name, "#777777")
        label = GROUP_LABELS.get(group_name, group_name)
        layer_ids = stats["layer_id"].to_numpy(dtype=float)
        cos_mean = stats["cos_mean"].to_numpy(dtype=float)
        cos_std = stats["cos_std"].to_numpy(dtype=float)
        l2_mean = stats["l2_mean"].to_numpy(dtype=float)
        l2_std = stats["l2_std"].to_numpy(dtype=float)
        axes[0].plot(layer_ids, cos_mean, color=color, linewidth=1.8, label=label)
        axes[0].fill_between(
            layer_ids,
            np.maximum(cos_mean - cos_std, 1e-12),
            cos_mean + cos_std,
            color=color,
            alpha=0.13,
        )
        axes[1].plot(layer_ids, l2_mean, color=color, linewidth=1.8, label=label)
        axes[1].fill_between(
            layer_ids,
            np.maximum(l2_mean - l2_std, 0),
            l2_mean + l2_std,
            color=color,
            alpha=0.13,
        )

    axes[0].set_yscale("log")
    axes[0].set_title("Layer-wise cosine distance")
    axes[0].set_ylabel("1 - KV cosine")
    axes[1].set_title("Layer-wise L2 difference")
    axes[1].set_ylabel("KV L2")
    for axis in axes:
        axis.set_xlabel("Transformer layer")
        axis.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "03_layerwise_kv_similarity", formats, dpi)


def build_paired_online(online: pd.DataFrame) -> pd.DataFrame:
    baseline = online.loc[online["benchmark"] == "baseline"].copy()
    optimized = online.loc[online["benchmark"] == "online"].copy()
    if baseline.empty or optimized.empty:
        return pd.DataFrame()

    baseline = baseline[["request_id", "ttft_ms", "total_ms", "tps"]].rename(
        columns={
            "ttft_ms": "baseline_ttft_ms",
            "total_ms": "baseline_total_ms",
            "tps": "baseline_tps",
        }
    )
    paired = optimized.merge(baseline, on="request_id", how="inner")
    paired["mode_label"] = paired["mode"].map(MODE_LABELS).fillna(paired["mode"])
    paired["ttft_speedup"] = paired["baseline_ttft_ms"] / paired["ttft_ms"]
    paired["ttft_reduction_rate"] = 1.0 - paired["ttft_ms"] / paired["baseline_ttft_ms"]
    return paired


def summarize_online(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows = []
    for mode, group in paired.groupby("mode", sort=False):
        baseline_mean = group["baseline_ttft_ms"].mean()
        online_mean = group["ttft_ms"].mean()
        rows.append(
            {
                "mode": mode,
                "mode_label": MODE_LABELS.get(mode, mode),
                "requests": len(group),
                "baseline_ttft_ms_mean": baseline_mean,
                "online_ttft_ms_mean": online_mean,
                "online_ttft_ms_sem": sem(group["ttft_ms"]),
                "ttft_speedup": baseline_mean / online_mean,
                "ttft_reduction_rate": 1.0 - online_mean / baseline_mean,
                "online_total_ms_mean": group["total_ms"].mean(),
                "online_tps_mean": group["tps"].mean(),
                "prefix_ms_mean": group["prefix_ms"].mean(),
                "suffix_ms_mean": group["suffix_ms"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_online_ttft(
    paired: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    if paired.empty:
        return
    summary = summarize_online(paired)
    x = np.arange(len(summary))
    width = 0.34

    fig, axis = plt.subplots(figsize=(8.8, 4.1))
    baseline_bars = axis.bar(
        x - width / 2,
        summary["baseline_ttft_ms_mean"],
        width,
        color=COLORS["baseline"],
        label="Baseline full prefill",
    )
    online_colors = [MODE_COLORS.get(mode, COLORS["online"]) for mode in summary["mode"]]
    online_bars = axis.bar(
        x + width / 2,
        summary["online_ttft_ms_mean"],
        width,
        color=online_colors,
        label="Online path",
    )
    axis.set_xticks(x, summary["mode_label"], rotation=12, ha="right")
    axis.set_ylabel("Mean TTFT (ms)")
    axis.set_title("Paired TTFT comparison by online execution path")
    axis.legend(ncols=2)
    axis.bar_label(baseline_bars, fmt="%.0f", padding=2, fontsize=8)
    axis.bar_label(online_bars, fmt="%.0f", padding=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_online_ttft_by_mode", formats, dpi)


def plot_online_timeline(
    paired: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    if paired.empty:
        return
    paired = paired.sort_values("request_id")
    fig, axis = plt.subplots(figsize=(10.4, 4.0))
    axis.plot(
        paired["request_id"],
        paired["baseline_ttft_ms"],
        color=COLORS["baseline"],
        linewidth=1.2,
        alpha=0.8,
        label="Baseline",
    )
    for mode, group in paired.groupby("mode", sort=False):
        axis.scatter(
            group["request_id"],
            group["ttft_ms"],
            color=MODE_COLORS.get(mode, COLORS["online"]),
            s=20,
            alpha=0.8,
            label=MODE_LABELS.get(mode, mode),
        )
    axis.set_xlabel("Request arrival index")
    axis.set_ylabel("TTFT (ms)")
    axis.set_title("Online request timeline")
    axis.legend(ncols=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "05_online_request_timeline", formats, dpi)


def plot_cache_timeline(
    paired: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    if paired.empty:
        return
    paired = paired.sort_values("request_id")
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.9), sharex=True)
    axes[0].step(
        paired["request_id"],
        paired["cache_nodes"],
        where="post",
        label="Prefix nodes",
        color="#3C6EAF",
    )
    axes[0].step(
        paired["request_id"],
        paired["cache_variants"],
        where="post",
        label="LoRA variants",
        color="#D45B35",
    )
    axes[0].set_ylabel("Cached objects")
    axes[0].set_title("Online prefix-cache occupancy")
    axes[0].legend(ncols=2)

    axes[1].step(
        paired["request_id"],
        paired["physical_cache_tokens"],
        where="post",
        color="#147D92",
        linewidth=1.5,
    )
    reuse = paired["same_lora_variant_hit"] == 1
    cross = paired["cross_lora_prefix_match"] == 1
    axes[1].scatter(
        paired.loc[reuse, "request_id"],
        paired.loc[reuse, "physical_cache_tokens"],
        color=COLORS["reuse"],
        s=18,
        label="Same-LoRA hit",
    )
    axes[1].scatter(
        paired.loc[cross, "request_id"],
        paired.loc[cross, "physical_cache_tokens"],
        color=COLORS["cross_lora"],
        s=18,
        label="Cross-LoRA prefix match",
    )
    axes[1].set_xlabel("Request arrival index")
    axes[1].set_ylabel("Physical cached prefix tokens")
    axes[1].legend(ncols=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "06_cache_occupancy_timeline", formats, dpi)


def plot_prefix_length_relationship(
    probe: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    for group_name, group in probe.groupby("group_name", sort=False):
        color = COLORS.get(group_name, "#777777")
        label = GROUP_LABELS.get(group_name, group_name)
        axes[0].scatter(
            group["prefix_tokens"],
            group["cosine_distance"],
            color=color,
            alpha=0.55,
            s=18,
            label=label,
        )
        axes[1].scatter(
            group["prefix_tokens"],
            group["delta_build_ms"],
            color=color,
            alpha=0.55,
            s=18,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Identical prefix length (tokens)")
    axes[0].set_ylabel("1 - KV cosine")
    axes[0].set_title("Prefix length vs. KV difference")
    axes[1].set_xlabel("Identical prefix length (tokens)")
    axes[1].set_ylabel("Delta encoding time (ms)")
    axes[1].set_title("Prefix length vs. encoding cost")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "07_prefix_length_relationship", formats, dpi)


def plot_delta_timing_breakdown(
    probe: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    timing_columns = ["anchor_eval_ms", "child_eval_ms", "probe_ms", "delta_build_ms"]
    timing_labels = ["Anchor prefill", "Child prefill", "KV probe", "Delta encoding"]
    grouped = probe.groupby("group_name", sort=False)[timing_columns].mean()
    x = np.arange(len(grouped))
    width = 0.18
    timing_colors = ["#777777", "#D9A441", "#3C6EAF", "#C64A55"]

    fig, axis = plt.subplots(figsize=(8.7, 4.1))
    for index, (column, label, color) in enumerate(
        zip(timing_columns, timing_labels, timing_colors)
    ):
        offset = (index - 1.5) * width
        bars = axis.bar(x + offset, grouped[column], width, label=label, color=color)
        axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)
    axis.set_xticks(
        x,
        [GROUP_LABELS.get(group, group) for group in grouped.index],
        rotation=10,
        ha="right",
    )
    axis.set_ylabel("Mean time (ms)")
    axis.set_title("Cold delta construction cost")
    axis.legend(ncols=2)
    axis.margins(y=0.15)
    fig.tight_layout()
    save_figure(fig, output_dir, "08_delta_timing_breakdown", formats, dpi)


def plot_dashboard(
    probe: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    groups = list(probe["group_name"].drop_duplicates())
    labels = [GROUP_LABELS.get(group, group) for group in groups]
    colors = [COLORS.get(group, "#777777") for group in groups]

    distance_values = [
        probe.loc[probe["group_name"] == group, "cosine_distance"] for group in groups
    ]
    boxes = axes[0, 0].boxplot(distance_values, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(range(1, len(labels) + 1), labels, rotation=10, ha="right")
    axes[0, 0].set_ylabel("1 - KV cosine")
    axes[0, 0].set_title("(a) Cross-LoRA KV difference")

    storage = probe.groupby("group_name", sort=False)[
        ["full_kv_mib", "delta_total_mib"]
    ].mean()
    x = np.arange(len(storage))
    axes[0, 1].bar(x - 0.17, storage["full_kv_mib"], 0.34, color="#888888", label="Full")
    axes[0, 1].bar(x + 0.17, storage["delta_total_mib"], 0.34, color="#147D92", label="Delta")
    axes[0, 1].set_xticks(
        x,
        [GROUP_LABELS.get(group, group) for group in storage.index],
        rotation=10,
        ha="right",
    )
    axes[0, 1].set_ylabel("MiB per child branch")
    axes[0, 1].set_title("(b) Logical storage")
    axes[0, 1].legend(ncols=2)

    if not paired.empty:
        online_summary = summarize_online(paired)
        x = np.arange(len(online_summary))
        axes[1, 0].bar(
            x - 0.17,
            online_summary["baseline_ttft_ms_mean"],
            0.34,
            color=COLORS["baseline"],
            label="Baseline",
        )
        axes[1, 0].bar(
            x + 0.17,
            online_summary["online_ttft_ms_mean"],
            0.34,
            color=[MODE_COLORS.get(mode, COLORS["online"]) for mode in online_summary["mode"]],
            label="Online",
        )
        axes[1, 0].set_xticks(
            x,
            online_summary["mode_label"],
            rotation=12,
            ha="right",
        )
        axes[1, 0].set_ylabel("Mean TTFT (ms)")
        axes[1, 0].set_title("(c) TTFT by execution path")
        axes[1, 0].legend(ncols=2)

        ordered = paired.sort_values("request_id")
        axes[1, 1].step(
            ordered["request_id"],
            ordered["physical_cache_tokens"],
            where="post",
            color="#147D92",
        )
        axes[1, 1].set_xlabel("Request arrival index")
        axes[1, 1].set_ylabel("Cached prefix tokens")
        axes[1, 1].set_title("(d) Cache occupancy")
    else:
        axes[1, 0].text(0.5, 0.5, "No online results", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "No online results", ha="center", va="center")

    fig.suptitle("Multi-LoRA KV delta and prefix-cache experiment", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir, "08_experiment_dashboard", formats, dpi)


def write_summaries(
    probe: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
    adapter_pair_summary: pd.DataFrame,
) -> None:
    delta_summary = summarize_delta(probe)
    delta_summary.to_csv(output_dir / "delta_group_summary.csv", index=False)

    online_summary = summarize_online(paired)
    if not online_summary.empty:
        online_summary.to_csv(output_dir / "online_mode_summary.csv", index=False)
        paired.to_csv(output_dir / "online_paired_requests.csv", index=False)
    adapter_pair_summary.to_csv(output_dir / "adapter_pair_summary.csv", index=False)

    payload = {
        "delta_pairs": int(len(probe)),
        "delta_groups": {
            str(row["group_name"]): {
                "pairs": int(row["pairs"]),
                "kv_cos_mean": float(row["kv_cos_mean"]),
                "kv_l2_mean": float(row["kv_l2_mean"]),
                "logical_saved_rate_mean": float(row["logical_saved_rate_mean"]),
                "delta_success_rate": float(row["delta_success_rate"]),
            }
            for _, row in delta_summary.iterrows()
        },
        "online_requests": int(len(paired)),
    }
    if not online_summary.empty:
        payload["online_modes"] = {
            str(row["mode"]): {
                "requests": int(row["requests"]),
                "baseline_ttft_ms_mean": float(row["baseline_ttft_ms_mean"]),
                "online_ttft_ms_mean": float(row["online_ttft_ms_mean"]),
                "ttft_speedup": float(row["ttft_speedup"]),
            }
            for _, row in online_summary.iterrows()
        }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "figures").resolve()
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    delta_path = input_dir / "delta_prefix_probe.csv"
    online_path = input_dir / "online_request_results.csv"
    if delta_path.exists() and online_path.exists():
        delta_mtime = os.path.getmtime(delta_path)
        online_mtime = os.path.getmtime(online_path)
        if online_mtime + 60 < delta_mtime:
            print(
                "WARNING: online_request_results.csv is older than the delta results. "
                "The experiment may still be running, so online figures can be stale."
            )

    probe = add_group_columns(read_csv(input_dir, "delta_prefix_probe.csv"))
    layers = read_csv(input_dir, "delta_prefix_layers.csv")
    online = read_csv(input_dir, "online_request_results.csv", required=False)
    paired = build_paired_online(online) if not online.empty else pd.DataFrame()
    weight_cosines = load_weight_cosines(args.profile)
    adapter_pair_summary = pair_summary(probe, weight_cosines)

    plot_delta_group_comparison(probe, output_dir, formats, args.dpi)
    plot_delta_pair_comparison(probe, output_dir, formats, args.dpi)
    plot_delta_storage(probe, output_dir, formats, args.dpi)
    plot_layerwise_similarity(layers, probe, output_dir, formats, args.dpi)
    plot_online_ttft(paired, output_dir, formats, args.dpi)
    plot_online_timeline(paired, output_dir, formats, args.dpi)
    plot_cache_timeline(paired, output_dir, formats, args.dpi)
    plot_prefix_length_relationship(probe, output_dir, formats, args.dpi)
    plot_delta_timing_breakdown(probe, output_dir, formats, args.dpi)
    plot_dashboard(probe, paired, output_dir, formats, args.dpi)
    plot_weight_kv_correlation(
        adapter_pair_summary, output_dir, formats, args.dpi
    )
    write_summaries(
        probe, paired, output_dir, adapter_pair_summary
    )

    print(f"Loaded delta pairs: {len(probe)}")
    print(f"Loaded paired online requests: {len(paired)}")
    print(f"Figures and summaries saved to: {output_dir}")


if __name__ == "__main__":
    main()
