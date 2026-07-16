#!/usr/bin/env python3
"""Create colorful paper-ready figures for the 87-LoRA delta experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#2878B5",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "teal": "#2A9D8F",
    "gold": "#EDC948",
    "purple": "#8E6CBE",
    "pink": "#D37295",
    "navy": "#1D3557",
    "gray": "#8A8F98",
}


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "axes.edgecolor": "#C8D0DA",
            "grid.color": "#D9E0E8",
            "grid.alpha": 0.55,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#25313C",
            "xtick.color": "#44515E",
            "ytick.color": "#44515E",
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png")
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def family_color(group_name: str) -> str:
    if "code" in group_name:
        return PALETTE["orange"]
    if "reasoning" in group_name:
        return PALETTE["red"]
    if "summarization" in group_name:
        return PALETTE["green"]
    return PALETTE["blue"]


def plot_group_comparison(probe: pd.DataFrame, output: Path) -> None:
    groups = (
        probe.groupby("group_name")
        .agg(pairs=("pair_name", "size"), cosine=("kv_cos", "mean"), l2=("kv_l2", "mean"))
        .sort_values("cosine")
    )
    labels = [name.replace("group_", "G").replace("_mobile_context", " mobile") for name in groups.index]
    colors = [family_color(name) for name in groups.index]
    y = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"wspace": 0.34})

    axes[0].hlines(y, 0.95, groups.cosine, color=colors, linewidth=3, alpha=0.45)
    axes[0].scatter(groups.cosine, y, s=90, c=colors, edgecolor="white", linewidth=1.2, zorder=3)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(0.95, 1.001)
    axes[0].set_xlabel("Mean KV cosine (higher is better)")
    axes[0].set_title("Similarity by LoRA group")
    for index, value in enumerate(groups.cosine):
        axes[0].text(value + 0.0007, index, f"{value:.4f}", va="center", fontsize=8)

    axes[1].barh(y, groups.l2, color=colors, alpha=0.88, edgecolor="white")
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Mean KV L2 (lower is better)")
    axes[1].set_title("Delta magnitude by LoRA group")
    for index, value in enumerate(groups.l2):
        axes[1].text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("In-group delta quality across 385 anchor-child pairs", fontsize=14, fontweight="bold")
    save(fig, output, "01_color_group_comparison")


def plot_storage(probe: pd.DataFrame, output: Path) -> None:
    full = probe.full_kv_bytes.mean() / 1024**2
    delta = (probe.delta_q8_bytes + probe.delta_scale_bytes).mean() / 1024**2
    saved = (full - delta) / full * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Full KV\nFP16", "Delta + scale\nQ8"],
        [full, delta],
        color=[PALETTE["blue"], PALETTE["orange"]],
        width=0.55,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.bar_label(bars, labels=[f"{full:.2f} MB", f"{delta:.2f} MB"], padding=4, fontsize=11)
    ax.annotate(
        f"{saved:.2f}% smaller",
        xy=(1, delta),
        xytext=(0.5, full * 0.86),
        ha="center",
        color=PALETTE["red"],
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": PALETTE["red"], "lw": 1.6},
    )
    ax.set_ylabel("Mean logical storage per child branch (MB)")
    ax.set_title("Logical KV storage: full tensor vs quantized delta")
    ax.set_ylim(0, full * 1.22)
    save(fig, output, "02_color_storage")


def plot_layerwise(layers: pd.DataFrame, output: Path) -> None:
    data = layers.groupby("layer_id")[["k_cos", "v_cos", "k_l2", "v_l2"]].mean()
    x = data.index.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(x, data.k_cos, color=PALETTE["blue"], marker="o", markersize=4, linewidth=2, label="Key cosine")
    axes[0].plot(x, data.v_cos, color=PALETTE["orange"], marker="s", markersize=4, linewidth=2, linestyle="--", label="Value cosine")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_ylim(min(data.v_cos.min() - 0.003, 0.97), 1.001)
    axes[0].set_title("Layer-wise Key/Value similarity")
    axes[0].legend(ncol=2)

    axes[1].plot(x, data.k_l2, color=PALETTE["teal"], marker="o", markersize=4, linewidth=2, label="Key L2")
    axes[1].plot(x, data.v_l2, color=PALETTE["red"], marker="s", markersize=4, linewidth=2, linestyle="--", label="Value L2")
    axes[1].fill_between(x, data.k_l2, data.v_l2, color=PALETTE["red"], alpha=0.08)
    axes[1].set_xlabel("Transformer layer")
    axes[1].set_ylabel("Normalized L2 difference")
    axes[1].set_title("Value divergence grows in later layers")
    axes[1].legend(ncol=2)
    save(fig, output, "03_color_layerwise_kv")


def plot_prefix_timing(probe: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    groups = sorted(probe.group_name.unique())
    cmap = plt.get_cmap("turbo")
    colors = {group: cmap(index / max(1, len(groups) - 1)) for index, group in enumerate(groups)}
    for group in groups:
        part = probe[probe.group_name == group]
        axes[0].scatter(part.prefix_tokens, part.probe_ms, s=25, alpha=0.65, color=colors[group], label=group.replace("group_", "G"))
        axes[1].scatter(part.prefix_tokens, part.delta_build_ms, s=25, alpha=0.65, color=colors[group])

    for ax, column, title in [
        (axes[0], "probe_ms", "Probe cost"),
        (axes[1], "delta_build_ms", "Delta build cost"),
    ]:
        coefficients = np.polyfit(probe.prefix_tokens, probe[column], 1)
        x = np.linspace(probe.prefix_tokens.min(), probe.prefix_tokens.max(), 100)
        ax.plot(x, coefficients[0] * x + coefficients[1], color=PALETTE["navy"], linewidth=2.5, label="Linear trend")
        ax.set_xlabel("Prefix length (tokens)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Delta overhead scales almost linearly with prefix length", fontsize=14, fontweight="bold")
    save(fig, output, "04_color_prefix_timing")


def plot_timing_breakdown(probe: pd.DataFrame, output: Path) -> None:
    stages = ["Anchor eval", "Child eval", "Probe", "Delta build"]
    values = [
        probe.anchor_eval_ms.mean(),
        probe.child_eval_ms.mean(),
        probe.probe_ms.mean(),
        probe.delta_build_ms.mean(),
    ]
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["red"]]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(stages, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, labels=[f"{value:.0f} ms" for value in values], padding=4)
    ax.set_ylabel("Mean time per pair (ms)")
    ax.set_title("Offline delta construction is dominated by probe and encoding")
    ax.set_ylim(0, max(values) * 1.2)
    save(fig, output, "05_color_timing_breakdown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = pd.read_csv(args.input_dir / "delta_prefix_probe.csv")
    layers = pd.read_csv(args.input_dir / "delta_prefix_layers.csv")
    setup_style()
    plot_group_comparison(probe, args.output_dir)
    plot_storage(probe, args.output_dir)
    plot_layerwise(layers, args.output_dir)
    plot_prefix_timing(probe, args.output_dir)
    plot_timing_breakdown(probe, args.output_dir)
    print(f"Color figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
