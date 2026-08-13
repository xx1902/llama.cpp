from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = ["traditional", "layer-wise", "cuda-sync", "cuda-overlap"]
MODE_LABELS = {
    "traditional": "Traditional",
    "layer-wise": "Layer-wise",
    "cuda-sync": "CUDA sync",
    "cuda-overlap": "CUDA overlap",
}
COLORS = {
    "traditional": "#6B7280",
    "layer-wise": "#C4573A",
    "cuda-sync": "#2878B5",
    "cuda-overlap": "#2A9D6F",
}
GRID_COLOR = "#D9D9D9"


def mean_ci95(values: pd.Series) -> tuple[float, float]:
    samples = values.astype(float).to_numpy()
    mean = float(np.mean(samples))
    if len(samples) < 2:
        return mean, 0.0
    # Student-t critical value for df=11; this experiment has 12 samples/mode.
    critical = 2.201 if len(samples) == 12 else 1.96
    half_width = critical * float(np.std(samples, ddof=1)) / np.sqrt(len(samples))
    return mean, half_width


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)


def label_bars(
    axis: plt.Axes,
    bars,
    values: list[float],
) -> None:
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color="white",
        )


def draw_latency(results: pd.DataFrame, output_dir: Path) -> None:
    means_ttft = []
    ci_ttft = []
    means_total = []
    ci_total = []
    for mode in MODE_ORDER:
        rows = results[results["mode"] == mode]
        mean, ci = mean_ci95(rows["ttft_ms"])
        means_ttft.append(mean)
        ci_ttft.append(ci)
        mean, ci = mean_ci95(rows["total_ms"])
        means_total.append(mean)
        ci_total.append(ci)

    x = np.arange(len(MODE_ORDER))
    width = 0.34
    fig, axis = plt.subplots(figsize=(10.4, 5.6), constrained_layout=True)
    ttft_bars = axis.bar(
        x - width / 2,
        means_ttft,
        width,
        yerr=ci_ttft,
        capsize=4,
        color="#2878B5",
        label="TTFT",
    )
    total_bars = axis.bar(
        x + width / 2,
        means_total,
        width,
        yerr=ci_total,
        capsize=4,
        color="#E18B3A",
        label="Total",
    )
    label_bars(axis, ttft_bars, means_ttft)
    label_bars(axis, total_bars, means_total)
    axis.set_title("End-to-end latency by delta scheduling scheme")
    axis.set_ylabel("Latency (ms, mean with 95% CI)")
    axis.set_xticks(x, [MODE_LABELS[mode] for mode in MODE_ORDER])
    axis.set_ylim(0, max(np.array(means_total) + np.array(ci_total)) * 1.20)
    axis.legend(frameon=False, ncol=2)
    style_axis(axis)
    fig.savefig(output_dir / "ttft-total-comparison.png", dpi=200, facecolor="white")
    plt.close(fig)


def paired_differences(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = results.pivot(
        index=["repeat", "pair_name"], columns="mode", values="ttft_ms"
    ).dropna(subset=MODE_ORDER)
    differences = pd.DataFrame(index=pivot.index)
    for mode in MODE_ORDER[1:]:
        differences[mode] = pivot[mode] - pivot["traditional"]

    records = []
    for mode in MODE_ORDER[1:]:
        mean, ci = mean_ci95(differences[mode])
        records.append(
            {
                "mode": mode,
                "count": len(differences),
                "mean_paired_ttft_delta_ms": mean,
                "ci95_half_width_ms": ci,
                "ci95_low_ms": mean - ci,
                "ci95_high_ms": mean + ci,
                "median_paired_ttft_delta_ms": float(differences[mode].median()),
            }
        )
    return differences, pd.DataFrame.from_records(records)


def draw_paired_delta(
    differences: pd.DataFrame, paired_summary: pd.DataFrame, output_dir: Path
) -> None:
    modes = MODE_ORDER[1:]
    x = np.arange(len(modes))
    rng = np.random.default_rng(20260811)
    fig, axis = plt.subplots(figsize=(9.8, 5.5), constrained_layout=True)

    for index, mode in enumerate(modes):
        values = differences[mode].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=30,
            alpha=0.72,
            color=COLORS[mode],
            edgecolors="none",
        )
        row = paired_summary[paired_summary["mode"] == mode].iloc[0]
        mean = float(row["mean_paired_ttft_delta_ms"])
        ci = float(row["ci95_half_width_ms"])
        axis.errorbar(
            index,
            mean,
            yerr=ci,
            fmt="D",
            markersize=6,
            capsize=6,
            color="#202020",
            linewidth=1.6,
        )
        label_y_offset = 14 if abs(mean) < 5 else 0
        axis.annotate(
            f"{mean:+.1f} ms",
            (index, mean),
            xytext=(10, label_y_offset),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    axis.axhline(0, color="#202020", linewidth=1, linestyle="--")
    axis.set_title("Paired TTFT difference from traditional inference")
    axis.set_ylabel("TTFT difference (ms; lower is better)")
    axis.set_xticks(x, [MODE_LABELS[mode] for mode in modes])
    style_axis(axis)
    fig.savefig(output_dir / "paired-ttft-difference.png", dpi=200, facecolor="white")
    plt.close(fig)


def draw_compression(results: pd.DataFrame, output_dir: Path) -> None:
    delta_rows = results[results["mode"] != "traditional"]
    full_mib = float(delta_rows["full_kv_bytes"].mean()) / 1024**2
    delta_mib = float(delta_rows["delta_bytes"].mean()) / 1024**2
    saved_rate = float(delta_rows["saved_rate"].mean())
    cosine_error = 1.0 - float(delta_rows["reconstruction_cos"].mean())
    l2_error = float(delta_rows["reconstruction_l2"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    storage = [full_mib, delta_mib]
    bars = axes[0].bar(
        ["Full KV", "Q8 delta + scale"], storage, color=["#6B7280", "#2A9D6F"], width=0.58
    )
    for bar, value in zip(bars, storage):
        axes[0].annotate(
            f"{value:.2f} MiB",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_title(f"Logical KV footprint ({saved_rate * 100:.2f}% saved)")
    axes[0].set_ylabel("Mean footprint (MiB)")
    axes[0].set_ylim(0, max(storage) * 1.20)
    style_axis(axes[0])

    errors = [cosine_error, l2_error]
    bars = axes[1].bar(
        ["1 - mean cosine", "Mean L2"], errors, color=["#2878B5", "#C4573A"], width=0.58
    )
    for bar, value in zip(bars, errors):
        axes[1].annotate(
            f"{value:.3e}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_yscale("log")
    axes[1].set_title("Q8 delta reconstruction error")
    axes[1].set_ylabel("Error (log scale, lower is better)")
    style_axis(axes[1])
    fig.savefig(output_dir / "kv-compression-quality.png", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    results = pd.read_csv(output_dir / "kv_delta_results.csv")
    results = results[results["status"] == "ok"].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#303030",
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "text.color": "#202020",
        }
    )
    differences, summary = paired_differences(results)
    summary.to_csv(output_dir / "paired_ttft_summary.csv", index=False)
    draw_latency(results, output_dir)
    draw_paired_delta(differences, summary, output_dir)
    draw_compression(results, output_dir)


if __name__ == "__main__":
    main()
