#!/usr/bin/env python3
"""Compare off, sync, and deferred online cross-LoRA policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "off": "#6B7280",
    "sync": "#E76F51",
    "deferred": "#2A9D8F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "online_policy_comparison",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "grid.color": "#DCE3EA",
            "grid.alpha": 0.55,
            "axes.titleweight": "bold",
        }
    )


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png")
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def load_policy(base_dir: Path, policy: str) -> pd.DataFrame:
    path = base_dir / f"output_87_{policy}" / "online_request_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {policy} result: {path}")
    data = pd.read_csv(path)
    online = data[data["benchmark"] == "online"].copy()
    if online.empty:
        raise ValueError(f"No online rows in {path}")
    online["policy"] = policy
    return online


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, part in data.groupby("policy", sort=False):
        cross = part[part["cross_lora_prefix_match"] == 1]
        exact = part[part["exact_prefix_hit"] == 1]
        rows.append(
            {
                "policy": policy,
                "requests": len(part),
                "cross_lora_requests": len(cross),
                "exact_hit_requests": len(exact),
                "mean_ttft_ms": part["ttft_ms"].mean(),
                "median_ttft_ms": part["ttft_ms"].median(),
                "p95_ttft_ms": part["ttft_ms"].quantile(0.95),
                "cross_mean_ttft_ms": cross["ttft_ms"].mean() if len(cross) else np.nan,
                "exact_mean_ttft_ms": exact["ttft_ms"].mean() if len(exact) else np.nan,
                "mean_probe_ms": part["delta_probe_ms"].mean(),
                "mean_build_ms": part["delta_build_ms"].mean(),
                "mean_background_ms": part["delta_background_ms"].mean(),
                "total_probe_ms": part["delta_probe_ms"].sum(),
                "total_build_ms": part["delta_build_ms"].sum(),
                "total_background_ms": part["delta_background_ms"].sum(),
                "delta_build_ok": int(part["delta_build_ok"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("policy").loc[["off", "sync", "deferred"]].reset_index()


def plot_ttft(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms"]
    labels = ["Mean", "Median", "P95"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for index, row in summary.iterrows():
        values = [row[metric] for metric in metrics]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=row["policy"],
            color=COLORS[row["policy"]],
        )
        ax.bar_label(bars, labels=[f"{value:.0f}" for value in values], padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Online policy TTFT comparison")
    ax.legend(title="Policy")
    save(fig, output_dir, "01_policy_ttft")


def plot_request_types(summary: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cross = ax.bar(
        x - width / 2,
        summary["cross_mean_ttft_ms"],
        width,
        label="Cross-LoRA matched prefix",
        color="#F4A261",
    )
    exact = ax.bar(
        x + width / 2,
        summary["exact_mean_ttft_ms"],
        width,
        label="Exact same-LoRA hit",
        color="#3A86FF",
    )
    ax.bar_label(cross, fmt="%.0f", padding=3)
    ax.bar_label(exact, fmt="%.0f", padding=3)
    ax.set_xticks(x, summary["policy"])
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title("Cold cross-LoRA path versus hot exact-hit path")
    ax.legend()
    save(fig, output_dir, "02_request_type_ttft")


def plot_delta_cost(summary: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(summary))
    probe = summary["total_probe_ms"].to_numpy()
    build = summary["total_build_ms"].to_numpy()
    background = summary["total_background_ms"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x, probe, label="Foreground probe", color="#E9C46A")
    ax.bar(x, build, bottom=probe, label="Foreground build", color="#E15759")
    ax.bar(x, background, bottom=probe + build, label="Deferred/background", color="#2A9D8F")
    totals = probe + build + background
    for pos, total in zip(x, totals):
        ax.text(pos, total, f"{total / 1000:.1f}s", ha="center", va="bottom")
    ax.set_xticks(x, summary["policy"])
    ax.set_ylabel("Accumulated delta work (ms)")
    ax.set_title("Delta work is moved, not eliminated, by deferred execution")
    ax.legend()
    save(fig, output_dir, "03_delta_work")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([load_policy(args.base_dir, policy) for policy in ["off", "sync", "deferred"]])
    summary = summarize(combined)
    summary.to_csv(args.output_dir / "online_policy_summary.csv", index=False)
    setup_style()
    plot_ttft(summary, args.output_dir)
    plot_request_types(summary, args.output_dir)
    plot_delta_cost(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
