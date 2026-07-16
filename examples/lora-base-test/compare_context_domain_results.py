"""Compare common-context and task-specific LoRA KV-delta experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_COLORS = {
    "Summary high": "#E4572E",
    "Countdown search/react": "#2E86AB",
    "Countdown search/optimal": "#5AA9C7",
    "Code v3/v2": "#2A9D8F",
    "Code v3/v1": "#75B79E",
    "Summary low": "#264653",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-summary", type=Path, required=True)
    parser.add_argument("--indomain-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def save(fig: plt.Figure, output_dir: Path, formats: list[str], dpi: int) -> None:
    for extension in formats:
        fig.savefig(
            output_dir / f"context_domain_comparison.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = pd.read_csv(args.common_summary)
    indomain = pd.read_csv(args.indomain_summary)
    keys = ["anchor_lora_name", "child_lora_name"]
    merged = common.merge(indomain, on=keys, suffixes=("_common", "_indomain"))
    merged["l2_change_rate"] = (
        merged["kv_l2_mean_indomain"] / merged["kv_l2_mean_common"] - 1.0
    )
    merged["cosine_distance_ratio"] = (
        merged["kv_cosine_distance_indomain"]
        / merged["kv_cosine_distance_common"]
    )
    merged.to_csv(args.output_dir / "context_domain_comparison.csv", index=False)

    labels = merged["pair_label_common"].tolist()
    colors = [PAIR_COLORS.get(label, "#777777") for label in labels]
    x = np.arange(len(merged))
    width = 0.34

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].bar(
        x - width / 2,
        merged["kv_l2_mean_common"],
        width,
        color="#B7B7B7",
        label="Common XSum context",
    )
    axes[0].bar(
        x + width / 2,
        merged["kv_l2_mean_indomain"],
        width,
        color=colors,
        label="Task-specific context",
    )
    axes[0].set_ylabel("Mean KV L2 difference")
    axes[0].set_title("Context domain changes KV magnitude")
    axes[0].legend(ncols=2)

    change_percent = merged["l2_change_rate"] * 100
    axes[1].bar(x, change_percent, color=colors)
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_ylabel("L2 change from common context (%)")
    axes[1].set_title("Domain sensitivity by adapter pair")
    for index, value in enumerate(change_percent):
        axes[1].text(
            index,
            value + (4 if value >= 0 else -4),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
    fig.tight_layout()
    save(fig, args.output_dir, [x.strip() for x in args.formats.split(",")], args.dpi)
    print(f"Saved comparison to: {args.output_dir}")


if __name__ == "__main__":
    main()
