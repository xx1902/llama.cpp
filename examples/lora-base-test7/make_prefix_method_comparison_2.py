#!/usr/bin/env python3
"""Create v1 vs adaptive_reuse_v2 prefix-method comparison artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import DATASET_LABELS, PREFIX_METHODS, PREFIX_METHODS_V2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OLD_DIR = SCRIPT_DIR / "output" / "all_prefix_5"
DEFAULT_NEW_DIR = SCRIPT_DIR / "output" / "optimized_prefix_5"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "optimized_prefix_5"


def load_data(dataset: str, old_dir: Path, new_dir: Path) -> pd.DataFrame:
    old_path = old_dir / f"{dataset}_prefix_method_figure_data.csv"
    new_path = new_dir / f"{dataset}_prefix_method_figure_data.csv"
    frames = []
    if old_path.is_file():
        frames.append(pd.read_csv(old_path))
    if new_path.is_file():
        frames.append(pd.read_csv(new_path))
    if not frames:
        raise FileNotFoundError(f"no v1/v2 data for {dataset}")
    frame = pd.concat(frames, ignore_index=True)
    catalog = dict([*PREFIX_METHODS, *PREFIX_METHODS_V2])
    frame = frame[frame["method_id"].isin(catalog)].copy()
    frame["method_order"] = frame["method_id"].map(
        {method_id: index for index, method_id in enumerate(catalog)}
    )
    return frame.sort_values("method_order").reset_index(drop=True)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "Microsoft YaHei",
        }
    )


def safe_float(row: pd.Series, column: str) -> float:
    value = row.get(column, math.nan)
    return float(value) if pd.notna(value) else math.nan


def make_figure(frame: pd.DataFrame, output_path: Path) -> None:
    labels = frame["method_label"].tolist()
    x = np.arange(len(frame))
    colors = ["#3b76c5", "#7d91ae", "#d58a2b", "#4b9368", "#c44b55", "#7b4fa3"]
    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.3], hspace=0.32)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    metrics = [
        ("mean_chunks_per_context", "Mean chunks per context", "Count", "{:.2f}"),
        ("mean_tokens_per_chunk", "Mean estimated tokens per chunk", "Tokens", "{:.1f}"),
        ("natural_boundary_coverage", "Natural-boundary coverage", "Ratio", "{:.2f}"),
        (
            "mean_tokens_lost_at_natural_edit_point",
            "Mean tokens lost at natural edit point",
            "Tokens",
            "{:.1f}",
        ),
    ]
    for ax, (column, title, ylabel, formatter) in zip(axes, metrics):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        ax.bar(x, values, color=colors[: len(values)], width=0.72)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        finite = values[np.isfinite(values)]
        upper = max(1.0, float(np.max(finite)) * 1.22 if finite.size else 1.0)
        ax.set_ylim(0, upper)
        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.text(index, value + upper * 0.025, formatter.format(value), ha="center", va="bottom", fontsize=8)

    table_ax = fig.add_subplot(grid[2, :])
    table_ax.axis("off")
    dataset_label = str(frame["dataset_label"].iloc[0])
    requests = int(pd.to_numeric(frame["runtime_requests"], errors="coerce").max())
    table_ax.set_title(f"Results: {dataset_label} / {requests} requests, v1 vs v2", loc="left", fontsize=12, pad=8)

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%" if np.isfinite(value) else "n/a"

    table_rows = []
    for _, row in frame.iterrows():
        speed = safe_float(row, "runtime_ttft_speedup_percent")
        half = safe_float(row, "runtime_ttft_ci95_half_percent")
        ttft = f"{speed:+.2f}% +/- {half:.2f}" if np.isfinite(speed) and np.isfinite(half) else "n/a"
        host = safe_float(row, "runtime_host_peak_mb")
        nodes = safe_float(row, "runtime_peak_nodes")
        table_rows.append(
            [
                str(row["method_label"]),
                pct(safe_float(row, "runtime_request_hit_rate")),
                pct(safe_float(row, "runtime_token_hit_rate")),
                pct(safe_float(row, "complete_prefix_repeat_rate")),
                ttft,
                f"{host:.1f} MiB" if np.isfinite(host) else "n/a",
                f"{nodes:.0f}" if np.isfinite(nodes) else "n/a",
            ]
        )
    table = table_ax.table(
        cellText=table_rows,
        colLabels=[
            "Method",
            "Runtime\nrequest hit",
            "Runtime\ntoken coverage",
            "Complete prefix\nhit",
            "TTFT change\n(positive = lower)",
            "Host peak",
            "Node peak",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.17, 0.12, 0.14, 0.14, 0.19, 0.12, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row_index == 0:
            cell.set_facecolor("#d8edf7")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#eef8fc")
    fig.suptitle(f"{dataset_label}: prefix splitting v1 vs Adaptive reuse v2", fontsize=14, weight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_report(frame: pd.DataFrame, output_path: Path, figure_name: str, request_limit: int) -> None:
    dataset = str(frame["dataset"].iloc[0])
    label = str(frame["dataset_label"].iloc[0])
    lines = [
        f"# {label} prefix splitting comparison 2",
        "",
        f"- Requests: `{request_limit}` for this comparison output.",
        "- Methods: the original five v1 methods plus `Adaptive reuse v2`.",
        "- Positive TTFT change means lower online TTFT than the matched full-prefill baseline.",
        "- v2 score: `abs(chunk_tokens - target) + 10 * future_nodes + 80 * prefetch_overflow - reuse_weight * reuse_boundary - 18 * natural_boundary`.",
        "- `reuse_weight = 48 + 64 * context_reaccess_rate + 64 * exact_prefix_rate`; v2 therefore adapts to workload reuse instead of using the v1 fixed bonus 72.",
        "",
        f"![{label} prefix splitting comparison 2]({figure_name})",
        "",
        "| Method | Runtime request hit | Runtime token coverage | Complete prefix hit | TTFT change | Host peak | Node peak | Mean chunks | Mean tokens/chunk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        speed = safe_float(row, "runtime_ttft_speedup_percent")
        half = safe_float(row, "runtime_ttft_ci95_half_percent")
        ttft = f"{speed:+.2f}% +/- {half:.2f}" if np.isfinite(speed) and np.isfinite(half) else "n/a"
        lines.append(
            f"| {row['method_label']} | {pct(safe_float(row, 'runtime_request_hit_rate'))} | "
            f"{pct(safe_float(row, 'runtime_token_hit_rate'))} | {pct(safe_float(row, 'complete_prefix_repeat_rate'))} | "
            f"{ttft} | {safe_float(row, 'runtime_host_peak_mb'):.1f} MiB | "
            f"{safe_float(row, 'runtime_peak_nodes'):.0f} | {safe_float(row, 'mean_chunks_per_context'):.2f} | "
            f"{safe_float(row, 'mean_tokens_per_chunk'):.1f} |"
        )
    lines.extend(
        [
            "",
            "## v2 interpretation",
            "",
            "- v2 is designed to reduce chunk count when context reaccess is sparse, align with historical reuse points when reaccess is frequent, and fit long parallel prefixes into the two-chunk prefetch budget.",
            "- The v2 row is not a claim that all negative TTFT disappears: long cross-LoRA prefixes can still pay KV materialization and uncached suffix cost.",
            "- Compare this report with the original report in `../all_prefix_5/` or `../opus100_prefix_100/`; the old outputs are intentionally unchanged.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%" if np.isfinite(value) else "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD_DIR)
    parser.add_argument("--new-dir", type=Path, default=DEFAULT_NEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-limit", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = load_data(args.dataset, args.old_dir, args.new_dir)
    figure_path = args.output_dir / f"{args.dataset}_prefix_method_effect_2.png"
    report_path = args.output_dir / f"{args.dataset.upper()}_PREFIX_METHOD_RESULTS_2.md"
    data_path = args.output_dir / f"{args.dataset}_prefix_method_figure_data_2.csv"
    configure_plots()
    make_figure(frame, figure_path)
    write_report(frame, report_path, figure_path.name, args.request_limit)
    frame.to_csv(data_path, index=False)
    print(f"figure: {figure_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
