#!/usr/bin/env python3
"""Create a screenshot-style prefix-method figure for one real workload."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import (
    DATASET_LABELS,
    PREFIX_METHOD_CATALOG,
    PREFIX_METHODS,
    build_causal_lora_focus_profiles,
    estimated_tokens,
    natural_boundaries,
    read_jsonl,
    split_prefix_experiment_method,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKLOADS_ROOT = SCRIPT_DIR.parent / "lora-base-datasets" / "output" / "real_100"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "opus100_prefix_100"


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def structural_metrics(
    workload_path: Path,
    request_limit: int,
    methods: list[tuple[str, str]] | None = None,
    summary_path: Path | None = None,
) -> pd.DataFrame:
    requests = read_jsonl(workload_path)[:request_limit]
    lora_focus_profiles = build_causal_lora_focus_profiles(requests)
    prefixes_by_context: dict[str, list[str]] = {}
    for request in requests:
        context_id = str(request.get("context_id", ""))
        prefixes_by_context.setdefault(context_id, []).append(
            str(request.get("common_prefix_text", ""))
        )

    methods = methods or PREFIX_METHODS
    summary = {}
    if summary_path and summary_path.is_file():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    context_rate = float(
        summary.get(
            "request_level_context_reuse_rate",
            summary.get("request_level_context_reaccess_rate", 0.0),
        )
    )
    exact_rate = float(summary.get("request_level_exact_prefix_reuse_rate", 0.0))
    rows: list[dict[str, object]] = []
    for method_id, method_label in methods:
        seen_nodes: set[str] = set()
        seen_prefixes: set[str] = set()
        total_chunks = 0
        total_chunk_tokens = 0
        total_prefix_tokens = 0
        reused_tokens = 0
        complete_hits = 0
        natural_ends = 0
        internal_ends = 0
        natural_covered_tokens = 0
        for request_index, request in enumerate(requests):
            text = str(request.get("common_prefix_text", ""))
            context_id = str(request.get("context_id", ""))
            reuse_points = {
                len(prefix)
                for prefix in prefixes_by_context.get(context_id, [])
                if len(prefix) < len(text) and text.startswith(prefix)
            }
            chunks, _ = split_prefix_experiment_method(
                request,
                method_id,
                reuse_points,
                context_reaccess_rate=context_rate,
                exact_prefix_rate=exact_rate,
                lora_focus_profile=lora_focus_profiles[request_index],
            )
            chunk_tokens = [estimated_tokens(chunk) for chunk in chunks]
            prefix_tokens = estimated_tokens(text)
            total_chunks += len(chunks)
            total_chunk_tokens += sum(chunk_tokens)
            total_prefix_tokens += prefix_tokens

            boundaries = natural_boundaries(
                text, list(request.get("prefix_segments") or [])
            )
            offset = 0
            for index, chunk in enumerate(chunks[:-1]):
                offset += len(chunk)
                is_natural = offset in boundaries
                internal_ends += 1
                natural_ends += int(is_natural)
                if is_natural:
                    natural_covered_tokens += chunk_tokens[index]

            parent = "root"
            node_ids: list[str] = []
            for chunk in chunks:
                node_id = hashlib.sha256(
                    (parent + "\0" + chunk).encode("utf-8")
                ).hexdigest()[:24]
                node_ids.append(node_id)
                parent = node_id
            reused_tokens += sum(
                tokens
                for node_id, tokens in zip(node_ids, chunk_tokens)
                if node_id in seen_nodes
            )
            prefix_hash = str(request.get("common_prefix_hash", ""))
            complete_hits += int(prefix_hash in seen_prefixes)
            seen_nodes.update(node_ids)
            seen_prefixes.add(prefix_hash)

        mean_prefix_tokens = total_prefix_tokens / max(1, len(requests))
        rows.append(
            {
                "method_id": method_id,
                "method_label": method_label,
                "mean_chunks_per_context": total_chunks / max(1, len(requests)),
                "mean_tokens_per_chunk": total_chunk_tokens / max(1, total_chunks),
                "natural_boundary_coverage": natural_ends / max(1, internal_ends),
                # A natural edit point is an internal chunk boundary that also
                # ends a source segment/sentence. Tokens in chunks after the
                # last such point are not structurally reusable at that point.
                "mean_tokens_lost_at_natural_edit_point": (
                    total_prefix_tokens - natural_covered_tokens
                )
                / max(1, len(requests)),
                "structural_token_coverage": reused_tokens / max(1, total_chunk_tokens),
                "complete_prefix_repeat_rate": complete_hits / max(1, len(requests)),
                "unique_nodes": len(seen_nodes),
                "mean_prefix_tokens": mean_prefix_tokens,
            }
        )
    return pd.DataFrame(rows)


def load_runtime_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"method_id", "method_label", "runtime_ttft_speedup_percent"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"runtime result is missing columns: {missing}")
    return frame


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


def make_figure(structure: pd.DataFrame, runtime: pd.DataFrame, output_path: Path) -> None:
    frame = structure.merge(
        runtime,
        on=["method_id", "method_label"],
        how="left",
        validate="one_to_one",
    )
    catalog = dict(PREFIX_METHOD_CATALOG)
    order = [item for item in [*catalog] if item in set(frame["method_id"])]
    frame["method_order"] = frame["method_id"].map({item: i for i, item in enumerate(order)})
    frame = frame.sort_values("method_order").reset_index(drop=True)
    labels = frame["method_label"].tolist()
    x = np.arange(len(frame))
    colors = ["#3b76c5", "#7d91ae", "#d58a2b", "#4b9368", "#c44b55", "#7b4fa3"]

    fig = plt.figure(figsize=(11.5, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.28], hspace=0.32)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    metrics = [
        ("mean_chunks_per_context", "Mean chunks per context", "Count", 1.0, "{:.2f}"),
        ("mean_tokens_per_chunk", "Mean estimated tokens per chunk", "Tokens", 1.0, "{:.1f}"),
        ("natural_boundary_coverage", "Natural-boundary coverage", "Ratio", 1.0, "{:.2f}"),
        (
            "mean_tokens_lost_at_natural_edit_point",
            "Mean tokens lost at natural edit point",
            "Tokens",
            1.0,
            "{:.1f}",
        ),
    ]
    for ax, (column, title, ylabel, scale, fmt) in zip(axes, metrics):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float) * scale
        ax.bar(x, values, color=colors, width=0.72)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        upper = max(1.0, float(np.nanmax(values)) * 1.22 if len(values) else 1.0)
        ax.set_ylim(0, upper)
        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.text(index, value + upper * 0.025, fmt.format(value), ha="center", va="bottom", fontsize=8)

    table_ax = fig.add_subplot(grid[2, :])
    table_ax.axis("off")
    table_ax.set_title(
        f"Results: {runtime['dataset_label'].iloc[0]} / {int(runtime['runtime_requests'].max()) if runtime['runtime_requests'].notna().any() else 0} requests",
        loc="left",
        fontsize=12,
        pad=8,
    )
    def pct(value: object) -> str:
        number = float(value)
        return f"{number * 100:.1f}%" if math.isfinite(number) else "n/a"

    def speedup(row: pd.Series) -> str:
        value = float(row.get("runtime_ttft_speedup_percent", math.nan))
        half = float(row.get("runtime_ttft_ci95_half_percent", math.nan))
        if math.isfinite(value) and math.isfinite(half):
            return f"{value:+.2f}% +/- {half:.2f}"
        return "n/a"

    table_rows = []
    for _, row in frame.iterrows():
        host_value = row.get("runtime_host_peak_mb", 0.0)
        host_peak = float(host_value) if pd.notna(host_value) else math.nan
        node_value = row.get("runtime_peak_nodes", 0.0)
        node_peak = float(node_value) if pd.notna(node_value) else math.nan
        table_rows.append(
            [
                str(row["method_label"]),
                pct(row.get("runtime_request_hit_rate", math.nan)),
                pct(row.get("runtime_token_hit_rate", math.nan)),
                pct(row["complete_prefix_repeat_rate"]),
                speedup(row),
                f"{host_peak:.1f} MiB" if math.isfinite(host_peak) else "n/a",
                f"{node_peak:.0f}" if math.isfinite(node_peak) else "n/a",
            ]
        )
    columns = [
        "Method",
        "Runtime\nrequest hit",
        "Runtime\ntoken coverage",
        "Complete prefix\nhit",
        "TTFT change\n(positive = lower)",
        "Host peak",
        "Node peak",
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.16, 0.13, 0.15, 0.14, 0.18, 0.12, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row_index == 0:
            cell.set_facecolor("#d8edf7")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#eef8fc")
    title = DATASET_LABELS.get(runtime["dataset"].iloc[0], runtime["dataset"].iloc[0])
    fig.suptitle(f"{title}: structural validation and online prefix-method effects", fontsize=14, weight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    structure: pd.DataFrame,
    runtime: pd.DataFrame,
    output_path: Path,
    figure_path: Path,
    dataset: str,
    request_limit: int,
) -> None:
    frame = structure.merge(runtime, on=["method_id", "method_label"], how="left")
    catalog = dict(PREFIX_METHOD_CATALOG)
    order = {item: index for index, item in enumerate(catalog)}
    frame["order"] = frame["method_id"].map(order)
    frame = frame.sort_values("order")
    lines = [
        f"# {DATASET_LABELS.get(dataset, dataset)} prefix-method example",
        "",
        f"- Requests: `{request_limit}`",
        "- Workload form: determined by the generated workload; continuous datasets revisit growing conversation context, while parallel datasets reuse a source across LoRA roles.",
        "- TTFT change: `(matched full-prefill TTFT - online TTFT) / matched full-prefill TTFT`; positive means lower TTFT.",
        "- Natural-boundary coverage: internal chunk ends that coincide with a source segment/sentence boundary divided by all internal chunk ends.",
        "- Mean tokens lost at natural edit point: prefix tokens not covered by chunks ending at an internal natural boundary. The final chunk is intentionally counted as not reusable at an edit point.",
        "",
        f"![{DATASET_LABELS.get(dataset, dataset)} prefix-method effect]({figure_path.name})",
        "",
        "| Method | Runtime request hit | Runtime token coverage | Complete prefix hit | TTFT change | Host peak | Node peak |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        value = float(row["runtime_ttft_speedup_percent"])
        half = float(row["runtime_ttft_ci95_half_percent"])
        ttft = f"{value:+.2f}% +/- {half:.2f}" if math.isfinite(value) and math.isfinite(half) else "n/a"
        request_hit = row["runtime_request_hit_rate"]
        token_hit = row["runtime_token_hit_rate"]
        host_peak = row["runtime_host_peak_mb"]
        node_peak = row["runtime_peak_nodes"]
        lines.append(
            f"| {row['method_label']} | {float(request_hit) * 100:.1f}% | "
            f"{float(token_hit) * 100:.1f}% | {float(row['complete_prefix_repeat_rate']) * 100:.1f}% | "
            f"{ttft} | {float(host_peak):.1f} MiB | {float(node_peak):.0f} |"
            if pd.notna(request_hit) and pd.notna(token_hit) and pd.notna(host_peak) and pd.notna(node_peak)
            else f"| {row['method_label']} | n/a | n/a | {float(row['complete_prefix_repeat_rate']) * 100:.1f}% | {ttft} | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## Structural metrics",
            "",
            "| Method | Mean chunks/context | Mean tokens/chunk | Natural-boundary coverage | Mean tokens lost |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['method_label']} | {float(row['mean_chunks_per_context']):.2f} | "
            f"{float(row['mean_tokens_per_chunk']):.1f} | {float(row['natural_boundary_coverage']):.2f} | "
            f"{float(row['mean_tokens_lost_at_natural_edit_point']):.1f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="opus100_parallel")
    parser.add_argument("--request-limit", type=int, default=100)
    parser.add_argument("--workloads-root", type=Path, default=DEFAULT_WORKLOADS_ROOT)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional filename suffix such as 1_1, 2_1, or 3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    results_path = args.results or output_dir / "prefix_method_results.csv"
    workload_path = args.workloads_root / args.dataset / "grouped" / "grouped_requests.jsonl"
    runtime = load_runtime_results(results_path)
    runtime = runtime[runtime["dataset"] == args.dataset].copy()
    catalog = dict(PREFIX_METHOD_CATALOG)
    methods = [(method_id, catalog[method_id]) for method_id in runtime["method_id"] if method_id in catalog]
    if not methods:
        raise ValueError(f"no known runtime methods for {args.dataset}")
    structure = structural_metrics(
        workload_path,
        args.request_limit,
        methods=methods,
        summary_path=args.workloads_root / args.dataset / "summary.json",
    )
    suffix = f"_{args.suffix}" if args.suffix else ""
    figure_path = output_dir / f"{args.dataset}_prefix_method_effect{suffix}.png"
    report_path = output_dir / f"{args.dataset.upper()}_PREFIX_METHOD_RESULTS{suffix}.md"
    configure_plots()
    make_figure(structure, runtime, figure_path)
    write_markdown(structure, runtime, report_path, figure_path, args.dataset, args.request_limit)
    structure.merge(runtime, on=["method_id", "method_label"], how="left").to_csv(
        output_dir / f"{args.dataset}_prefix_method_figure_data{suffix}.csv", index=False
    )
    print(f"figure: {figure_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
