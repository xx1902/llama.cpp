#!/usr/bin/env python3
"""Build the cross-dataset prefix-method summary for completed runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_results import DATASET_LABELS, PREFIX_METHODS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "all_prefix_5"


def pct(value: object) -> str:
    number = float(value)
    return f"{number * 100:.1f}%" if pd.notna(number) else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-limit", type=int, default=5)
    parser.add_argument(
        "--opus-dir",
        type=Path,
        default=SCRIPT_DIR / "output" / "opus100_prefix_100",
        help="Existing OPUS-100 figure-data directory to include in the overall report",
    )
    args = parser.parse_args()

    result_path = args.output_dir / "prefix_method_results.csv"
    frame = pd.read_csv(result_path)
    opus_runtime_path = args.opus_dir / "prefix_method_results.csv"
    if opus_runtime_path.is_file():
        frame = pd.concat([frame, pd.read_csv(opus_runtime_path)], ignore_index=True)
    structure_frames = []
    for dataset in frame["dataset"].drop_duplicates():
        path = args.output_dir / f"{dataset}_prefix_method_figure_data.csv"
        if path.is_file():
            structure_frames.append(pd.read_csv(path))
    if not structure_frames:
        raise FileNotFoundError("no per-dataset figure data found")
    opus_path = args.opus_dir / "opus100_parallel_prefix_method_figure_data.csv"
    if opus_path.is_file():
        structure_frames.append(pd.read_csv(opus_path))
    all_data = pd.concat(structure_frames, ignore_index=True)
    all_data.to_csv(args.output_dir / "prefix_method_summary.csv", index=False)
    runtime_frame = frame.copy()
    structural_columns = [
        "dataset",
        "dataset_label",
        "method_id",
        "method_label",
        "mean_chunks_per_context",
        "mean_tokens_per_chunk",
        "natural_boundary_coverage",
        "mean_tokens_lost_at_natural_edit_point",
        "structural_token_coverage",
        "complete_prefix_repeat_rate",
        "unique_nodes",
        "mean_prefix_tokens",
    ]
    structure_only = all_data[[column for column in structural_columns if column in all_data]]
    frame = structure_only.merge(
        runtime_frame,
        on=["dataset", "dataset_label", "method_id", "method_label"],
        how="left",
        validate="one_to_one",
    )

    lines = [
        "# Prefix-method experiments across real datasets",
        "",
        f"> This report covers `{len(frame['dataset'].unique())}` datasets and five boundary methods. The five non-OPUS datasets use `{args.request_limit}` requests each as a smoke-scale structural/runtime validation; OPUS-100 uses its completed 100-request run.",
        "",
        "## Runtime summary",
        "",
        "| Dataset | Method | Requests | Runtime request hit | Runtime token coverage | Complete prefix hit | TTFT change | Host peak (MiB) | Node peak | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = {item[0]: index for index, item in enumerate(PREFIX_METHODS)}
    frame["method_order"] = frame["method_id"].map(order)
    frame = frame.sort_values(["dataset", "method_order"])
    for _, row in frame.iterrows():
        speed = row.get("runtime_ttft_speedup_percent")
        half = row.get("runtime_ttft_ci95_half_percent")
        ttft = f"{float(speed):+.2f}% +/- {float(half):.2f}" if pd.notna(speed) and pd.notna(half) else "n/a"
        host = f"{float(row['runtime_host_peak_mb']):.1f}" if pd.notna(row.get("runtime_host_peak_mb")) else "n/a"
        nodes = f"{float(row['runtime_peak_nodes']):.0f}" if pd.notna(row.get("runtime_peak_nodes")) else "n/a"
        lines.append(
            f"| {row['dataset_label']} | {row['method_label']} | {int(row['runtime_requests'])} | "
            f"{pct(row.get('runtime_request_hit_rate'))} | {pct(row.get('runtime_token_hit_rate'))} | "
            f"{float(row['complete_prefix_repeat_rate']) * 100:.1f}% | {ttft} | {host} | {nodes} | {row.get('status', 'unknown')} |"
        )

    lines.extend(["", "## Per-dataset figures", ""])
    for dataset in frame["dataset"].drop_duplicates():
        label = DATASET_LABELS.get(dataset, dataset)
        figure_name = f"{dataset}_prefix_method_effect.png"
        report_name = f"{dataset.upper()}_PREFIX_METHOD_RESULTS.md"
        if dataset == "opus100_parallel":
            figure_name = f"../opus100_prefix_100/{figure_name}"
            report_name = f"../opus100_prefix_100/{report_name}"
        lines.extend(
            [
                f"### {label}",
                "",
                f"![{label} prefix-method effect]({figure_name})",
                "",
                f"- Detailed report: [`{report_name.split('/')[-1]}`]({report_name})",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Continuous datasets (MSC, Taskmaster, LMSYS-33K, ShareGPT52K) measure returning to growing conversation contexts; parallel datasets (XSum and OPUS-100) measure reuse of a shared source across different LoRA roles.",
            "- Positive TTFT change means the online prefix method is faster than its matched full-prefill baseline. A confidence interval crossing zero is inconclusive at this request count.",
            "- The physical adapters are routing artifacts for cache behavior; these runs do not claim semantic LoRA quality.",
            "- ShareGPT and XSum have longer prompts, so their node count and token-loss panels should be read together with the runtime table rather than as a pure quality ranking.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "D:\\anaconda\\envs\\qwen2.5_vl\\python.exe -B examples/lora-base-test7/run_prefix_method_experiments.py `",
            "  --datasets msc_continuous,taskmaster_continuous,lmsys_continuous,sharegpt_continuous,xsum_parallel `",
            "  --request-limit 5 --output-dir examples/lora-base-test7/output/all_prefix_5 `",
            "  --continue-on-failure",
            "```",
        ]
    )
    (args.output_dir / "ALL_PREFIX_METHOD_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "ALL_PREFIX_METHOD_RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
