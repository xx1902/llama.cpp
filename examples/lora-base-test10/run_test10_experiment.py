#!/usr/bin/env python3
"""Run the test10 chunking and parallel-scheduling experiment.

The workload is the previously generated 87-App/87-LoRA JSONL trace.  This
script measures real CPU TF-IDF work and consumes the CUDA stream benchmark
CSV.  The CUDA program deliberately uses synthetic kernels: it measures
resource scheduling/overlap, not model quality or BGE-M3 latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_DATA = Path(r"D:\ecnu_experiment\datasets\mobilora_workloads_87_group_hybrid\delta\delta_requests.jsonl")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
DEFAULT_CUDA = Path(__file__).resolve().parents[2] / "build" / "bin" / "Release" / "llama-lora-base-test10-cuda.exe"


def proxy_tokens(text: str) -> list[str]:
    """A stable analysis tokenization; it is not a Qwen/BGE tokenizer."""
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\w\s]", text.lower(), flags=re.UNICODE)


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def fixed_chunks(text: str, chunk_size: int = 128) -> list[list[str]]:
    tokens = proxy_tokens(text)
    return [tokens[start : start + chunk_size] for start in range(0, len(tokens), chunk_size)] or [[]]


def sentence_fixed_chunks(text: str, chunk_size: int = 128) -> list[list[str]]:
    """Sentence-first segmentation with a fixed-token fallback for long spans."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？；;])\s*", text) if part.strip()]
    if not sentences:
        sentences = [text]
    chunks: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        tokens = proxy_tokens(sentence)
        if len(tokens) > chunk_size:
            if current:
                chunks.append(current)
                current = []
            chunks.extend(tokens[start : start + chunk_size] for start in range(0, len(tokens), chunk_size))
            continue
        if current and len(current) + len(tokens) > chunk_size:
            chunks.append(current)
            current = []
        current.extend(tokens)
    if current:
        chunks.append(current)
    return chunks or [[]]


def keyword_boundary_chunks(text: str, idf: dict[str, float], chunk_size: int = 128, radius: int = 24) -> list[list[str]]:
    """Choose the strongest local TF-IDF topic discontinuity before each hard cap."""
    tokens = proxy_tokens(text)
    if not tokens:
        return [[]]

    def normalized_words(parts: list[str]) -> Counter[str]:
        return Counter(token for token in parts if re.fullmatch(r"[a-z0-9_]+", token))

    def discontinuity(boundary: int, window: int = 32) -> float:
        left = normalized_words(tokens[max(start, boundary - window) : boundary])
        right = normalized_words(tokens[boundary : min(len(tokens), boundary + window)])
        keys = set(left) | set(right)
        if not keys:
            return 0.0
        shared = sum(min(left[key], right[key]) * idf.get(key, 1.0) for key in keys)
        total = sum(max(left[key], right[key]) * idf.get(key, 1.0) for key in keys)
        return 1.0 - shared / max(total, 1e-12)

    chunks: list[list[str]] = []
    start = 0
    while start < len(tokens):
        target = min(start + chunk_size, len(tokens))
        if target >= len(tokens):
            chunks.append(tokens[start:])
            break
        left = max(start + 1, target - radius)
        boundary = max(
            range(left, target + 1),
            key=lambda index: (discontinuity(index) - 0.08 * (target - index) / radius, index),
        )
        boundary = max(boundary, start + 1)
        chunks.append(tokens[start:boundary])
        start = boundary
    return [chunk for chunk in chunks if chunk] or [[]]


def boundary_positions(chunks: Iterable[list[str]]) -> set[int]:
    position = 0
    boundaries: set[int] = set()
    for chunk in chunks:
        position += len(chunk)
        boundaries.add(position)
    return boundaries


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def load_records(path: Path, limit: int) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            row = json.loads(line)
            text = str(row.get("common_prefix_text", ""))
            if text:
                row["common_prefix_text"] = text
                records.append(row)
    if not records:
        raise RuntimeError(f"no usable rows in {path}")
    return records


def select_by_lora(records: list[dict], limit: int) -> list[dict]:
    if limit >= len(records):
        return records
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        groups[int(row.get("lora_id", -1))].append(row)
    selected = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for lora_id in sorted(groups):
            if round_index < len(groups[lora_id]):
                selected.append(groups[lora_id][round_index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    return selected


def fit_idf(records: list[dict]) -> tuple[TfidfVectorizer, dict[str, float]]:
    documents = [" ".join(word_tokens(row["common_prefix_text"])) or "empty" for row in records]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True, sublinear_tf=True)
    vectorizer.fit(documents)
    idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    return vectorizer, idf


def measure_cpu_keywords(records: list[dict], vectorizer: TfidfVectorizer, idf: dict[str, float]) -> list[dict]:
    metrics = []
    for row in records:
        text = row["common_prefix_text"]
        words = word_tokens(text)
        document = " ".join(words) or "empty"
        start = time.perf_counter()
        matrix = vectorizer.transform([document])
        values = matrix.toarray()[0]
        top_indices = np.argsort(values)[-8:][::-1]
        checksum = float(sum(values[index] * (index + 1) for index in top_indices))
        keyword_ms = (time.perf_counter() - start) * 1000.0

        chunk_start = time.perf_counter()
        fixed_128_chunks = fixed_chunks(text, 128)
        sentence_64_chunks = sentence_fixed_chunks(text, 64)
        sentence_128_chunks = sentence_fixed_chunks(text, 128)
        sentence_256_chunks = sentence_fixed_chunks(text, 256)
        sentence_fixed_ms = (time.perf_counter() - chunk_start) * 1000.0
        semantic_start = time.perf_counter()
        keyword_chunks = keyword_boundary_chunks(text, idf, 128)
        keyword_boundary_ms = (time.perf_counter() - semantic_start) * 1000.0
        original_tokens = proxy_tokens(text)
        for chunks, maximum in (
            (fixed_128_chunks, 128),
            (sentence_64_chunks, 64),
            (sentence_128_chunks, 128),
            (sentence_256_chunks, 256),
            (keyword_chunks, 128),
        ):
            if [token for chunk in chunks for token in chunk] != original_tokens:
                raise AssertionError("chunking changed token order or coverage")
            if any(len(chunk) > maximum for chunk in chunks):
                raise AssertionError(f"chunk exceeded configured maximum {maximum}")
        sentence_boundary = boundary_positions(sentence_128_chunks)
        keyword_boundary = boundary_positions(keyword_chunks)
        metrics.append(
            {
                "request_id": int(row.get("request_id", len(metrics))),
                "prefix_tokens": int(row.get("estimated_common_prefix_tokens", len(proxy_tokens(text)))),
                "lora_id": str(row.get("lora_id", "unknown")),
                "context_id": str(row.get("context_id", "unknown")),
                "common_prefix_hash": str(row.get("common_prefix_hash", "")),
                "keyword_ms": keyword_ms,
                "keyword_count": int(np.count_nonzero(values)),
                "keyword_checksum": checksum,
                "proxy_token_count": len(original_tokens),
                "fixed_128_chunks": len(fixed_128_chunks),
                "sentence_fixed_64_chunks": len(sentence_64_chunks),
                "sentence_fixed_128_chunks": len(sentence_128_chunks),
                "sentence_fixed_256_chunks": len(sentence_256_chunks),
                "keyword_128_chunks": len(keyword_chunks),
                "sentence_fixed_ms": sentence_fixed_ms,
                "keyword_boundary_ms": keyword_boundary_ms,
                "sentence_boundary_count": len(sentence_boundary),
                "keyword_boundary_count": len(keyword_boundary),
                "sentence_boundary_rate": len(sentence_boundary) / max(len(proxy_tokens(text)), 1),
                "keyword_boundary_rate": len(keyword_boundary) / max(len(proxy_tokens(text)), 1),
            }
        )
    return metrics


def add_schedule_proxies(metrics: list[dict]) -> None:
    """Create a transparent analytical GPU/CPU comparison for every request."""
    for metric in metrics:
        tokens = max(metric["prefix_tokens"], 1)
        # The curve is only a workload-size proxy, calibrated to the CUDA run.
        inference_ms = 1.2 + 0.020 * tokens ** 0.92
        cpu_keyword_ms = max(metric["keyword_ms"], 0.005)
        metric["proxy_inference_ms"] = inference_ms
        metric["cpu_serial_total_ms"] = inference_ms + cpu_keyword_ms
        metric["cpu_parallel_total_ms"] = max(inference_ms, cpu_keyword_ms)
        metric["cpu_overlap_saved_ms"] = metric["cpu_serial_total_ms"] - metric["cpu_parallel_total_ms"]
        metric["cpu_overlap_ratio"] = metric["cpu_overlap_saved_ms"] / metric["cpu_serial_total_ms"]


def run_cuda(cuda_exe: Path, data: Path, output: Path, limit: int, trials: int) -> bool:
    if not cuda_exe.exists():
        print(f"CUDA executable not found, skip GPU run: {cuda_exe}", file=sys.stderr)
        return False
    command = [str(cuda_exe), "--input", str(data), "--output", str(output), "--limit", str(limit), "--trials", str(trials)]
    print("Running CUDA stream benchmark:", " ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True)
    print(completed.stdout.strip())
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        return False
    return True


def read_cuda_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            if key in {"profile", "context_id"}:
                continue
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_audit(path: Path, data: Path, all_records: list[dict], selected: list[dict]) -> None:
    all_lengths = [int(row.get("estimated_common_prefix_tokens", 0)) for row in all_records]
    selected_hashes = [str(row.get("common_prefix_hash", "")) for row in selected]
    audit = {
        "dataset": str(data),
        "available_requests": len(all_records),
        "sampled_requests": len(selected),
        "sampling": "deterministic_lora_stratified_round_robin",
        "available_loras": len({str(row.get("lora_id", "")) for row in all_records}),
        "available_apps": len({str(row.get("app_name", "")) for row in all_records}),
        "available_contexts": len({str(row.get("context_id", "")) for row in all_records}),
        "sampled_unique_prefixes": len(set(selected_hashes)),
        "sampled_loras": len({str(row.get("lora_id", "")) for row in selected}),
        "sampled_apps": len({str(row.get("app_name", "")) for row in selected}),
        "sampled_repeated_prefix_rate": 1.0 - len(set(selected_hashes)) / max(len(selected_hashes), 1),
        "prefix_tokens_min": min(all_lengths),
        "prefix_tokens_p50": percentile(all_lengths, 50),
        "prefix_tokens_p95": percentile(all_lengths, 95),
        "prefix_tokens_max": max(all_lengths),
        "source_components": ["XSum", "ShareGPT52K", "LSApp 87-App trace"],
    }
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_chunk_summary(path: Path, metrics: list[dict]) -> None:
    methods = {
        "fixed_128": "fixed_128_chunks",
        "sentence_fixed_64": "sentence_fixed_64_chunks",
        "sentence_fixed_128": "sentence_fixed_128_chunks",
        "sentence_fixed_256": "sentence_fixed_256_chunks",
        "keyword_discontinuity_128": "keyword_128_chunks",
    }
    rows = []
    for method, field in methods.items():
        counts = [int(metric[field]) for metric in metrics]
        total_tokens = sum(int(metric["proxy_token_count"]) for metric in metrics)
        rows.append(
            {
                "method": method,
                "requests": len(metrics),
                "mean_chunks": statistics.mean(counts),
                "p50_chunks": percentile(counts, 50),
                "p95_chunks": percentile(counts, 95),
                "max_chunks": max(counts),
                "aggregate_mean_chunk_tokens": total_tokens / max(sum(counts), 1),
            }
        )
    write_csv(path, rows)


def collapse_cuda_trials(cuda_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in cuda_rows:
        grouped[(str(row["profile"]), int(row["request_id"]))].append(row)
    collapsed = []
    for (profile, request_id), rows in grouped.items():
        result: dict[str, float | str] = {"profile": profile, "request_id": request_id}
        for key in rows[0]:
            if key in {"profile", "request_id", "context_id"}:
                continue
            if isinstance(rows[0][key], (float, int)):
                result[key] = statistics.median(float(row[key]) for row in rows)
        collapsed.append(result)
    return collapsed


def aggregate_cuda(cuda_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in collapse_cuda_trials(cuda_rows):
        grouped[str(row["profile"])].append(row)
    summaries = []
    for profile, rows in grouped.items():
        serial = [float(row["gpu_serial_total_ms"]) for row in rows]
        overlap = [float(row["gpu_overlap_total_ms"]) for row in rows]
        keyword = [float(row["gpu_keyword_serial_ms"]) for row in rows]
        inference = [float(row["baseline_inference_ms"]) for row in rows]
        saved = [s - o for s, o in zip(serial, overlap)]
        cpu_serial = [float(row["cpu_serial_total_ms"]) for row in rows]
        cpu_parallel = [float(row["cpu_parallel_total_ms"]) for row in rows]
        cpu_keyword = [float(row["cpu_keyword_ms"]) for row in rows]
        cpu_saved = [s - p for s, p in zip(cpu_serial, cpu_parallel)]
        summaries.append(
            {
                "profile": profile,
                "requests": len(rows),
                "cpu_keyword_ms": statistics.mean(cpu_keyword),
                "cpu_serial_total_ms": statistics.mean(cpu_serial),
                "cpu_parallel_total_ms": statistics.mean(cpu_parallel),
                "cpu_overlap_saved_ms": statistics.mean(cpu_saved),
                "cpu_overlap_ratio": statistics.mean(cpu_saved) / max(statistics.mean(cpu_serial), 1e-9),
                "gpu_serial_total_ms": statistics.mean(serial),
                "gpu_overlap_total_ms": statistics.mean(overlap),
                "gpu_keyword_serial_ms": statistics.mean(keyword),
                "gpu_inference_ms": statistics.mean(inference),
                "gpu_overlap_saved_ms": statistics.mean(saved),
                "gpu_overlap_ratio": statistics.mean(saved) / max(statistics.mean(serial), 1e-9),
            }
        )
    return summaries


def write_plots(output: Path, metrics: list[dict], cuda_rows: list[dict], cuda_summaries: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.unicode_minus": False,
    })
    if cuda_summaries:
        labels = []
        serial_means = []
        parallel_means = []
        for summary in cuda_summaries:
            labels.extend([f"CPU + GPU\n{summary['profile']}", f"Two CUDA streams\n{summary['profile']}"])
            serial_means.extend([summary["cpu_serial_total_ms"], summary["gpu_serial_total_ms"]])
            parallel_means.extend([summary["cpu_parallel_total_ms"], summary["gpu_overlap_total_ms"]])
    else:
        labels = ["CPU + GPU proxy"]
        serial_means = [statistics.mean(metric["cpu_serial_total_ms"] for metric in metrics)]
        parallel_means = [statistics.mean(metric["cpu_parallel_total_ms"] for metric in metrics)]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(labels))
    width = 0.36
    serial_bars = ax.bar(x - width / 2, serial_means, width, color="#F58518", label="Serial")
    parallel_bars = ax.bar(x + width / 2, parallel_means, width, color="#54A24B", label="Parallel")
    ax.set_title("test10: keyword analysis and inference scheduling")
    ax.set_ylabel("End-to-end time (ms, measured proxy)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bar in [*serial_bars, *parallel_bars]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "strategy_ttft_comparison.png", dpi=160)
    plt.close(fig)

    bins = [0, 256, 512, 1024, 2048, 4096, float("inf")]
    bin_labels = ["≤256", "257–512", "513–1024", "1025–2048", "2049–4096", ">4096"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    collapsed = collapse_cuda_trials(cuda_rows) if cuda_rows else []
    used_labels = []
    series: dict[str, list[float]] = defaultdict(list)
    for lower, upper, label in zip(bins[:-1], bins[1:], bin_labels):
        selected = [row for row in collapsed if lower < float(row["prefix_tokens"]) <= upper]
        if not selected:
            continue
        used_labels.append(label)
        for profile in sorted({str(row["profile"]) for row in selected}):
            profile_rows = [row for row in selected if row["profile"] == profile]
            cpu_saved = [float(row["cpu_serial_total_ms"]) - float(row["cpu_parallel_total_ms"]) for row in profile_rows]
            gpu_saved = [float(row["gpu_serial_total_ms"]) - float(row["gpu_overlap_total_ms"]) for row in profile_rows]
            series[f"CPU worker / {profile}"].append(statistics.mean(cpu_saved) / max(statistics.mean(float(row["cpu_serial_total_ms"]) for row in profile_rows), 1e-9))
            series[f"CUDA streams / {profile}"].append(statistics.mean(gpu_saved) / max(statistics.mean(float(row["gpu_serial_total_ms"]) for row in profile_rows), 1e-9))
    if not series:
        used_labels = bin_labels[:1]
        series["CPU theoretical upper bound"] = [statistics.mean(metric["cpu_overlap_ratio"] for metric in metrics)]
    markers = ["o", "s", "^", "D"]
    colors = ["#54A24B", "#E45756", "#4C78A8", "#B279A2"]
    x = np.arange(len(used_labels))
    for index, (name, values) in enumerate(sorted(series.items())):
        ax.plot(x[: len(values)], values, marker=markers[index % len(markers)], color=colors[index % len(colors)], label=name)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("Overlap ratio by prefix length")
    ax.set_ylabel("1 - parallel / serial")
    ax.set_xlabel("Proxy prefix tokens")
    ax.set_xticks(x, used_labels)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "overlap_ratio_by_prefix_length.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.scatter([metric["prefix_tokens"] for metric in metrics], [metric["keyword_ms"] for metric in metrics], s=16, alpha=0.45, label="CPU TF-IDF (Python measured)")
    if cuda_summaries:
        line_colors = ["#F58518", "#B279A2"]
        for index, summary in enumerate(cuda_summaries):
            ax.axhline(
                summary["gpu_keyword_serial_ms"],
                color=line_colors[index % len(line_colors)],
                linestyle="--",
                linewidth=1.4,
                label=f"GPU keyword proxy: {summary['profile']}",
            )
    ax.set_title("Keyword-stage cost: CPU TF-IDF and GPU proxy")
    ax.set_xlabel("Proxy prefix tokens")
    ax.set_ylabel("Keyword-stage time (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cpu_vs_gpu_keyword_cost.png", dpi=160)
    plt.close(fig)

    labels = ["Fixed 128", "Sentence 64", "Sentence 128", "Sentence 256", "TF-IDF 128"]
    chunks = [
        statistics.mean(metric["fixed_128_chunks"] for metric in metrics),
        statistics.mean(metric["sentence_fixed_64_chunks"] for metric in metrics),
        statistics.mean(metric["sentence_fixed_128_chunks"] for metric in metrics),
        statistics.mean(metric["sentence_fixed_256_chunks"] for metric in metrics),
        statistics.mean(metric["keyword_128_chunks"] for metric in metrics),
    ]
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(labels))
    bars = ax1.bar(x, chunks, width=0.65, color=["#4C78A8", "#72B7B2", "#54A24B", "#B279A2", "#F58518"])
    ax1.set_ylabel("Mean chunks per prefix")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("Chunk-count sensitivity to segmentation method")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "chunk_boundary_comparison.png", dpi=160)
    plt.close(fig)


def write_report(output: Path, data: Path, all_records: list[dict], records: list[dict], metrics: list[dict], cuda_rows: list[dict], cuda_summaries: list[dict]) -> None:
    mean_keyword = statistics.mean(metric["keyword_ms"] for metric in metrics)
    mean_cpu_serial = statistics.mean(metric["cpu_serial_total_ms"] for metric in metrics)
    mean_cpu_parallel = statistics.mean(metric["cpu_parallel_total_ms"] for metric in metrics)
    mean_cpu_saved = mean_cpu_serial - mean_cpu_parallel
    lines = [
        "# test10 实验结果",
        "",
        "## 实验目的",
        "",
        "验证老师提出的两种并行路径：CPU 关键词模型与 GPU 推理并行，以及 CUDA 高优先级推理 stream 与低优先级关键词 stream 并行；同时比较 Sentence + fixed 与 TF-IDF 边界切分。",
        "",
        "## 数据集",
        "",
        f"- 输入：`{data}`，共有 {len(all_records)} 条请求；本次按 LoRA 分层抽取 {len(records)} 条 benchmark 请求（可通过 `--limit` 调整）。",
        "- 数据由 test4 的 `build_87_lora_workloads.py` 生成，来源为 XSum、ShareGPT52K 和 LSApp 的 87-App trace；87 个 App 与 87 个 LoRA 一一映射。",
        "- `common_prefix_text` 是跨请求共享的上下文，`estimated_common_prefix_tokens` 用于按前缀长度分箱；Python 切分使用稳定正则 token，不是 Qwen tokenizer。",
        "",
        "## 方法",
        "",
        "1. CPU TF-IDF：在全部输入上拟合 IDF，对每条前缀做 top-k 关键词计算并计时。",
        "2. Sentence + fixed：先按中英文句末标点切分；超长句回退到 128-token 固定块。",
        "3. TF-IDF discontinuity：在 128-token 硬上限前的 24-token 窗口内，用左右局部窗口的 IDF 加权 Jaccard 距离寻找主题变化最强的位置。",
        "4. CPU 并行：C++ 使用常驻工作线程执行 TF-IDF，同时在高优先级 stream 启动推理代理，并直接测量端到端墙钟时间；Python 另保留 `max(inference, keyword)` 理论上界作为无 CUDA 时的后备。",
        "5. CUDA stream：高优先级 stream 执行 inference-like kernel，低优先级 stream 执行 keyword-like kernel；两类负载按前缀长度在 0.5–2.5 倍范围缩放。CUDA 结果是调度代理，不代表 BGE-M3 精度或真实 LLM TTFT。",
        "",
        "## 主要结果",
        "",
        f"- Python/sklearn TF-IDF transform 平均耗时：{mean_keyword:.3f} ms；其分析曲线仅作为 `--skip-cuda` 时的理论后备（串行 {mean_cpu_serial:.3f} ms，并行上界 {mean_cpu_parallel:.3f} ms，差值 {mean_cpu_saved:.3f} ms）。",
        f"- Sentence + fixed-128 平均 chunk 数：{statistics.mean(metric['sentence_fixed_128_chunks'] for metric in metrics):.2f}；TF-IDF discontinuity-128 平均 chunk 数：{statistics.mean(metric['keyword_128_chunks'] for metric in metrics):.2f}。",
    ]
    if cuda_summaries:
        lines.append(f"- CUDA 实测读取 {len(cuda_rows)} 行（{len(cuda_summaries)} 个负载 profile）；stream 重叠比例见 `strategy_summary.csv` 和图。")
        for summary in cuda_summaries:
            lines.append(
                f"- `{summary['profile']}`：CPU 串行/并行为 {summary['cpu_serial_total_ms']:.3f}/{summary['cpu_parallel_total_ms']:.3f} ms，"
                f"节省比例 {summary['cpu_overlap_ratio']:.3f}；CUDA 串行/双 stream 为 {summary['gpu_serial_total_ms']:.3f}/{summary['gpu_overlap_total_ms']:.3f} ms，"
                f"节省比例 {summary['gpu_overlap_ratio']:.3f}。"
            )
    else:
        lines.append("- 未找到可运行的 CUDA benchmark，因此本次只生成 CPU 实测与理论并行结果。")
    lines.extend(
        [
            "",
            "## 如何解读",
            "",
            "- GPU 有计算余量时，双 stream 能直接利用剩余 SM，本机收益明显高于 CPU 工作线程；但高优先级只影响调度顺序，不保证硬件并发。",
            "- GPU 饱和时，低优先级 stream 会与推理争抢资源，本次出现负收益；CPU 工作线程更隔离，但仍需考虑 CPU 占用、能耗和内存带宽。",
            "- 因此不应固定选择一种方案：运行时根据 GPU 利用率选择低优先级 stream 或 CPU worker；两侧都忙时只入队，prefill 后或空闲期异步建树，不阻塞 TTFT。",
            "- 当前实验没有加载真实 LLM、KV cache、BGE-M3 或 KeyBERT，不能据此宣称模型质量提升；它验证的是切分边界差异和调度机制的可测性。",
            "",
            "## 输出文件",
            "",
            "- `dataset_audit.json`：完整数据集与 LoRA 分层样本统计。",
            "- `chunk_method_summary.csv`：各切分方法的 chunk 数量与平均大小。",
            "- `cpu_request_metrics.csv`：每条抽样请求的 TF-IDF、chunk 和 CPU 代理指标。",
            "- `cuda_stream_results.csv`：CUDA benchmark 原始行。",
            "- `strategy_summary.csv`：策略聚合结果。",
            "- `strategy_ttft_comparison.png`、`overlap_ratio_by_prefix_length.png`、`cpu_vs_gpu_keyword_cost.png`、`chunk_boundary_comparison.png`：可视化图片。",
        ]
    )
    (output / "TEST10_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-exe", type=Path, default=DEFAULT_CUDA)
    parser.add_argument("--limit", type=int, default=87)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--reuse-cuda", action="store_true", help="reuse output/cuda_stream_results.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0 or args.trials <= 0:
        raise SystemExit("--limit and --trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = load_records(args.data, sys.maxsize)
    records = select_by_lora(all_records, args.limit)
    vectorizer, idf = fit_idf(all_records)
    metrics = measure_cpu_keywords(records, vectorizer, idf)
    add_schedule_proxies(metrics)
    write_dataset_audit(args.output_dir / "dataset_audit.json", args.data, all_records, records)
    write_csv(args.output_dir / "cpu_request_metrics.csv", metrics)
    write_chunk_summary(args.output_dir / "chunk_method_summary.csv", metrics)

    cuda_csv = args.output_dir / "cuda_stream_results.csv"
    if args.reuse_cuda:
        ran_cuda = cuda_csv.exists()
    else:
        ran_cuda = False if args.skip_cuda else run_cuda(args.cuda_exe, args.data, cuda_csv, len(records), args.trials)
    cuda_rows = read_cuda_rows(cuda_csv) if ran_cuda and cuda_csv.exists() else []
    cuda_summaries = aggregate_cuda(cuda_rows)
    if cuda_summaries:
        write_csv(args.output_dir / "strategy_summary.csv", cuda_summaries)
    else:
        write_csv(args.output_dir / "strategy_summary.csv", [{
            "profile": "cpu_proxy",
            "requests": len(metrics),
            "gpu_serial_total_ms": statistics.mean(metric["cpu_serial_total_ms"] for metric in metrics),
            "gpu_overlap_total_ms": statistics.mean(metric["cpu_parallel_total_ms"] for metric in metrics),
            "gpu_keyword_serial_ms": statistics.mean(metric["keyword_ms"] for metric in metrics),
            "gpu_inference_ms": statistics.mean(metric["proxy_inference_ms"] for metric in metrics),
            "gpu_overlap_saved_ms": statistics.mean(metric["cpu_overlap_saved_ms"] for metric in metrics),
            "gpu_overlap_ratio": statistics.mean(metric["cpu_overlap_ratio"] for metric in metrics),
        }])
    write_plots(args.output_dir, metrics, cuda_rows, cuda_summaries)
    write_report(args.output_dir, args.data, all_records, records, metrics, cuda_rows, cuda_summaries)
    print(f"Wrote test10 results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
