#!/usr/bin/env python3
"""Aggregate test7 runs and generate reproducible plots and a Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "real_100_matrix"

DATASET_LABELS = {
    "msc_continuous": "MSC",
    "taskmaster_continuous": "Taskmaster",
    "lmsys_continuous": "LMSYS-33K",
    "sharegpt_continuous": "ShareGPT52K",
    "opus100_parallel": "OPUS-100",
    "xsum_parallel": "XSum",
}

PREFIX_METHODS = [
    ("fixed_64_128", "Fixed 64/128"),
    ("fixed_192_384", "Fixed 192/384"),
    ("sentence", "Sentence"),
    ("keyword", "Keyword"),
    ("reuse_aware", "Reuse-aware"),
]

# v2 is intentionally kept separate from the original five methods so old
# reports remain reproducible and the optimized experiment can be compared
# against the exact same workload order.
PREFIX_METHODS_V2 = [("adaptive_reuse_v2", "Adaptive reuse v2")]

PREFIX_METHODS_V1_ABLATION = [
    ("reuse_v1_full_signals", "v1 full signals"),
    ("reuse_v1_no_sentence", "v1 no Sentence"),
    ("reuse_v1_no_keyword", "v1 no Keyword (original)"),
    ("reuse_v1_only", "v1 Reuse only"),
]

PREFIX_METHODS_V2_ABLATION = [
    ("adaptive_v2_reference", "v2 reference"),
    ("adaptive_v2_no_sentence", "v2 no Sentence"),
    ("adaptive_v2_no_reuse", "v2 no Reuse"),
    ("adaptive_v2_no_future_cost", "v2 no future-node cost"),
    ("adaptive_v2_no_prefetch_cost", "v2 no prefetch cost"),
    ("adaptive_v2_length_only", "v2 length only"),
]

PREFIX_METHODS_V3 = [
    ("lora_focus_v3", "LoRA-focus filter v3"),
    ("lora_focus_v3_no_private", "v3 no LoRA-private"),
    ("lora_focus_v3_no_focus", "v3 no user-focus"),
    ("lora_focus_v3_cost_only", "v3 cost only"),
]

PREFIX_METHOD_CATALOG = [
    *PREFIX_METHODS,
    *PREFIX_METHODS_V2,
    *PREFIX_METHODS_V1_ABLATION,
    *PREFIX_METHODS_V2_ABLATION,
    *PREFIX_METHODS_V3,
]

STORAGE_STRATEGY_IDS = [
    "full_prefill",
    "legacy_exact",
    "gpu_only_128",
    "gpu_host_full_128",
    "tiered_no_prefetch_128",
    "tiered_oracle_128",
    "tiered_oracle_full",
    "tiered_disk_build_128",
    "tiered_disk_warm_128",
]

PREFIX_RUNTIME_COLUMNS = [
    "runtime_requests",
    "runtime_request_hit_rate",
    "runtime_token_hit_rate",
    "runtime_ttft_speedup_percent",
    "runtime_ttft_ci95_low_percent",
    "runtime_ttft_ci95_high_percent",
    "runtime_ttft_ci95_half_percent",
    "runtime_host_peak_mb",
    "runtime_peak_nodes",
    "runtime_family_evictions",
    "runtime_reusable_prefix_tokens",
    "run_elapsed_seconds",
    "ttft_change_display",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def estimated_tokens(text: str) -> int:
    """Use the same lightweight estimator as build_real_workloads.py."""
    ascii_count = sum(ord(char) < 128 for char in text)
    return max(1, math.ceil(ascii_count / 4 + (len(text) - ascii_count) / 1.5))


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def percentile(values: pd.Series | np.ndarray, value: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, value)) if array.size else 0.0


def bootstrap_mean_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), float(values[0])
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def event_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "family_cache_events.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "cache_events": 0,
            "family_evictions": 0,
            "gpu_to_host_offloads": 0,
            "delta_demotions": 0,
        }
    frame = pd.read_csv(path)
    if frame.empty or "action" not in frame:
        return {
            "cache_events": 0,
            "family_evictions": 0,
            "gpu_to_host_offloads": 0,
            "delta_demotions": 0,
        }
    counts = frame["action"].value_counts()
    return {
        "cache_events": int(len(frame)),
        "family_evictions": int(counts.get("family_evict", 0)),
        "gpu_to_host_offloads": int(counts.get("gpu_full_to_host", 0)),
        "delta_demotions": int(counts.get("tier3_delta_demote", 0)),
    }


def tree_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "online_prefix_tree.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "final_tree_rows": 0,
            "final_gpu_variants": 0,
            "final_host_full_variants": 0,
            "final_host_delta_variants": 0,
            "final_delta_bytes": 0,
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "final_tree_rows": 0,
            "final_gpu_variants": 0,
            "final_host_full_variants": 0,
            "final_host_delta_variants": 0,
            "final_delta_bytes": 0,
        }
    residency = frame.get("variant_residency", pd.Series(dtype=str)).value_counts()
    return {
        "final_tree_rows": int(len(frame)),
        "final_gpu_variants": int(residency.get("gpu_full", 0)),
        "final_host_full_variants": int(residency.get("host_full", 0)),
        "final_host_delta_variants": int(residency.get("host_delta", 0)),
        "final_delta_bytes": float(numeric(frame, "delta_bytes").sum()),
    }


def summarize_run(
    manifest: dict[str, Any],
    strategy: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    run_dir = Path(manifest["run_dir"])
    frame = pd.read_csv(run_dir / "online_request_results.csv")
    baseline = frame[frame["benchmark"] == "baseline"].copy()
    online = frame[frame["benchmark"] == "online"].copy()
    if baseline.empty or online.empty:
        raise ValueError(f"missing baseline or online rows in {run_dir}")
    invalid_modes = online["mode"].astype(str).str.contains("failed", case=False, na=False)
    invalid_ttft = numeric(online, "ttft_ms") <= 0
    if invalid_modes.any() or invalid_ttft.any():
        invalid = online.loc[invalid_modes | invalid_ttft, ["request_id", "mode", "ttft_ms"]]
        raise ValueError(
            f"invalid online rows in {run_dir}: "
            f"{invalid.head(3).to_dict(orient='records')} (count={len(invalid)})"
        )

    baseline_ttft = baseline[["request_id", "ttft_ms"]].rename(
        columns={"ttft_ms": "baseline_ttft_ms"}
    )
    requests = online.merge(baseline_ttft, on="request_id", how="inner")
    requests["dataset"] = manifest["dataset"]
    requests["strategy_id"] = manifest["strategy_id"]
    requests["strategy_label"] = manifest["strategy_label"]
    requests["ttft_saved_ms"] = (
        numeric(requests, "baseline_ttft_ms") - numeric(requests, "ttft_ms")
    )
    requests["ttft_speedup_percent"] = np.where(
        numeric(requests, "baseline_ttft_ms") > 0,
        100.0 * requests["ttft_saved_ms"] / numeric(requests, "baseline_ttft_ms"),
        0.0,
    )

    chunk_hits = numeric(online, "chunk_hit_tokens")
    same_lora_hits = numeric(online, "same_lora_variant_hit") > 0
    legacy_hit_tokens = np.where(
        same_lora_hits,
        numeric(online, "prefix_tokens"),
        0.0,
    )
    reused_tokens = np.maximum(chunk_hits.to_numpy(), legacy_hit_tokens)
    prefix_tokens = numeric(online, "prefix_tokens").to_numpy()
    token_hit_rate = float(reused_tokens.sum() / max(1.0, prefix_tokens.sum()))
    request_hit_rate = float(np.mean(reused_tokens > 0))

    paired_saved = requests["ttft_saved_ms"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(
        paired_saved,
        f"{manifest['dataset']}:{manifest['strategy_id']}",
    )
    baseline_mean = float(numeric(baseline, "ttft_ms").mean())
    online_mean = float(numeric(online, "ttft_ms").mean())
    speedup = 100.0 * (baseline_mean - online_mean) / baseline_mean if baseline_mean else 0.0

    summary: dict[str, Any] = {
        "dataset": manifest["dataset"],
        "dataset_label": DATASET_LABELS.get(manifest["dataset"], manifest["dataset"]),
        "strategy_id": manifest["strategy_id"],
        "strategy_label": manifest["strategy_label"],
        "category": strategy.get("category", "unknown"),
        "chunk_tokens": strategy.get("chunk_tokens", strategy.get("options", {}).get("prefix-chunk-tokens")),
        "requests": int(len(online)),
        "run_elapsed_seconds": float(manifest.get("elapsed_seconds", 0.0)),
        "baseline_ttft_mean_ms": baseline_mean,
        "ttft_mean_ms": online_mean,
        "ttft_p50_ms": percentile(numeric(online, "ttft_ms"), 50),
        "ttft_p95_ms": percentile(numeric(online, "ttft_ms"), 95),
        "total_mean_ms": float(numeric(online, "total_ms").mean()),
        "prefix_mean_ms": float(numeric(online, "prefix_ms").mean()),
        "paired_ttft_saved_mean_ms": float(paired_saved.mean()),
        "paired_ttft_saved_ci95_low_ms": ci_low,
        "paired_ttft_saved_ci95_high_ms": ci_high,
        "ttft_speedup_percent": speedup,
        "prefix_token_hit_rate": token_hit_rate,
        "request_hit_rate": request_hit_rate,
        "same_lora_hit_rate": float(numeric(online, "same_lora_variant_hit").mean()),
        "cross_lora_match_rate": float(numeric(online, "cross_lora_prefix_match").mean()),
        "max_cache_nodes": float(numeric(online, "cache_nodes").max()),
        "max_cache_variants": float(numeric(online, "cache_variants").max()),
        "max_physical_cache_tokens": float(numeric(online, "physical_cache_tokens").max()),
        "max_host_delta_bytes": float(numeric(online, "host_delta_bytes").max()),
        "max_host_full_bytes": float(numeric(online, "host_full_bytes").max()),
        "max_host_delta_variants": float(numeric(online, "host_delta_variants").max()),
        "max_host_full_variants": float(numeric(online, "host_full_variants").max()),
        "delta_store_loaded": float(numeric(online, "delta_store_loaded").sum()),
        "delta_store_saved": float(numeric(online, "delta_store_saved").sum()),
        "delta_store_load_ms": float(numeric(online, "delta_store_load_ms").sum()),
        "delta_store_bytes_read": float(numeric(online, "delta_store_bytes_read").sum()),
        "delta_compressed_background": float(numeric(online, "delta_compressed_background").sum()),
        "delta_build_ok": float(numeric(online, "delta_build_ok").sum()),
        "materialize_ok": float(numeric(online, "materialize_ok").sum()),
        "reconstruction_checks": float(numeric(online, "reconstruction_checks").sum()),
        "reconstruction_cos_min": float(
            numeric(online, "reconstruction_cos_min").replace(0, np.nan).min()
            if (numeric(online, "reconstruction_cos_min").replace(0, np.nan).notna().any())
            else 0.0
        ),
        "reconstruction_l2_max": float(numeric(online, "reconstruction_l2_max").max()),
        "background_overrun_ms": float(numeric(online, "background_overrun_ms").sum()),
    }
    summary.update(event_metrics(run_dir))
    summary.update(tree_metrics(run_dir))

    baseline_requests = baseline.copy()
    baseline_requests["dataset"] = manifest["dataset"]
    baseline_requests["strategy_id"] = "full_prefill"
    baseline_requests["strategy_label"] = "Full prefill"
    return summary, requests, baseline_requests


def baseline_summary(
    dataset: str,
    baseline_requests: pd.DataFrame,
) -> dict[str, Any]:
    ttft = numeric(baseline_requests, "ttft_ms")
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "strategy_id": "full_prefill",
        "strategy_label": "Full prefill",
        "category": "baseline",
        "chunk_tokens": np.nan,
        "requests": int(len(baseline_requests)),
        "run_elapsed_seconds": float("nan"),
        "baseline_ttft_mean_ms": float(ttft.mean()),
        "ttft_mean_ms": float(ttft.mean()),
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "total_mean_ms": float(numeric(baseline_requests, "total_ms").mean()),
        "prefix_mean_ms": 0.0,
        "paired_ttft_saved_mean_ms": 0.0,
        "paired_ttft_saved_ci95_low_ms": 0.0,
        "paired_ttft_saved_ci95_high_ms": 0.0,
        "ttft_speedup_percent": 0.0,
        "prefix_token_hit_rate": 0.0,
        "request_hit_rate": 0.0,
        "same_lora_hit_rate": 0.0,
        "cross_lora_match_rate": 0.0,
        "max_cache_nodes": 0.0,
        "max_cache_variants": 0.0,
        "max_physical_cache_tokens": 0.0,
        "max_host_delta_bytes": 0.0,
        "max_host_full_bytes": 0.0,
        "max_host_delta_variants": 0.0,
        "max_host_full_variants": 0.0,
        "delta_store_loaded": 0.0,
        "delta_store_saved": 0.0,
        "delta_store_load_ms": 0.0,
        "delta_store_bytes_read": 0.0,
        "delta_compressed_background": 0.0,
        "delta_build_ok": 0.0,
        "materialize_ok": 0.0,
        "reconstruction_checks": 0.0,
        "reconstruction_cos_min": 0.0,
        "reconstruction_l2_max": 0.0,
        "background_overrun_ms": 0.0,
        "cache_events": 0,
        "family_evictions": 0,
        "gpu_to_host_offloads": 0,
        "delta_demotions": 0,
        "final_tree_rows": 0,
        "final_gpu_variants": 0,
        "final_host_full_variants": 0,
        "final_host_delta_variants": 0,
        "final_delta_bytes": 0.0,
    }


def collect_delta_quality(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_datasets: set[str] = set()
    for manifest in manifests:
        dataset = str(manifest["dataset"])
        if dataset in seen_datasets:
            continue
        path = Path(manifest["run_dir"]) / "delta_prefix_probe.csv"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["dataset"] = dataset
        frame["dataset_label"] = DATASET_LABELS.get(dataset, dataset)
        frames.append(frame)
        seen_datasets.add(dataset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_at_positions(text: str, positions: list[int]) -> list[str]:
    boundaries = [0, *sorted({item for item in positions if 0 < item < len(text)}), len(text)]
    return [text[start:end] for start, end in zip(boundaries, boundaries[1:]) if end > start]


def fixed_chunks(text: str, target_tokens: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        remaining_tokens = estimated_tokens(remaining)
        if remaining_tokens <= target_tokens:
            chunks.append(remaining)
            break
        estimate = max(1, int(round(len(remaining) * target_tokens / remaining_tokens)))
        end = min(len(text), start + estimate)
        while end > start + 1 and estimated_tokens(text[start:end]) > target_tokens:
            end -= 1
        while end < len(text) and estimated_tokens(text[start : end + 1]) <= target_tokens:
            end += 1
        chunks.append(text[start:end])
        start = end
    return chunks


def segment_boundaries(text: str, segments: list[dict[str, Any]]) -> set[int]:
    boundaries: set[int] = set()
    offset = 0
    for segment in segments:
        offset += len(str(segment.get("text", "")))
        if 0 < offset <= len(text):
            boundaries.add(offset)
    boundaries.add(len(text))
    return boundaries


def sentence_boundaries(text: str) -> set[int]:
    boundaries: set[int] = set()
    for match in re.finditer(r"(?:[.!?。！？][\"')\]】]*\s*|\n+)", text):
        boundaries.add(match.end())
    boundaries.add(len(text))
    return boundaries


def keyword_boundaries(text: str) -> set[int]:
    pattern = re.compile(
        r"(?im)(?:^|\n)\s*(?:user|assistant|system|task|context|article|question|answer|"
        r"session|profile|instruction|dialogue|summary|headline|translate|source|target)\b|"
        r"\b(?:however|meanwhile|therefore|because|finally|next|then)\b"
    )
    boundaries = {match.start() for match in pattern.finditer(text) if match.start() > 0}
    boundaries.add(len(text))
    return boundaries


def natural_boundaries(text: str, segments: list[dict[str, Any]]) -> set[int]:
    return segment_boundaries(text, segments) | sentence_boundaries(text)


def pack_units(units: list[str], target_tokens: int = 128) -> list[str]:
    packed: list[str] = []
    current = ""
    for unit in units:
        if not unit:
            continue
        if estimated_tokens(unit) > int(target_tokens * 1.5):
            if current:
                packed.append(current)
                current = ""
            packed.extend(fixed_chunks(unit, target_tokens))
            continue
        candidate = current + unit
        if current and estimated_tokens(candidate) > target_tokens:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def split_prefix(
    row: dict[str, Any],
    method_id: str,
    reuse_points: set[int],
) -> list[str]:
    text = str(row.get("common_prefix_text", ""))
    segments = list(row.get("prefix_segments") or [])
    if method_id in {"fixed_64_128", "fixed_192_384"}:
        system_target, context_target = (
            (64, 128) if method_id == "fixed_64_128" else (192, 384)
        )
        chunks: list[str] = []
        if not segments:
            return fixed_chunks(text, context_target)
        for segment in segments:
            segment_text = str(segment.get("text", ""))
            segment_type = str(segment.get("type", ""))
            target = system_target if segment_type == "shared_system" else context_target
            chunks.extend(fixed_chunks(segment_text, target))
        return chunks

    boundaries = natural_boundaries(text, segments)
    if method_id == "sentence":
        return pack_units(split_at_positions(text, list(boundaries)), 128)

    if method_id == "keyword":
        units = split_at_positions(text, list(boundaries))
        keyword_pattern = re.compile(
            r"(?i)(?:^|\n)\s*(?:user|assistant|system|task|context|article|question|answer|"
            r"session|profile|instruction|dialogue|summary|headline|translate|source|target)\b|"
            r"\b(?:however|meanwhile|therefore|because|finally|next|then)\b"
        )
        chunks: list[str] = []
        current = ""
        for unit in units:
            if estimated_tokens(unit) > 288:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(fixed_chunks(unit, 192))
                continue
            keyword_start = bool(keyword_pattern.search(unit))
            candidate = current + unit
            if current and (keyword_start or estimated_tokens(candidate) > 192):
                chunks.append(current)
                current = unit
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    if method_id != "reuse_aware":
        raise ValueError(f"unknown prefix method: {method_id}")

    candidates = sorted(boundaries | reuse_points)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        if estimated_tokens(remaining) <= 128:
            chunks.append(remaining)
            break
        viable: list[tuple[float, int]] = []
        for end in candidates:
            if end <= start or end >= len(text):
                continue
            size = estimated_tokens(text[start:end])
            if 64 <= size <= 192:
                reuse_bonus = 72.0 if end in reuse_points else 0.0
                viable.append((abs(size - 128) - reuse_bonus, end))
        if viable:
            end = min(viable)[1]
        else:
            fallback = fixed_chunks(remaining, 128)[0]
            end = start + len(fallback)
        chunks.append(text[start:end])
        start = end
    return chunks


def split_prefix_adaptive_v2(
    row: dict[str, Any],
    reuse_points: set[int],
    context_reaccess_rate: float = 0.0,
    exact_prefix_rate: float = 0.0,
    max_prefetch_chunks: int = 2,
    use_sentence_boundaries: bool = True,
    use_reuse_reward: bool = True,
    use_future_cost: bool = True,
    use_prefetch_cost: bool = True,
    use_natural_reward: bool = True,
) -> list[str]:
    """Adaptive v2 splitter with explicit reuse/cost scoring.

    The target is selected from workload shape: exact-prefix parallel pairs
    keep the reusable source inside the prefetch budget, while continuous
    traces use larger chunks when reaccess is sparse and align append-only
    history to known reuse points when they exist.
    """
    text = str(row.get("common_prefix_text", ""))
    segments = list(row.get("prefix_segments") or [])
    total_tokens = estimated_tokens(text)
    parallel = exact_prefix_rate >= 0.25 or bool(row.get("exact_shared_english_intersection"))
    if parallel:
        target = max(96, min(512, math.ceil(total_tokens / max(1, max_prefetch_chunks))))
        min_tokens = max(48, int(target * 0.55))
        max_tokens = max(target, int(target * 1.35))
    else:
        # Sparse reaccess benefits from fewer, larger nodes. Frequent return
        # visits can afford smaller nodes to maximize partial-prefix reuse.
        target = 192 if context_reaccess_rate < 0.70 else 128
        if total_tokens > 768:
            target = 256
        elif total_tokens > 512:
            target = max(target, 192)
        min_tokens = 96 if target == 192 else 64
        if target == 256:
            min_tokens, max_tokens = 128, 384
        else:
            max_tokens = 288 if target == 192 else 192

    boundaries = segment_boundaries(text, segments)
    if use_sentence_boundaries:
        boundaries |= sentence_boundaries(text)
    active_reuse_points = reuse_points if use_reuse_reward else set()
    candidates = sorted(boundaries | active_reuse_points)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        if estimated_tokens(remaining) <= max_tokens:
            chunks.append(remaining)
            break
        viable: list[tuple[float, int]] = []
        for end in candidates:
            if end <= start or end >= len(text):
                continue
            size = estimated_tokens(text[start:end])
            if not min_tokens <= size <= max_tokens:
                continue
            is_reuse = end in active_reuse_points
            is_natural = end in boundaries
            remaining_after = estimated_tokens(text[end:]) if end < len(text) else 0
            future_nodes = math.ceil(remaining_after / max(1, target))
            reuse_weight = 48.0 + 64.0 * context_reaccess_rate + 64.0 * exact_prefix_rate
            prefetch_overflow = 0
            if parallel:
                slots_left = max(0, max_prefetch_chunks - (len(chunks) + 1))
                prefetch_overflow = max(0, future_nodes - slots_left)
            # Lower is better. All terms use token-equivalent weights so they
            # can be inspected and ablated directly.
            score = abs(size - target)
            score += 10.0 * future_nodes if use_future_cost else 0.0
            score += 80.0 * prefetch_overflow if use_prefetch_cost else 0.0
            score -= (reuse_weight if is_reuse else 0.0)
            score -= (18.0 if is_natural and use_natural_reward else 0.0)
            viable.append((score, end))
        if viable:
            end = min(viable)[1]
        else:
            fallback = fixed_chunks(remaining, target)[0]
            end = start + len(fallback)
        chunks.append(text[start:end])
        start = end
    return chunks


_FOCUS_STOPWORDS = {
    "about", "after", "again", "also", "answer", "assistant", "before", "being",
    "continue", "could", "from", "have", "into", "latest", "message", "more",
    "please", "should", "task", "that", "their", "there", "these", "they", "this",
    "using", "what", "when", "where", "which", "while", "with", "would", "your",
}


def focus_terms(text: str) -> set[str]:
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
    }
    return {term for term in terms if term not in _FOCUS_STOPWORDS}


def build_lora_focus_profile(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Build deterministic filter statistics from workload prompts only."""
    term_total: dict[str, int] = {}
    term_loras: dict[str, set[int]] = {}
    term_contexts: dict[str, set[str]] = {}
    term_by_lora: dict[str, dict[int, int]] = {}
    term_by_context: dict[str, dict[str, int]] = {}
    requests_by_context: dict[str, int] = {}
    loras: set[int] = set()
    for request in requests:
        lora_id = int(request.get("lora_id", -1))
        context_id = str(request.get("context_id", ""))
        loras.add(lora_id)
        requests_by_context[context_id] = requests_by_context.get(context_id, 0) + 1
        for term in focus_terms(str(request.get("common_prefix_text", ""))):
            term_total[term] = term_total.get(term, 0) + 1
            term_loras.setdefault(term, set()).add(lora_id)
            term_contexts.setdefault(term, set()).add(context_id)
            by_lora = term_by_lora.setdefault(term, {})
            by_lora[lora_id] = by_lora.get(lora_id, 0) + 1
            by_context = term_by_context.setdefault(term, {})
            by_context[context_id] = by_context.get(context_id, 0) + 1
    return {
        "request_count": len(requests),
        "lora_count": len(loras),
        "term_total": term_total,
        "term_loras": term_loras,
        "term_contexts": term_contexts,
        "term_by_lora": term_by_lora,
        "term_by_context": term_by_context,
        "requests_by_context": requests_by_context,
    }


def build_causal_lora_focus_profiles(
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one profile per request without observing future requests."""
    return [build_lora_focus_profile(requests[: index + 1]) for index in range(len(requests))]


def lora_focus_signals(
    text: str,
    row: dict[str, Any],
    profile: dict[str, Any],
    end_ratio: float,
) -> tuple[float, float, float]:
    terms = focus_terms(text)
    if not terms:
        return 0.0, 0.0, min(1.0, 0.30 * end_ratio)
    lora_id = int(row.get("lora_id", -1))
    context_id = str(row.get("context_id", ""))
    total_loras = max(1, int(profile.get("lora_count", 1)))
    context_requests = max(1, int(profile.get("requests_by_context", {}).get(context_id, 1)))
    shared_values: list[float] = []
    private_values: list[float] = []
    context_values: list[float] = []
    for term in terms:
        total = int(profile.get("term_total", {}).get(term, 0))
        support = min(1.0, max(0, total - 1) / 3.0)
        lora_count = len(profile.get("term_loras", {}).get(term, set()))
        cross_lora = (lora_count - 1) / max(1, total_loras - 1)
        current_lora_count = int(profile.get("term_by_lora", {}).get(term, {}).get(lora_id, 0))
        concentration = current_lora_count / max(1, total)
        shared_values.append(support * cross_lora)
        private_values.append(support * concentration * (1.0 - cross_lora))
        context_count = int(
            profile.get("term_by_context", {}).get(term, {}).get(context_id, 0)
        )
        context_values.append(min(1.0, max(0, context_count - 1) / max(1, context_requests - 1)))
    shared = float(sum(shared_values) / len(shared_values))
    private = float(sum(private_values) / len(private_values))
    context_reuse = float(sum(context_values) / len(context_values))
    task_terms = focus_terms(str(row.get("task", "")))
    task_overlap = len(terms & task_terms) / max(1, len(task_terms))
    role_marker = float(bool(re.search(r"(?im)(?:^|\n)\s*(?:user|question|profile|state)\s*:", text)))
    focus = min(1.0, 0.50 * task_overlap + 0.30 * end_ratio + 0.20 * role_marker)
    reuse = min(1.0, 0.55 * shared + 0.45 * context_reuse)
    return reuse, private, focus


def split_prefix_lora_focus_v3(
    row: dict[str, Any],
    reuse_points: set[int],
    profile: dict[str, Any],
    context_reaccess_rate: float = 0.0,
    exact_prefix_rate: float = 0.0,
    max_prefetch_chunks: int = 2,
    use_private_signal: bool = True,
    use_focus_signal: bool = True,
    use_reuse_signal: bool = True,
) -> tuple[list[str], list[str]]:
    """Filter-style LoRA affinity and user-focus aware splitter.

    The returned roles are descriptive admission hints. The current runtime
    still receives ordinary prefix_method segments and does not pin them.
    """
    text = str(row.get("common_prefix_text", ""))
    segments = list(row.get("prefix_segments") or [])
    total_tokens = estimated_tokens(text)
    parallel = exact_prefix_rate >= 0.25 or bool(row.get("exact_shared_english_intersection"))
    base_target = (
        max(96, min(512, math.ceil(total_tokens / max(1, max_prefetch_chunks))))
        if parallel
        else (128 if context_reaccess_rate >= 0.70 else 192)
    )
    if not parallel and total_tokens > 512:
        base_target = max(base_target, 192)
    if not parallel and total_tokens > 768:
        base_target = 256
    boundaries = natural_boundaries(text, segments) | keyword_boundaries(text)
    candidates = sorted(boundaries | reuse_points)
    chunks: list[str] = []
    roles: list[str] = []
    start = 0
    while start < len(text):
        remaining_tokens = estimated_tokens(text[start:])
        if remaining_tokens <= base_target:
            chunk = text[start:]
            reuse, private, focus = lora_focus_signals(chunk, row, profile, 1.0)
            if not use_reuse_signal:
                reuse = 0.0
            if not use_private_signal:
                private = 0.0
            if not use_focus_signal:
                focus = 0.0
            role = "shared_reuse" if reuse >= max(private, focus, 0.35) else (
                "lora_private_resident" if private >= max(focus, 0.35) else (
                    "user_focus" if focus >= 0.35 else "cold_bulk"
                )
            )
            chunks.append(chunk)
            roles.append(role)
            break
        viable: list[tuple[float, int, str]] = []
        for end in candidates:
            if end <= start or end >= len(text):
                continue
            size = estimated_tokens(text[start:end])
            end_ratio = end / max(1, len(text))
            reuse, private, focus = lora_focus_signals(
                text[start:end], row, profile, end_ratio
            )
            if not use_reuse_signal:
                reuse = 0.0
            if not use_private_signal:
                private = 0.0
            if not use_focus_signal:
                focus = 0.0
            signal_discount = min(160.0, 96.0 * reuse + 80.0 * focus + 64.0 * private)
            signal_target = int(round(256.0 - signal_discount))
            # Salience should select a boundary, not force long contexts into
            # many small nodes. Parallel prefixes are bounded by prefetch slots;
            # long continuous prefixes retain the cost-derived base target.
            if parallel or total_tokens > 512:
                target = base_target
            else:
                target = signal_target
            target = max(96, min(512, target))
            if not max(48, int(target * 0.50)) <= size <= max(target, int(target * 1.50)):
                continue
            remaining_after = estimated_tokens(text[end:])
            # Do not create a boundary that strands a tiny, high-overhead tail.
            if 0 < remaining_after < 48:
                continue
            future_nodes = math.ceil(remaining_after / max(1, target))
            slots_left = max(0, max_prefetch_chunks - (len(chunks) + 1))
            prefetch_overflow = max(0, future_nodes - slots_left) if parallel else 0
            mixing = min(private, max(reuse, focus))
            score = abs(size - target)
            score += 10.0 * future_nodes + 80.0 * prefetch_overflow
            score += 32.0 * mixing
            score -= 64.0 * reuse + 48.0 * private + 56.0 * focus
            score -= 48.0 if end in reuse_points else 0.0
            score -= 18.0 if end in boundaries else 0.0
            role = "shared_reuse" if reuse >= max(private, focus, 0.35) else (
                "lora_private_resident" if private >= max(focus, 0.35) else (
                    "user_focus" if focus >= 0.35 else "cold_bulk"
                )
            )
            viable.append((score, end, role))
        if viable:
            _, end, role = min(viable)
        else:
            fallback = fixed_chunks(text[start:], base_target)[0]
            end = start + len(fallback)
            if 0 < estimated_tokens(text[end:]) < 48:
                end = len(text)
            role = "cold_bulk"
        chunks.append(text[start:end])
        roles.append(role)
        start = end
    return chunks, roles


def split_prefix_experiment_method(
    row: dict[str, Any],
    method_id: str,
    reuse_points: set[int],
    context_reaccess_rate: float = 0.0,
    exact_prefix_rate: float = 0.0,
    lora_focus_profile: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    if method_id in dict(PREFIX_METHODS):
        chunks = split_prefix(row, method_id, reuse_points)
        return chunks, ["legacy"] * len(chunks)
    if method_id == "adaptive_reuse_v2":
        chunks = split_prefix_adaptive_v2(
            row, reuse_points, context_reaccess_rate, exact_prefix_rate
        )
        return chunks, ["adaptive"] * len(chunks)
    if method_id in dict(PREFIX_METHODS_V1_ABLATION):
        text = str(row.get("common_prefix_text", ""))
        segments = list(row.get("prefix_segments") or [])
        candidates = set(reuse_points) | {len(text)}
        if method_id in {"reuse_v1_full_signals", "reuse_v1_no_keyword"}:
            candidates |= natural_boundaries(text, segments)
        if method_id in {"reuse_v1_full_signals", "reuse_v1_no_sentence"}:
            candidates |= segment_boundaries(text, segments) | keyword_boundaries(text)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            remaining = text[start:]
            if estimated_tokens(remaining) <= 128:
                chunks.append(remaining)
                break
            viable = []
            for end in sorted(candidates):
                if end <= start or end >= len(text):
                    continue
                size = estimated_tokens(text[start:end])
                if 64 <= size <= 192:
                    viable.append((abs(size - 128) - (72.0 if end in reuse_points else 0.0), end))
            if viable:
                end = min(viable)[1]
            else:
                fallback = fixed_chunks(remaining, 128)[0]
                end = start + len(fallback)
            chunks.append(text[start:end])
            start = end
        return chunks, ["v1_ablation"] * len(chunks)
    if method_id in dict(PREFIX_METHODS_V2_ABLATION):
        options = {
            "adaptive_v2_reference": {},
            "adaptive_v2_no_sentence": {"use_sentence_boundaries": False},
            "adaptive_v2_no_reuse": {"use_reuse_reward": False},
            "adaptive_v2_no_future_cost": {"use_future_cost": False},
            "adaptive_v2_no_prefetch_cost": {"use_prefetch_cost": False},
            "adaptive_v2_length_only": {
                "use_reuse_reward": False,
                "use_future_cost": False,
                "use_prefetch_cost": False,
                "use_natural_reward": False,
            },
        }[method_id]
        chunks = split_prefix_adaptive_v2(
            row,
            reuse_points,
            context_reaccess_rate,
            exact_prefix_rate,
            **options,
        )
        return chunks, ["v2_ablation"] * len(chunks)
    if method_id in dict(PREFIX_METHODS_V3):
        if lora_focus_profile is None:
            raise ValueError("LoRA-focus v3 requires a workload profile")
        options = {
            "lora_focus_v3": {},
            "lora_focus_v3_no_private": {"use_private_signal": False},
            "lora_focus_v3_no_focus": {"use_focus_signal": False},
            "lora_focus_v3_cost_only": {
                "use_private_signal": False,
                "use_focus_signal": False,
                "use_reuse_signal": False,
            },
        }[method_id]
        return split_prefix_lora_focus_v3(
            row,
            reuse_points,
            lora_focus_profile,
            context_reaccess_rate=context_reaccess_rate,
            exact_prefix_rate=exact_prefix_rate,
            **options,
        )
    raise ValueError(f"unknown prefix experiment method: {method_id}")


def analyze_prefix_structure(snapshot: dict[str, Any]) -> pd.DataFrame:
    workloads_root = Path(snapshot["workloads_root"])
    rows_out: list[dict[str, Any]] = []
    request_limit = int(snapshot.get("request_limit", 100))
    for dataset in snapshot["selected_datasets"]:
        path = workloads_root / dataset / "grouped" / "grouped_requests.jsonl"
        if not path.is_file():
            continue
        requests = read_jsonl(path)[:request_limit]
        prefixes_by_context: dict[str, list[str]] = {}
        for request in requests:
            context_id = str(request.get("context_id", ""))
            prefixes_by_context.setdefault(context_id, []).append(
                str(request.get("common_prefix_text", ""))
            )

        for method_id, method_label in PREFIX_METHODS:
            seen_nodes: set[str] = set()
            seen_prefixes: set[str] = set()
            total_chunks = 0
            total_chunk_tokens = 0
            total_prefix_tokens = 0
            reused_tokens = 0
            requests_with_reuse = 0
            complete_prefix_hits = 0
            natural_ends = 0
            internal_ends = 0
            for request in requests:
                text = str(request.get("common_prefix_text", ""))
                context_id = str(request.get("context_id", ""))
                reuse_points = {
                    len(prefix)
                    for prefix in prefixes_by_context.get(context_id, [])
                    if len(prefix) < len(text) and text.startswith(prefix)
                }
                chunks = split_prefix(request, method_id, reuse_points)
                chunk_tokens = [estimated_tokens(chunk) for chunk in chunks]
                total_chunks += len(chunks)
                total_chunk_tokens += sum(chunk_tokens)
                total_prefix_tokens += estimated_tokens(text)

                boundaries = natural_boundaries(
                    text,
                    list(request.get("prefix_segments") or []),
                )
                offset = 0
                for chunk in chunks[:-1]:
                    offset += len(chunk)
                    internal_ends += 1
                    natural_ends += int(offset in boundaries)

                parent = "root"
                node_ids: list[str] = []
                for chunk in chunks:
                    node_id = hashlib.sha256(
                        (parent + "\0" + chunk).encode("utf-8")
                    ).hexdigest()[:24]
                    node_ids.append(node_id)
                    parent = node_id
                request_reused = sum(
                    tokens for node_id, tokens in zip(node_ids, chunk_tokens) if node_id in seen_nodes
                )
                reused_tokens += request_reused
                requests_with_reuse += int(request_reused > 0)
                prefix_hash = str(request.get("common_prefix_hash", ""))
                complete_prefix_hits += int(prefix_hash in seen_prefixes)
                seen_nodes.update(node_ids)
                seen_prefixes.add(prefix_hash)

            rows_out.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "method_id": method_id,
                    "method_label": method_label,
                    "requests": len(requests),
                    "mean_prefix_tokens": total_prefix_tokens / max(1, len(requests)),
                    "mean_chunks_per_prefix": total_chunks / max(1, len(requests)),
                    "mean_tokens_per_chunk": total_chunk_tokens / max(1, total_chunks),
                    "structural_request_hit_rate": requests_with_reuse / max(1, len(requests)),
                    "structural_token_coverage": reused_tokens / max(1, total_chunk_tokens),
                    "complete_prefix_repeat_rate": complete_prefix_hits / max(1, len(requests)),
                    "natural_boundary_coverage": natural_ends / max(1, internal_ends),
                    "unique_nodes": len(seen_nodes),
                }
            )
    return pd.DataFrame(rows_out)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_speedup_heatmap(summary: pd.DataFrame, figures_dir: Path) -> None:
    dataset_order = [dataset for dataset in DATASET_LABELS if dataset in set(summary["dataset"])]
    strategy_order = [
        strategy
        for strategy in summary["strategy_id"].drop_duplicates()
        if strategy != "full_prefill"
    ]
    pivot = summary.pivot(index="dataset", columns="strategy_id", values="ttft_speedup_percent")
    values = pivot.reindex(index=dataset_order, columns=strategy_order).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(5.0, float(np.max(np.abs(finite))) if finite.size else 5.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = plt.subplots(figsize=(14, 4.6), constrained_layout=True)
    image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_title("TTFT speedup relative to the matched full-prefill run")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Dataset")
    labels = (
        summary.drop_duplicates("strategy_id")
        .set_index("strategy_id")["strategy_label"]
        .to_dict()
    )
    ax.set_xticks(range(len(strategy_order)), [labels.get(item, item) for item in strategy_order], rotation=35, ha="right")
    ax.set_yticks(range(len(dataset_order)), [DATASET_LABELS[item] for item in dataset_order])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("TTFT speedup (%)")
    fig.savefig(figures_dir / "ttft_speedup_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_chunk_ablation(summary: pd.DataFrame, figures_dir: Path) -> None:
    chunk = summary[summary["category"] == "chunk"].copy()
    if chunk.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    metrics = [
        ("ttft_speedup_percent", "TTFT speedup (%)"),
        ("prefix_token_hit_rate", "Prefix-token hit rate"),
        ("ttft_p95_ms", "TTFT p95 (ms)"),
        ("max_physical_cache_tokens", "Peak physical cache tokens"),
    ]
    colors = plt.get_cmap("tab10").colors
    for dataset_index, (dataset, group) in enumerate(chunk.groupby("dataset", sort=False)):
        group = group.sort_values("chunk_tokens")
        for ax, (metric, label) in zip(axes.flat, metrics):
            values = group[metric].to_numpy(dtype=float)
            if metric == "prefix_token_hit_rate":
                values *= 100.0
                label = "Prefix-token hit rate (%)"
            ax.plot(
                group["chunk_tokens"],
                values,
                marker="o",
                linewidth=1.7,
                label=DATASET_LABELS.get(dataset, dataset),
                color=colors[dataset_index % len(colors)],
            )
            ax.set_xlabel("Fixed chunk size (tokens)")
            ax.set_ylabel(label)
            ax.set_xticks(sorted(chunk["chunk_tokens"].dropna().unique()))
            ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_title("Latency")
    axes[0, 1].set_title("Reuse")
    axes[1, 0].set_title("Tail latency")
    axes[1, 1].set_title("GPU cache pressure")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=min(5, len(labels)), frameon=False)
    fig.savefig(figures_dir / "prefix_chunk_ablation.png", bbox_inches="tight")
    plt.close(fig)


def save_storage_ablation(summary: pd.DataFrame, figures_dir: Path) -> None:
    storage_ids = STORAGE_STRATEGY_IDS
    storage = summary[summary["strategy_id"].isin(storage_ids)].copy()
    if storage.empty:
        return
    order = [item for item in storage_ids if item in set(storage["strategy_id"])]
    labels = (
        storage.drop_duplicates("strategy_id")
        .set_index("strategy_id")["strategy_label"]
        .to_dict()
    )
    datasets = [item for item in DATASET_LABELS if item in set(storage["dataset"])]
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.16
    for index, dataset in enumerate(datasets):
        group = storage[storage["dataset"] == dataset].set_index("strategy_id")
        speedup = [float(group.loc[item, "ttft_speedup_percent"]) if item in group.index else np.nan for item in order]
        hit_rate = [100.0 * float(group.loc[item, "prefix_token_hit_rate"]) if item in group.index else np.nan for item in order]
        offset = (index - (len(datasets) - 1) / 2) * width
        axes[0].bar(x + offset, speedup, width, label=DATASET_LABELS[dataset], color=colors[index])
        axes[1].bar(x + offset, hit_rate, width, label=DATASET_LABELS[dataset], color=colors[index])
    axes[0].axhline(0, color="0.35", linewidth=0.8)
    axes[0].set_ylabel("TTFT speedup (%)")
    axes[0].set_title("Storage-policy latency ablation")
    axes[1].set_ylabel("Prefix-token hit rate (%)")
    axes[1].set_title("Storage-policy reuse ablation")
    for ax in axes:
        ax.set_xticks(x, [labels.get(item, item) for item in order], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=min(5, len(datasets)), frameon=False, loc="best")
    fig.savefig(figures_dir / "storage_ablation.png", bbox_inches="tight")
    plt.close(fig)


def save_resource_tradeoff(summary: pd.DataFrame, figures_dir: Path) -> None:
    cache = summary[summary["strategy_id"] != "full_prefill"].copy()
    if cache.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, (dataset, group) in enumerate(cache.groupby("dataset", sort=False)):
        host_mb = (group["max_host_delta_bytes"] + group["max_host_full_bytes"]) / (1024.0 * 1024.0)
        sizes = 30.0 + np.clip(group["prefix_token_hit_rate"].to_numpy(dtype=float), 0, 1) * 130.0
        ax.scatter(
            group["max_physical_cache_tokens"],
            host_mb,
            s=sizes,
            alpha=0.78,
            label=DATASET_LABELS.get(dataset, dataset),
            color=colors[index % len(colors)],
            edgecolor="0.25",
            linewidth=0.4,
        )
    ax.set_title("Cache resource trade-off (marker size = prefix hit rate)")
    ax.set_xlabel("Peak physical GPU cache tokens")
    ax.set_ylabel("Peak host cache (MiB)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "storage_resource_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def save_delta_quality(delta: pd.DataFrame, figures_dir: Path) -> None:
    if delta.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, (dataset, group) in enumerate(delta.groupby("dataset", sort=False)):
        axes[0].scatter(
            numeric(group, "prefix_tokens"),
            100.0 * numeric(group, "logical_saved_rate"),
            label=DATASET_LABELS.get(dataset, dataset),
            color=colors[index],
            alpha=0.8,
        )
        axes[1].scatter(
            numeric(group, "kv_l2"),
            numeric(group, "kv_cos"),
            label=DATASET_LABELS.get(dataset, dataset),
            color=colors[index],
            alpha=0.8,
        )
    axes[0].set_title("Q8 KV-delta compression")
    axes[0].set_xlabel("Shared prefix tokens")
    axes[0].set_ylabel("Logical bytes saved (%)")
    axes[1].set_title("Cross-LoRA KV similarity")
    axes[1].set_xlabel("KV normalized L2")
    axes[1].set_ylabel("KV cosine similarity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(figures_dir / "delta_quality.png", bbox_inches="tight")
    plt.close(fig)


def save_dataset_final_effect(dataset_summary: pd.DataFrame, output_path: Path) -> None:
    preferred_order = [
        "full_prefill",
        "legacy_exact",
        "gpu_only_128",
        "gpu_host_full_128",
        "tiered_no_prefetch_128",
        "tiered_oracle_32",
        "tiered_oracle_64",
        "tiered_oracle_128",
        "tiered_oracle_256",
        "tiered_oracle_full",
        "tiered_disk_build_128",
        "tiered_disk_warm_128",
    ]
    order_map = {item: index for index, item in enumerate(preferred_order)}
    frame = dataset_summary.copy()
    frame["plot_order"] = frame["strategy_id"].map(order_map).fillna(len(order_map))
    frame = frame.sort_values("plot_order")
    short_labels = {
        "full_prefill": "Full prefill",
        "legacy_exact": "Legacy exact",
        "gpu_only_128": "GPU only",
        "gpu_host_full_128": "GPU + host full",
        "tiered_no_prefetch_128": "Tiered no-prefetch",
        "tiered_oracle_32": "Oracle 32 / 2 chunks",
        "tiered_oracle_64": "Oracle 64 / 2 chunks",
        "tiered_oracle_128": "Oracle 128 / 2 chunks",
        "tiered_oracle_256": "Oracle 256 / 2 chunks",
        "tiered_oracle_full": "Oracle 128 / full coverage",
        "tiered_disk_build_128": "Disk cold-build",
        "tiered_disk_warm_128": "Disk warm-load",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    label = str(frame["dataset_label"].iloc[0])
    fig.suptitle(f"{label}: prefix reuse and three-level storage", fontsize=14)

    speed = frame["ttft_speedup_percent"].to_numpy(dtype=float)
    colors = ["#3f8f62" if value > 0 else "#b95c5c" if value < 0 else "#858585" for value in speed]
    y = np.arange(len(frame))
    axes[0, 0].barh(y, speed, color=colors)
    axes[0, 0].set_yticks(y, [short_labels.get(item, item) for item in frame["strategy_id"]])
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(0, color="0.35", linewidth=0.8)
    axes[0, 0].set_xlabel("TTFT speedup vs matched full prefill (%)")
    axes[0, 0].set_title("End-to-end latency")
    axes[0, 0].grid(axis="x", alpha=0.25)
    for index, value in enumerate(speed):
        axes[0, 0].text(value, index, f" {value:.1f}%", va="center", fontsize=8)

    oracle_ids = [
        "tiered_oracle_32",
        "tiered_oracle_64",
        "tiered_oracle_128",
        "tiered_oracle_256",
        "tiered_oracle_full",
    ]
    oracle = frame[frame["strategy_id"].isin(oracle_ids)].set_index("strategy_id").reindex(oracle_ids)
    oracle = oracle.dropna(subset=["ttft_mean_ms"])
    oracle_labels = ["32\n2 chunks", "64\n2 chunks", "128\n2 chunks", "256\n2 chunks", "128\nfull"]
    oracle_labels = [oracle_labels[oracle_ids.index(item)] for item in oracle.index]
    oracle_x = np.arange(len(oracle))
    axes[0, 1].bar(oracle_x, oracle["ttft_mean_ms"], color=plt.get_cmap("Set2").colors[: len(oracle)])
    baseline = frame[frame["strategy_id"] == "full_prefill"]
    if not baseline.empty:
        axes[0, 1].axhline(
            float(baseline.iloc[0]["ttft_mean_ms"]),
            color="0.25",
            linestyle="--",
            linewidth=1.1,
            label="Full prefill",
        )
        axes[0, 1].legend(frameon=False)
    axes[0, 1].set_xticks(oracle_x, oracle_labels)
    axes[0, 1].set_ylabel("Mean TTFT (ms)")
    axes[0, 1].set_title("Oracle chunk coverage")
    axes[0, 1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(oracle["ttft_mean_ms"]):
        axes[0, 1].text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    reuse = frame[frame["strategy_id"] != "full_prefill"].copy()
    reuse_x = np.arange(len(reuse))
    width = 0.38
    axes[1, 0].bar(
        reuse_x - width / 2,
        100.0 * reuse["prefix_token_hit_rate"],
        width,
        label="Token hit",
        color="#4c78a8",
    )
    axes[1, 0].bar(
        reuse_x + width / 2,
        100.0 * reuse["request_hit_rate"],
        width,
        label="Request hit",
        color="#f2a65a",
    )
    axes[1, 0].set_xticks(
        reuse_x,
        [short_labels.get(item, item) for item in reuse["strategy_id"]],
        rotation=28,
        ha="right",
    )
    axes[1, 0].set_ylabel("Hit rate (%)")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].set_title("Observed prefix reuse")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.25)

    resources = frame[frame["strategy_id"].isin(STORAGE_STRATEGY_IDS[1:])].copy()
    resource_x = np.arange(len(resources))
    axes[1, 1].bar(
        resource_x,
        resources["max_physical_cache_tokens"],
        color="#5b8e7d",
        label="GPU cache tokens",
    )
    host_axis = axes[1, 1].twinx()
    host_mb = (
        resources["max_host_delta_bytes"] + resources["max_host_full_bytes"]
    ) / (1024.0 * 1024.0)
    host_axis.plot(resource_x, host_mb, color="#c45d39", marker="o", label="Host cache MiB")
    axes[1, 1].set_xticks(
        resource_x,
        [short_labels.get(item, item) for item in resources["strategy_id"]],
        rotation=28,
        ha="right",
    )
    axes[1, 1].set_ylabel("Peak physical GPU cache tokens")
    host_axis.set_ylabel("Peak host cache (MiB)")
    axes[1, 1].set_title("Storage pressure; annotations = wall time / evictions")
    axes[1, 1].grid(axis="y", alpha=0.25)
    for index, (_, row) in enumerate(resources.iterrows()):
        axes[1, 1].text(
            index,
            0.97 * float(row["max_physical_cache_tokens"]),
            f"{row['run_elapsed_seconds']:.0f}s / {int(row['family_evictions'])}",
            ha="center",
            va="top",
            fontsize=7,
            rotation=45,
        )
    handles1, labels1 = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = host_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_prefix_structure_plot(prefix: pd.DataFrame, output_path: Path) -> None:
    if prefix.empty:
        return
    order = [item[0] for item in PREFIX_METHODS]
    frame = prefix.set_index("method_id").reindex(order).dropna(subset=["method_label"])
    x = np.arange(len(frame))
    labels = frame["method_label"].tolist()
    colors = plt.get_cmap("Set2").colors[: len(frame)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    metrics = [
        ("mean_chunks_per_prefix", "Mean chunks per prefix", "Chunks"),
        ("mean_tokens_per_chunk", "Mean estimated tokens per chunk", "Tokens"),
        ("natural_boundary_coverage", "Natural-boundary coverage", "Rate (%)"),
        ("structural_token_coverage", "Structural reusable-token coverage", "Rate (%)"),
    ]
    for ax, (column, title, ylabel) in zip(axes.flat, metrics):
        values = frame[column].to_numpy(dtype=float)
        if "coverage" in column:
            values *= 100.0
        ax.bar(x, values, color=colors)
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            suffix = "%" if "coverage" in column else ""
            ax.text(index, value, f"{value:.1f}{suffix}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{prefix['dataset_label'].iloc[0]}: offline prefix-splitting structure", fontsize=13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_prefix_method_online_plot(prefix: pd.DataFrame, output_path: Path) -> None:
    required = {
        "runtime_request_hit_rate",
        "runtime_token_hit_rate",
        "runtime_ttft_speedup_percent",
        "runtime_host_peak_mb",
        "runtime_family_evictions",
        "runtime_reusable_prefix_tokens",
    }
    if prefix.empty or not required.issubset(prefix.columns):
        return
    if prefix["runtime_ttft_speedup_percent"].isna().all():
        return
    order = [item[0] for item in PREFIX_METHODS]
    frame = prefix.set_index("method_id").reindex(order).dropna(subset=["method_label"])
    x = np.arange(len(frame))
    labels = frame["method_label"].tolist()
    colors = plt.get_cmap("tab10").colors[: len(frame)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    metrics = [
        ("runtime_request_hit_rate", "Return request hit rate", "%", 100.0),
        ("runtime_token_hit_rate", "Return prefix token coverage", "%", 100.0),
        ("runtime_ttft_speedup_percent", "Online TTFT vs paired baseline", "%", 1.0),
        ("runtime_host_peak_mb", "Host KV peak", "MiB", 1.0),
        ("runtime_family_evictions", "Anchor-family evictions", "Count", 1.0),
        ("runtime_reusable_prefix_tokens", "Total reusable prefix tokens", "Tokens", 1.0),
    ]
    for ax, (column, title, ylabel, scale) in zip(axes.flat, metrics):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) * scale
        colors_for_values = colors[: len(values)]
        ax.bar(x, values, color=colors_for_values)
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if column == "runtime_ttft_speedup_percent":
            ax.axhline(0.0, color="0.35", linewidth=0.8)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            va = "bottom" if value >= 0 else "top"
            ax.text(index, value, f"{value:.1f}", ha="center", va=va, fontsize=8)
    fig.suptitle(
        f"{prefix['dataset_label'].iloc[0]}: online prefix-method effects",
        fontsize=13,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> list[str]:
    headers = [label for _, label, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if fmt != "text" else "---" for _, _, fmt in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for key, _, fmt in columns:
            value = row.get(key, "")
            if fmt == "text":
                values.append(str(value).replace("|", "\\|"))
            elif fmt != "text" and (value is None or not math.isfinite(float(value))):
                values.append("n/a")
            elif fmt == "int":
                values.append(str(int(round(float(value)))))
            elif fmt == "pct":
                values.append(f"{100.0 * float(value):.1f}%")
            elif fmt == "pct_value":
                values.append(f"{float(value):.1f}%")
            elif fmt == "mb":
                values.append(f"{float(value) / (1024.0 * 1024.0):.1f}")
            else:
                values.append(f"{float(value):.2f}")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_dataset_outputs(
    output_dir: Path,
    snapshot: dict[str, Any],
    summary: pd.DataFrame,
    prefix_structure: pd.DataFrame,
) -> None:
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    workloads_root = Path(snapshot["workloads_root"])
    for dataset in snapshot["selected_datasets"]:
        runtime = summary[summary["dataset"] == dataset].copy()
        if runtime.empty:
            continue
        dataset_dir = datasets_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix_structure[prefix_structure["dataset"] == dataset].copy()
        save_dataset_final_effect(runtime, dataset_dir / "final_effect.png")
        save_prefix_structure_plot(prefix, dataset_dir / "prefix_split_structure.png")
        save_prefix_method_online_plot(prefix, dataset_dir / "prefix_method_online_effect.png")

        workload_summary_path = workloads_root / dataset / "summary.json"
        workload = read_json(workload_summary_path) if workload_summary_path.is_file() else {}
        runtime["host_peak_bytes"] = runtime["max_host_delta_bytes"] + runtime["max_host_full_bytes"]
        preferred_order = {
            "full_prefill": 0,
            **{
                item: index + 1
                for index, item in enumerate(
                    [
                        "legacy_exact",
                        "gpu_only_128",
                        "gpu_host_full_128",
                        "tiered_no_prefetch_128",
                        "tiered_oracle_32",
                        "tiered_oracle_64",
                        "tiered_oracle_128",
                        "tiered_oracle_256",
                        "tiered_oracle_full",
                        "tiered_disk_build_128",
                        "tiered_disk_warm_128",
                    ]
                )
            },
        }
        runtime["report_order"] = runtime["strategy_id"].map(preferred_order).fillna(99)
        runtime = runtime.sort_values("report_order")
        online = runtime[runtime["strategy_id"] != "full_prefill"]
        best = online.loc[online["ttft_mean_ms"].idxmin()]
        oracle = runtime[runtime["strategy_id"].isin(
            [
                "tiered_oracle_32",
                "tiered_oracle_64",
                "tiered_oracle_128",
                "tiered_oracle_256",
                "tiered_oracle_full",
            ]
        )].copy()
        oracle["coverage"] = oracle["strategy_id"].map(
            {
                "tiered_oracle_32": "32 / 2 chunks",
                "tiered_oracle_64": "64 / 2 chunks",
                "tiered_oracle_128": "128 / 2 chunks",
                "tiered_oracle_256": "256 / 2 chunks",
                "tiered_oracle_full": "128 / full coverage",
            }
        )

        findings = [
            f"- 最低平均 TTFT 的非 full-prefill 策略是 **{best['strategy_label']}**："
            f"{best['ttft_mean_ms']:.2f} ms，相对 matched full-prefill "
            f"{best['ttft_speedup_percent']:+.1f}%。"
        ]
        full_rows = runtime[runtime["strategy_id"] == "tiered_oracle_full"]
        limited_rows = runtime[runtime["strategy_id"] == "tiered_oracle_128"]
        if not full_rows.empty and not limited_rows.empty:
            full = full_rows.iloc[0]
            limited = limited_rows.iloc[0]
            delta_ms = float(full["ttft_mean_ms"] - limited["ttft_mean_ms"])
            relation = "降低" if delta_ms < 0 else "增加"
            findings.append(
                f"- 完整覆盖 oracle 相对 128-token、最多两个 chunk 的 oracle "
                f"{relation} {abs(delta_ms):.2f} ms；其 token 命中率为 "
                f"{100.0 * full['prefix_token_hit_rate']:.1f}%。"
            )
        if not oracle.empty:
            best_oracle = oracle.loc[oracle["ttft_mean_ms"].idxmin()]
            findings.append(
                f"- Oracle 系列中最低平均 TTFT 为 **{best_oracle['coverage']}** "
                f"({best_oracle['ttft_mean_ms']:.2f} ms)。"
            )

        lines = [
            f"# {DATASET_LABELS.get(dataset, dataset)} 实验结果",
            "",
            f"生成时间：{datetime.now(timezone.utc).isoformat()}",
            "",
            "## Workload",
            "",
            f"- 形式：`{workload.get('form', 'unknown')}`",
            f"- 侧重点：{workload.get('focus', 'n/a')}",
            f"- 请求数：{int(workload.get('requests', len(runtime)))}",
            f"- 上下文数：{int(workload.get('contexts', 0))}",
        ]
        if workload.get("form") == "continuous":
            lines.extend(
                [
                    f"- LoRA transition：{int(workload.get('lora_transitions', 0))}",
                    f"- 同 LoRA transition rate：{100.0 * float(workload.get('same_lora_transition_rate', 0.0)):.1f}%",
                    f"- gap 后返回 rate：{100.0 * float(workload.get('return_after_gap_rate', 0.0)):.1f}%",
                ]
            )
        else:
            lines.extend(
                [
                    f"- 并列复用次数：{int(workload.get('context_reuses', 0))}",
                    f"- 请求级上下文复用率：{100.0 * float(workload.get('request_level_context_reuse_rate', 0.0)):.1f}%",
                ]
            )

        lines.extend(["", "## 最终效果", "", "![Final effect](final_effect.png)", "", *findings])
        lines.extend(["", "### Runtime 策略明细", ""])
        lines.extend(
            markdown_table(
                runtime,
                [
                    ("strategy_label", "策略", "text"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("ttft_p95_ms", "p95 (ms)", "float"),
                    ("ttft_speedup_percent", "相对 full-prefill", "pct_value"),
                    ("prefix_token_hit_rate", "token 命中率", "pct"),
                    ("request_hit_rate", "请求命中率", "pct"),
                    ("max_physical_cache_tokens", "GPU token 峰值", "int"),
                    ("host_peak_bytes", "Host 峰值 (MiB)", "mb"),
                    ("family_evictions", "淘汰", "int"),
                    ("run_elapsed_seconds", "cell wall time (s)", "float"),
                ],
            )
        )
        lines.extend(["", "### Oracle 覆盖与 chunk 对比", ""])
        if not oracle.empty:
            lines.extend(
                markdown_table(
                    oracle,
                    [
                        ("coverage", "切分 / 预取覆盖", "text"),
                        ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                        ("ttft_p95_ms", "p95 (ms)", "float"),
                        ("ttft_speedup_percent", "speedup", "pct_value"),
                        ("prefix_token_hit_rate", "token 命中率", "pct"),
                        ("max_physical_cache_tokens", "GPU token 峰值", "int"),
                        ("family_evictions", "淘汰", "int"),
                    ],
                )
            )

        lines.extend(
            [
                "",
                "## 前缀切分方法：结构与在线效果",
                "",
                "![Prefix split structure](prefix_split_structure.png)",
                "",
                "> 结构指标来自真实请求文本的离线分析；下面的 Runtime 列来自同一 workload 的派生边界在线实验。"
                "Fixed 64/128 表示 system/context 两类节点上限；Fixed 192/384 同理。"
                "Sentence 以自然句边界按约 128 token 合并，Keyword 在标题/话语关键词处切分且上限约 192 token，"
                "Reuse-aware 优先对齐历史 prefix 终点；运行时通过 prefix_segments 保留这些边界。",
                "",
            ]
        )
        if not prefix.empty and prefix["runtime_ttft_speedup_percent"].notna().any():
            lines.extend(
                [
                    "![Prefix method online effect](prefix_method_online_effect.png)",
                    "",
                    "> TTFT 变化是相对同一派生 workload cell 内 full-prefill 的配对均值；`+/-` 为请求级 bootstrap 95% CI 半宽。",
                    "",
                ]
            )
        if not prefix.empty:
            lines.extend(
                markdown_table(
                    prefix,
                    [
                        ("method_label", "方法", "text"),
                        ("runtime_request_hit_rate", "Runtime 返回命中", "pct"),
                        ("runtime_token_hit_rate", "Runtime token 覆盖", "pct"),
                        ("ttft_change_display", "TTFT 变化", "text"),
                        ("runtime_host_peak_mb", "Host KV 峰值 (MiB)", "float"),
                        ("runtime_peak_nodes", "Runtime 节点峰值", "int"),
                        ("runtime_family_evictions", "Anchor-family 淘汰", "int"),
                        ("runtime_reusable_prefix_tokens", "可复用 prefix token", "int"),
                        ("mean_chunks_per_prefix", "平均 chunk 数", "float"),
                        ("mean_tokens_per_chunk", "平均 token/chunk", "float"),
                        ("structural_token_coverage", "结构 token 覆盖", "pct"),
                        ("complete_prefix_repeat_rate", "完整 prefix 重复", "pct"),
                        ("natural_boundary_coverage", "自然边界覆盖", "pct"),
                        ("unique_nodes", "唯一节点", "int"),
                    ],
                )
            )
        lines.extend(
            [
                "",
                "## 解释边界",
                "",
                "- `oracle` 读取下一请求的真实 LoRA，只用于给预取上界定界。",
                "- `oracle_full` 的 0 表示不限制每个 LoRA 的预取 chunk 数，不表示零预取。",
                "- 物理 GGUF LoRA 与逻辑任务不匹配，本结果只衡量缓存、切换和 KV 路径。",
                f"- {int(snapshot.get('request_limit', 100))} 条请求且每个 cell 一次运行，结论仍需要跨进程重复运行验证。",
                "",
                "## 文件",
                "",
                f"- Runtime 原始结果：`../../runs/{dataset}/<strategy>/`",
                "- 全局汇总：`../../aggregate_results.csv`",
                "- 离线切分汇总：`../../prefix_structure_analysis.csv`",
                "- 在线切分汇总：`../../prefix_method_results.csv`",
            ]
        )
        (dataset_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    snapshot: dict[str, Any],
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    prefix_structure: pd.DataFrame,
    failures: list[dict[str, Any]],
) -> None:
    strategy_meta = {item["id"]: item for item in snapshot["strategies"]}
    best_rows = []
    for dataset, group in summary[summary["strategy_id"] != "full_prefill"].groupby("dataset", sort=False):
        best = group.loc[group["ttft_mean_ms"].idxmin()].copy()
        best_rows.append(best)
    best = pd.DataFrame(best_rows)
    if not best.empty:
        best["saved_ci95_ms"] = best.apply(
            lambda row: (
                f"[{row['paired_ttft_saved_ci95_low_ms']:.2f}, "
                f"{row['paired_ttft_saved_ci95_high_ms']:.2f}]"
            ),
            axis=1,
        )
        best["evidence"] = best.apply(
            lambda row: (
                "改善"
                if row["paired_ttft_saved_ci95_low_ms"] > 0
                else "回退"
                if row["paired_ttft_saved_ci95_high_ms"] < 0
                else "区间跨 0"
            ),
            axis=1,
        )

    storage_ids = STORAGE_STRATEGY_IDS
    storage = summary[summary["strategy_id"].isin(storage_ids)].copy()
    storage_mean = (
        storage.groupby(["strategy_id", "strategy_label"], as_index=False)
        .agg(
            run_elapsed_seconds=("run_elapsed_seconds", "mean"),
            ttft_mean_ms=("ttft_mean_ms", "mean"),
            ttft_p95_ms=("ttft_p95_ms", "mean"),
            ttft_speedup_percent=("ttft_speedup_percent", "mean"),
            prefix_token_hit_rate=("prefix_token_hit_rate", "mean"),
            max_physical_cache_tokens=("max_physical_cache_tokens", "mean"),
            max_host_delta_bytes=("max_host_delta_bytes", "mean"),
            max_host_full_bytes=("max_host_full_bytes", "mean"),
            family_evictions=("family_evictions", "sum"),
            delta_store_loaded=("delta_store_loaded", "sum"),
            delta_store_saved=("delta_store_saved", "sum"),
        )
    )
    storage_mean["order"] = storage_mean["strategy_id"].map(
        {item: index for index, item in enumerate(storage_ids)}
    )
    storage_mean = storage_mean.sort_values("order")

    chunk = summary[summary["category"] == "chunk"].copy()
    chunk = chunk.sort_values(["dataset", "chunk_tokens"])
    disk = summary[summary["category"] == "disk"].copy()
    prefix_mean = pd.DataFrame()
    if not prefix_structure.empty:
        prefix_mean = (
            prefix_structure.groupby(["method_id", "method_label"], as_index=False)
            .agg(
                mean_chunks_per_prefix=("mean_chunks_per_prefix", "mean"),
                mean_tokens_per_chunk=("mean_tokens_per_chunk", "mean"),
                structural_request_hit_rate=("structural_request_hit_rate", "mean"),
                structural_token_coverage=("structural_token_coverage", "mean"),
                complete_prefix_repeat_rate=("complete_prefix_repeat_rate", "mean"),
                natural_boundary_coverage=("natural_boundary_coverage", "mean"),
                unique_nodes=("unique_nodes", "mean"),
                runtime_request_hit_rate=("runtime_request_hit_rate", "mean"),
                runtime_token_hit_rate=("runtime_token_hit_rate", "mean"),
                runtime_ttft_speedup_percent=("runtime_ttft_speedup_percent", "mean"),
                runtime_ttft_ci95_half_percent=("runtime_ttft_ci95_half_percent", "mean"),
                runtime_host_peak_mb=("runtime_host_peak_mb", "mean"),
                runtime_peak_nodes=("runtime_peak_nodes", "mean"),
                runtime_family_evictions=("runtime_family_evictions", "sum"),
                runtime_reusable_prefix_tokens=("runtime_reusable_prefix_tokens", "mean"),
            )
        )
        prefix_mean["order"] = prefix_mean["method_id"].map(
            {item[0]: index for index, item in enumerate(PREFIX_METHODS)}
        )
        prefix_mean = prefix_mean.sort_values("order")
        prefix_mean["ttft_change_display"] = prefix_mean.apply(
            lambda row: (
                f"{float(row['runtime_ttft_speedup_percent']):+.2f}% +/- "
                f"{float(row['runtime_ttft_ci95_half_percent']):.2f}"
                if math.isfinite(float(row["runtime_ttft_speedup_percent"]))
                and math.isfinite(float(row["runtime_ttft_ci95_half_percent"]))
                else "n/a"
            ),
            axis=1,
        )

    def one_row(dataset: str, strategy_id: str) -> pd.Series | None:
        rows = summary[
            (summary["dataset"] == dataset)
            & (summary["strategy_id"] == strategy_id)
        ]
        return None if rows.empty else rows.iloc[0]

    interpretation: list[str] = []
    continuous_parts: list[str] = []
    for dataset, label in [
        ("msc_continuous", "MSC"),
        ("taskmaster_continuous", "Taskmaster"),
    ]:
        row = one_row(dataset, "gpu_host_full_128")
        if row is not None:
            continuous_parts.append(
                f"{label} {row['ttft_speedup_percent']:.1f}% "
                f"(95% CI [{row['paired_ttft_saved_ci95_low_ms']:.1f}, "
                f"{row['paired_ttft_saved_ci95_high_ms']:.1f}] ms)"
            )
    if continuous_parts:
        interpretation.append(
            "- 连续增长场景中，简单的 GPU + host-full / 128 路径仍有收益："
            + "；".join(continuous_parts)
            + "。完整 delta 分层并未延续这一收益。"
        )

    lmsys_rows = summary[
        (summary["dataset"] == "lmsys_continuous")
        & (summary["strategy_id"] != "full_prefill")
    ]
    if not lmsys_rows.empty:
        row = lmsys_rows.loc[lmsys_rows["ttft_mean_ms"].idxmin()]
        interpretation.append(
            f"- LMSYS-33K 的最佳非 full-prefill 策略仍回退 "
            f"{abs(row['ttft_speedup_percent']):.1f}%（{row['strategy_label']}）；"
            "普通短聊天中的复用量不足以覆盖缓存管理成本。"
        )

    opus_rows = chunk[chunk["dataset"] == "opus100_parallel"]
    if not opus_rows.empty:
        row = opus_rows.loc[opus_rows["ttft_mean_ms"].idxmin()]
        interpretation.append(
            f"- OPUS-100 是最适合并列复用的 workload：chunk={int(row['chunk_tokens'])} "
            f"时 token 命中率 {100.0 * row['prefix_token_hit_rate']:.1f}%，"
            f"TTFT 改善 {row['ttft_speedup_percent']:.1f}%，且没有 family eviction。"
        )

    xsum_rows = chunk[chunk["dataset"] == "xsum_parallel"]
    if not xsum_rows.empty:
        row = xsum_rows.loc[xsum_rows["ttft_mean_ms"].idxmin()]
        interpretation.append(
            f"- XSum 说明高命中率不是充分条件：表现最好的 chunk={int(row['chunk_tokens'])} "
            f"达到 {100.0 * row['prefix_token_hit_rate']:.1f}% token 命中率，"
            f"但 TTFT 仍回退 {abs(row['ttft_speedup_percent']):.1f}%；"
            "长 prefix 的 materialize、variant 构建和迁移成本抵消了 prefill 节省。"
        )

    no_prefetch = summary[summary["strategy_id"] == "tiered_no_prefetch_128"]
    if not no_prefetch.empty:
        regressions = int((no_prefetch["ttft_speedup_percent"] < 0).sum())
        interpretation.append(
            f"- Tiered no-prefetch / 128 在 {regressions}/{len(no_prefetch)} 个数据集上回退；"
            "oracle 仅表示下一 LoRA 完全已知的上界，不能视为可部署收益。"
        )

    if not disk.empty:
        saved = int(round(disk["delta_store_saved"].sum()))
        loaded = int(round(disk["delta_store_loaded"].sum()))
        load_ms = float(disk["delta_store_load_ms"].sum())
        interpretation.append(
            f"- 磁盘层实际保存 {saved} 个、加载 {loaded} 个 delta，累计加载 "
            f"{load_ms:.1f} ms；warm-load 只在部分数据集降低 cold-build TTFT，"
            "没有形成稳定的端到端优势。OPUS-100 保存/加载均为 0，不能据其结果声称磁盘层有效。"
        )

    if not delta.empty:
        interpretation.append(
            "- OPUS-100/XSum 的离线 KV delta cosine 较高，但 XSum 在线 TTFT 仍回退；"
            "可压缩性或重建精度不能替代端到端延迟测量。"
        )

    oracle_full = summary[summary["strategy_id"] == "tiered_oracle_full"]
    oracle_limited = summary[summary["strategy_id"] == "tiered_oracle_128"]
    if not oracle_full.empty and not oracle_limited.empty:
        merged = oracle_full[["dataset", "ttft_mean_ms"]].merge(
            oracle_limited[["dataset", "ttft_mean_ms"]],
            on="dataset",
            suffixes=("_full", "_limited"),
        )
        improved = int((merged["ttft_mean_ms_full"] < merged["ttft_mean_ms_limited"]).sum())
        interpretation.append(
            f"- `oracle_full` 在 {improved}/{len(merged)} 个数据集上低于 128-token、"
            "最多两个 chunk 的 oracle。它把“预取覆盖不足”和“完整 KV 构建/迁移成本”分开，"
            "但仍是使用真实下一 LoRA 的理论上界。"
        )

    lines = [
        "# Test7 真实数据集三级缓存与前缀切分实验",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 实验目的",
        "",
        "本实验在同一套真实请求上比较无缓存 full-prefill、旧版 exact-prefix、GPU-only、GPU + host full，以及包含 GPU full KV、host full/Q8 delta 和磁盘 delta store 的分层策略。连续数据用于观察逐轮增长和 LoRA 返回，并列数据用于观察同一 prefix 下的跨 LoRA 复用。",
        "",
        "> `oracle` 策略使用下一请求的真实 LoRA 作为预取上界，不代表线上可达到的预测精度。物理 GGUF LoRA 仅用于系统切换实验，并未针对这些逻辑任务训练。",
        "> `oracle_full` 使用 128-token 节点并将 `max-prefetch-chunks-per-lora=0`，其中 0 表示不限制预取节点数；普通 oracle 策略最多预取两个节点。",
        "",
        "## 实验配置",
        "",
        f"- 数据集：{', '.join(DATASET_LABELS.get(item, item) for item in snapshot['selected_datasets'])}",
        f"- 每个 workload 请求数：{snapshot['request_limit']}",
        f"- `n_ctx`：{snapshot['common_options']['n-ctx']}；GPU cache token 上限：{snapshot['common_options']['max-cache-tokens']}",
        f"- cache node / variant 上限：{snapshot['common_options']['max-cache-nodes']} / {snapshot['common_options']['max-cache-variants']}",
        f"- host delta / full 上限：256 MiB / 512 MiB（分层策略）",
        f"- 生成 token：{snapshot['common_options']['n-predict']}；delta 执行：`{snapshot['common_options']['delta-execution']}`",
        "- 正数 TTFT speedup 表示优于同一次运行内、同请求的 full-prefill；95% 区间由请求级配对 bootstrap 得到。",
        "- 存储汇总中的 Full-prefill 是 legacy cell 内的逐请求参考测量，不是额外运行的 cell；因此其 wall time 标为 n/a。",
        f"- 有效性校验：{len(summary[summary['strategy_id'] != 'full_prefill'])} 个策略 cell 完成，另有 {len(failures)} 个失败；"
        "任何 `failed` 模式或 TTFT <= 0 的在线行都会使 cell 失败。",
        "",
        "## 策略",
        "",
        "| ID | 策略 | 类别 | 说明 |",
        "|---|---|---|---|",
    ]
    for strategy_id in snapshot["selected_strategies"]:
        item = strategy_meta[strategy_id]
        options = item.get("options", {})
        description = (
            f"system-v2={options.get('system-v2')}, chunk={options.get('prefix-chunk-tokens')}, "
            f"cross-LoRA={options.get('cross-lora-policy')}, prefetch={options.get('prefetch-policy')}, "
            f"max-prefetch={options.get('max-prefetch-chunks-per-lora', snapshot['common_options'].get('max-prefetch-chunks-per-lora'))}, "
            f"disk={options.get('delta-store-policy')}"
        )
        lines.append(
            f"| `{strategy_id}` | {item['label']} | {item.get('category', '')} | {description} |"
        )

    lines.extend(
        [
            "",
            "## 主要结果",
            "",
            "### 各数据集最低平均 TTFT 的非 full-prefill 策略",
            "",
        ]
    )
    if not best.empty:
        lines.extend(
            markdown_table(
                best,
                [
                    ("dataset_label", "数据集", "text"),
                    ("strategy_label", "策略", "text"),
                    ("run_elapsed_seconds", "平均 cell wall time (s)", "float"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("ttft_p95_ms", "p95 TTFT (ms)", "float"),
                    ("ttft_speedup_percent", "相对 full-prefill", "pct_value"),
                    ("prefix_token_hit_rate", "prefix token 命中率", "pct"),
                    ("saved_ci95_ms", "配对节省 95% CI (ms)", "text"),
                    ("evidence", "证据", "text"),
                ],
            )
        )

    lines.extend(["", "![TTFT speedup](figures/ttft_speedup_heatmap.png)", ""])
    lines.extend([f"### 存储策略汇总（{summary['dataset'].nunique()} 个数据集等权平均）", ""])
    if not storage_mean.empty:
        lines.extend(
            markdown_table(
                storage_mean,
                [
                    ("strategy_label", "策略", "text"),
                    ("run_elapsed_seconds", "平均 cell wall time (s)", "float"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("ttft_p95_ms", "平均 p95 (ms)", "float"),
                    ("ttft_speedup_percent", "平均 speedup", "pct_value"),
                    ("prefix_token_hit_rate", "命中率", "pct"),
                    ("max_physical_cache_tokens", "GPU token 峰值", "float"),
                    ("max_host_delta_bytes", "host delta 峰值 (MiB)", "mb"),
                    ("max_host_full_bytes", "host full 峰值 (MiB)", "mb"),
                    ("family_evictions", "淘汰次数", "int"),
                ],
            )
        )
    lines.extend(
        [
            "",
            "![Storage ablation](figures/storage_ablation.png)",
            "",
            "![Storage resource tradeoff](figures/storage_resource_tradeoff.png)",
            "",
            "### Prefix chunk 大小对比",
            "",
        ]
    )
    if not chunk.empty:
        lines.extend(
            markdown_table(
                chunk,
                [
                    ("dataset_label", "数据集", "text"),
                    ("chunk_tokens", "chunk", "int"),
                    ("run_elapsed_seconds", "cell wall time (s)", "float"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("ttft_p95_ms", "p95 (ms)", "float"),
                    ("ttft_speedup_percent", "speedup", "pct_value"),
                    ("prefix_token_hit_rate", "token 命中率", "pct"),
                    ("max_physical_cache_tokens", "GPU token 峰值", "float"),
                    ("family_evictions", "淘汰", "int"),
                ],
            )
        )
    lines.extend(["", "![Prefix chunk ablation](figures/prefix_chunk_ablation.png)", ""])

    lines.extend(["### 磁盘 delta store", ""])
    if disk.empty:
        lines.append("本次结果中没有完成磁盘策略运行。")
    else:
        lines.extend(
            markdown_table(
                disk.sort_values(["dataset", "strategy_id"]),
                [
                    ("dataset_label", "数据集", "text"),
                    ("strategy_label", "策略", "text"),
                    ("ttft_mean_ms", "平均 TTFT (ms)", "float"),
                    ("delta_store_saved", "保存 delta", "int"),
                    ("delta_store_loaded", "加载 delta", "int"),
                    ("delta_store_load_ms", "加载耗时 (ms)", "float"),
                ],
            )
        )

    lines.extend(["", "### 跨 LoRA KV delta 质量", ""])
    if delta.empty:
        lines.append("连续 workload 没有 exact-prefix pair；本次没有可用的离线 delta quality 行。")
    else:
        delta_summary = (
            delta.groupby(["dataset", "dataset_label"], as_index=False)
            .agg(
                pairs=("pair_name", "count"),
                kv_cos=("kv_cos", "mean"),
                kv_l2=("kv_l2", "mean"),
                logical_saved_rate=("logical_saved_rate", "mean"),
                delta_build_ok=("delta_build_ok", "sum"),
            )
        )
        lines.extend(
            markdown_table(
                delta_summary,
                [
                    ("dataset_label", "数据集", "text"),
                    ("pairs", "pair", "int"),
                    ("kv_cos", "平均 cosine", "float"),
                    ("kv_l2", "平均 L2", "float"),
                    ("logical_saved_rate", "逻辑节省率", "pct"),
                    ("delta_build_ok", "构建成功", "int"),
                ],
            )
        )
        lines.extend(["", "![Delta quality](figures/delta_quality.png)"])

    lines.extend(
        [
            "",
            "### 前缀切分方法：结构与在线效果",
            "",
            "> 结构列来自真实文本离线分析；Runtime 列来自将同一边界写入 `prefix_segments` 后的 llama.cpp 在线实验，五种方法使用相同 oracle 预取和三级存储配置。",
            "> Fixed A/B 分别限制 system/context 节点；Sentence 约 128 token 并对齐句界，Keyword 约 192 token 并在标题/话语关键词处切分，Reuse-aware 优先对齐历史 prefix 终点。",
            "",
        ]
    )
    if not prefix_mean.empty:
        lines.extend(
            markdown_table(
                prefix_mean,
                [
                    ("method_label", "方法", "text"),
                    ("runtime_request_hit_rate", "Runtime 返回命中", "pct"),
                    ("runtime_token_hit_rate", "Runtime token 覆盖", "pct"),
                    ("ttft_change_display", "平均 TTFT 变化", "text"),
                    ("runtime_host_peak_mb", "平均 Host KV 峰值 (MiB)", "float"),
                    ("runtime_peak_nodes", "平均节点峰值", "float"),
                    ("runtime_family_evictions", "总淘汰", "int"),
                    ("mean_chunks_per_prefix", "平均 chunk 数", "float"),
                    ("mean_tokens_per_chunk", "平均 token/chunk", "float"),
                    ("structural_token_coverage", "结构 token 覆盖", "pct"),
                    ("complete_prefix_repeat_rate", "完整 prefix 重复", "pct"),
                    ("natural_boundary_coverage", "自然边界覆盖", "pct"),
                    ("unique_nodes", "平均唯一节点", "float"),
                ],
            )
        )

    lines.extend(["", "### 各数据集独立报告", ""])
    for dataset in snapshot["selected_datasets"]:
        label = DATASET_LABELS.get(dataset, dataset)
        lines.append(f"- [{label}](datasets/{dataset}/RESULTS.md)")

    lines.extend(
        [
            "",
            "## 如何解释这些结果",
            "",
            "- 逐 cell 的命中、缓存峰值、模式和失败行见 [`DIAGNOSIS.md`](DIAGNOSIS.md)。",
            *interpretation,
            "",
            "## 局限性",
            "",
            f"1. 每个数据集包含 {int(snapshot.get('request_limit', 100))} 条请求，但每个矩阵 cell 默认只运行一次，因此不是最终统计结论。",
            "2. 每个矩阵 cell 默认只运行一次；请求级 bootstrap 反映 trace 内变化，不包含跨进程、跨温度或跨机器方差。",
            "3. Full-prefill 在每个进程中先执行，首请求可能包含额外 warm-up；比较使用同进程配对 baseline，但仍建议后续增加重复次数和随机化顺序。",
            "4. 当前 GGUF LoRA 与逻辑任务不匹配，因此只能验证缓存、切换和 KV 重建行为，不能据此比较生成质量。",
            "5. XSum 的 QA 和 headline 使用摘要作为代理 reference；本实验未计算 ROUGE、BLEU 或人工质量指标。",
            "6. test7 复用 test4 system-v2 运行时；它的固定切分是在节点建立后再按容量淘汰，尚不是基于边际收益的动态 admission。",
            "",
            "## 原始结果",
            "",
            "- `aggregate_results.csv`：每个 dataset/strategy 的汇总指标。",
            "- `request_results.csv`：请求级 online 与配对 full-prefill 指标。",
            "- `delta_quality.csv`：OPUS-100/XSum exact-prefix pair 的离线 delta 质量。",
            "- `prefix_structure_analysis.csv`：五种切分方法的离线结构指标。",
            "- `prefix_method_results.csv`：五种切分方法的在线 TTFT、返回命中、Host KV 和淘汰指标。",
            "- `datasets/<dataset>/`：每个数据集的独立 Markdown 和两张效果图。",
            "- `runs/<dataset>/<strategy>/`：运行日志、完整 CSV、缓存树和参数快照。",
        ]
    )
    if failures:
        lines.extend(["", "## 未完成运行", ""])
        for failure in failures:
            lines.append(
                f"- `{failure.get('dataset')}/{failure.get('strategy_id')}`: {failure.get('error', 'unknown error')}"
            )

    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_path = args.output_dir / "matrix_snapshot.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"matrix snapshot not found: {snapshot_path}")
    snapshot = read_json(snapshot_path)
    strategy_by_id = {item["id"]: item for item in snapshot["strategies"]}

    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted((args.output_dir / "runs").glob("*/*/run_manifest.json")):
        manifest = read_json(path)
        # Prefix-method cells are auxiliary runs merged from prefix_method_results.csv.
        if manifest.get("strategy_id") not in strategy_by_id:
            continue
        if manifest.get("status") == "ok":
            manifests.append(manifest)
        else:
            failures.append(manifest)
    if not manifests:
        raise ValueError(f"no successful test7 runs found under {args.output_dir / 'runs'}")

    summaries: list[dict[str, Any]] = []
    request_frames: list[pd.DataFrame] = []
    baseline_by_dataset: dict[str, pd.DataFrame] = {}
    for manifest in manifests:
        strategy = strategy_by_id[manifest["strategy_id"]]
        summary, requests, baseline = summarize_run(manifest, strategy)
        summaries.append(summary)
        request_frames.append(requests)
        if manifest["strategy_id"] == "legacy_exact":
            baseline_by_dataset[manifest["dataset"]] = baseline

    for dataset, baseline in baseline_by_dataset.items():
        summaries.append(baseline_summary(dataset, baseline))

    summary_frame = pd.DataFrame(summaries)
    dataset_order = {item: index for index, item in enumerate(snapshot["selected_datasets"])}
    strategy_order = {"full_prefill": 0}
    strategy_order.update(
        {item: index + 1 for index, item in enumerate(snapshot["selected_strategies"])}
    )
    summary_frame["dataset_order"] = summary_frame["dataset"].map(dataset_order)
    summary_frame["strategy_order"] = summary_frame["strategy_id"].map(strategy_order)
    summary_frame = summary_frame.sort_values(["dataset_order", "strategy_order"])
    summary_frame.drop(columns=["dataset_order", "strategy_order"]).to_csv(
        args.output_dir / "aggregate_results.csv", index=False
    )

    requests = pd.concat(request_frames, ignore_index=True)
    requests.to_csv(args.output_dir / "request_results.csv", index=False)
    delta = collect_delta_quality(manifests)
    delta.to_csv(args.output_dir / "delta_quality.csv", index=False)
    prefix_structure = analyze_prefix_structure(snapshot)
    for column in PREFIX_RUNTIME_COLUMNS:
        if column not in prefix_structure:
            prefix_structure[column] = np.nan
    prefix_method_results_path = args.output_dir / "prefix_method_results.csv"
    if prefix_method_results_path.is_file() and prefix_method_results_path.stat().st_size:
        method_results = pd.read_csv(prefix_method_results_path)
        merge_columns = [
            "dataset",
            "method_id",
            *[column for column in PREFIX_RUNTIME_COLUMNS if column != "ttft_change_display"],
        ]
        available = [column for column in merge_columns if column in method_results.columns]
        prefix_structure = prefix_structure.drop(
            columns=[column for column in available if column not in {"dataset", "method_id"}],
            errors="ignore",
        ).merge(method_results[available], on=["dataset", "method_id"], how="left")
        for column in PREFIX_RUNTIME_COLUMNS:
            if column not in prefix_structure:
                prefix_structure[column] = np.nan
    prefix_structure["ttft_change_display"] = prefix_structure.apply(
        lambda row: (
            f"{float(row['runtime_ttft_speedup_percent']):+.2f}% +/- "
            f"{float(row['runtime_ttft_ci95_half_percent']):.2f}"
            if math.isfinite(float(row["runtime_ttft_speedup_percent"]))
            and math.isfinite(float(row["runtime_ttft_ci95_half_percent"]))
            else "n/a"
        ),
        axis=1,
    )
    prefix_structure.to_csv(
        args.output_dir / "prefix_structure_analysis.csv",
        index=False,
    )

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()
    save_speedup_heatmap(summary_frame, figures_dir)
    save_chunk_ablation(summary_frame, figures_dir)
    save_storage_ablation(summary_frame, figures_dir)
    save_resource_tradeoff(summary_frame, figures_dir)
    save_delta_quality(delta, figures_dir)
    write_dataset_outputs(
        args.output_dir,
        snapshot,
        summary_frame,
        prefix_structure,
    )
    write_report(
        args.output_dir,
        snapshot,
        summary_frame,
        delta,
        prefix_structure,
        failures,
    )

    print(
        f"test7 analysis complete: runs={len(manifests)} "
        f"aggregate_rows={len(summary_frame)} prefix_rows={len(prefix_structure)} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
