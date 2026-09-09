#!/usr/bin/env python3
"""Focused invariants for the local prefix splitting experiment methods."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from analyze_results import (
    PREFIX_METHODS_V1_ABLATION,
    PREFIX_METHODS_V2_ABLATION,
    PREFIX_METHODS_V3,
    build_causal_lora_focus_profiles,
    build_lora_focus_profile,
    split_prefix_adaptive_v2,
    split_prefix_experiment_method,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOADS_ROOT = SCRIPT_DIR.parent / "lora-base-datasets" / "output" / "real_100"


def read_rows(dataset: str, limit: int = 5) -> list[dict]:
    path = WORKLOADS_ROOT / dataset / "grouped" / "grouped_requests.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:limit]]


class PrefixMethodInvariantTests(unittest.TestCase):
    def test_causal_profiles_do_not_observe_future_requests(self) -> None:
        rows = read_rows("opus100_parallel")
        profiles = build_causal_lora_focus_profiles(rows)
        self.assertEqual([profile["request_count"] for profile in profiles], [1, 2, 3, 4, 5])

    def test_all_new_methods_preserve_prefix_bytes(self) -> None:
        methods = [
            *[item[0] for item in PREFIX_METHODS_V1_ABLATION],
            *[item[0] for item in PREFIX_METHODS_V2_ABLATION],
            *[item[0] for item in PREFIX_METHODS_V3],
        ]
        for dataset, context_rate, exact_rate in [
            ("sharegpt_continuous", 0.83, 0.0),
            ("xsum_parallel", 0.0, 0.66),
        ]:
            rows = read_rows(dataset)
            profile = build_lora_focus_profile(rows)
            prefixes_by_context: dict[str, list[str]] = {}
            for row in rows:
                prefixes_by_context.setdefault(str(row["context_id"]), []).append(
                    str(row["common_prefix_text"])
                )
            for row in rows:
                text = str(row["common_prefix_text"])
                reuse_points = {
                    len(prefix)
                    for prefix in prefixes_by_context[str(row["context_id"])]
                    if len(prefix) < len(text) and text.startswith(prefix)
                }
                for method in methods:
                    with self.subTest(dataset=dataset, request=row["request_id"], method=method):
                        chunks, roles = split_prefix_experiment_method(
                            row,
                            method,
                            reuse_points,
                            context_rate,
                            exact_rate,
                            profile,
                        )
                        self.assertEqual("".join(chunks), text)
                        self.assertTrue(all(chunks))
                        self.assertEqual(len(chunks), len(roles))

    def test_v2_reference_matches_original_v2(self) -> None:
        rows = read_rows("xsum_parallel")
        profile = build_lora_focus_profile(rows)
        for row in rows:
            expected = split_prefix_adaptive_v2(row, set(), 0.0, 0.66)
            actual, _ = split_prefix_experiment_method(
                row,
                "adaptive_v2_reference",
                set(),
                0.0,
                0.66,
                profile,
            )
            self.assertEqual(actual, expected)

    def test_v3_scores_short_high_reaccess_prefixes(self) -> None:
        rows = read_rows("taskmaster_continuous")
        profiles = build_causal_lora_focus_profiles(rows)
        prefixes_by_context: dict[str, list[str]] = {}
        for row in rows:
            prefixes_by_context.setdefault(str(row["context_id"]), []).append(
                str(row["common_prefix_text"])
            )

        signatures: dict[str, list[tuple[int, ...]]] = {}
        for method in ("lora_focus_v3", "lora_focus_v3_no_focus"):
            signatures[method] = []
            for index, row in enumerate(rows):
                text = str(row["common_prefix_text"])
                reuse_points = {
                    len(prefix)
                    for prefix in prefixes_by_context[str(row["context_id"])]
                    if len(prefix) < len(text) and text.startswith(prefix)
                }
                chunks, _ = split_prefix_experiment_method(
                    row,
                    method,
                    reuse_points,
                    context_reaccess_rate=0.93,
                    exact_prefix_rate=0.0,
                    lora_focus_profile=profiles[index],
                )
                signatures[method].append(tuple(len(chunk) for chunk in chunks))

        self.assertTrue(any(len(signature) > 1 for signature in signatures["lora_focus_v3"]))
        self.assertNotEqual(signatures["lora_focus_v3"], signatures["lora_focus_v3_no_focus"])


if __name__ == "__main__":
    unittest.main()
