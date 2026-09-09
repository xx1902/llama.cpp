#!/usr/bin/env python3
"""Run the test7 real-dataset prefix-cache experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BINARY = REPO_ROOT / "build" / "bin" / "Release" / "llama-lora-base-test7.exe"
DEFAULT_WORKLOADS_ROOT = (
    REPO_ROOT / "examples" / "lora-base-datasets" / "output" / "real_100"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "real_100_matrix"
DEFAULT_MATRIX = SCRIPT_DIR / "experiment_matrix.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def option_args(options: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for name, value in options.items():
        if isinstance(value, bool):
            value = 1 if value else 0
        args.extend([f"--{name}", str(value)])
    return args


def inspect_result_rows(path: Path) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    invalid: list[str] = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            benchmark = row.get("benchmark", "unknown")
            counts[benchmark] = counts.get(benchmark, 0) + 1
            if benchmark != "online":
                continue
            mode = row.get("mode", "")
            try:
                ttft_ms = float(row.get("ttft_ms", "nan"))
            except ValueError:
                ttft_ms = math.nan
            if "failed" in mode.lower() or not math.isfinite(ttft_ms) or ttft_ms <= 0:
                invalid.append(
                    f"request={row.get('request_id', '?')} mode={mode or '?'} "
                    f"ttft_ms={row.get('ttft_ms', '?')}"
                )
    return counts, invalid


def completed_cell(run_dir: Path, expected_requests: int) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "online_request_results.csv"
    if not manifest_path.is_file() or not results_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        counts, invalid = inspect_result_rows(results_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "ok"
        and counts.get("baseline") == expected_requests
        and counts.get("online") == expected_requests
        and not invalid
    )


def validate_inputs(
    binary: Path,
    workloads_root: Path,
    datasets: list[str],
) -> None:
    if not binary.is_file():
        raise FileNotFoundError(
            f"test7 binary not found: {binary}. Build llama-lora-base-test7 first."
        )
    for dataset in datasets:
        workload_dir = workloads_root / dataset
        required = [
            workload_dir / "lora_groups.json",
            workload_dir / "grouped" / "grouped_requests.jsonl",
            workload_dir / "delta" / "delta_requests.jsonl",
            workload_dir / "delta" / "delta_pairs.jsonl",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"workload {dataset} is incomplete: {', '.join(missing)}"
            )


def run_cell(
    *,
    binary: Path,
    workload_dir: Path,
    output_root: Path,
    dataset: str,
    strategy: dict[str, Any],
    common_options: dict[str, Any],
    request_limit: int,
    timeout_seconds: int,
    force: bool,
) -> dict[str, Any]:
    strategy_id = str(strategy["id"])
    run_dir = output_root / "runs" / dataset / strategy_id
    expected_requests = request_limit
    if not force and completed_cell(run_dir, expected_requests):
        print(f"[skip] {dataset:24s} {strategy_id}", flush=True)
        return read_json(run_dir / "run_manifest.json")

    run_dir.mkdir(parents=True, exist_ok=True)
    options = dict(common_options)
    options.update(strategy.get("options", {}))
    options["max-online-requests"] = request_limit
    options["max-delta-pairs"] = request_limit if strategy.get("measure_delta_pairs") else 0

    command = [
        str(binary),
        "--workload-dir",
        str(workload_dir),
        "--output-dir",
        str(run_dir),
    ]
    delta_store_key = strategy.get("delta_store_key")
    delta_store_dir: Path | None = None
    if delta_store_key:
        delta_store_dir = output_root / "delta_store" / dataset / str(delta_store_key)
        delta_store_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--delta-store-dir", str(delta_store_dir)])
    command.extend(option_args(options))

    started_at = utc_now()
    start = time.perf_counter()
    log_path = run_dir / "run.log"
    print(f"[run ] {dataset:24s} {strategy_id}", flush=True)
    status = "failed"
    return_code: int | None = None
    error = ""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_seconds,
                check=False,
            )
        return_code = process.returncode
        results_path = run_dir / "online_request_results.csv"
        if return_code == 0 and results_path.is_file():
            counts, invalid = inspect_result_rows(results_path)
            if (
                counts.get("baseline") == expected_requests
                and counts.get("online") == expected_requests
                and not invalid
            ):
                status = "ok"
            elif invalid:
                preview = "; ".join(invalid[:3])
                suffix = f"; ... ({len(invalid)} invalid rows)" if len(invalid) > 3 else ""
                error = f"invalid online results: {preview}{suffix}"
            else:
                error = f"unexpected result row counts: {counts}"
        else:
            error = f"runtime exited with code {return_code}"
    except subprocess.TimeoutExpired:
        error = f"timed out after {timeout_seconds} seconds"
    except OSError as exc:
        error = str(exc)

    elapsed = time.perf_counter() - start
    manifest = {
        "dataset": dataset,
        "strategy_id": strategy_id,
        "strategy_label": strategy["label"],
        "category": strategy.get("category", "unknown"),
        "chunk_tokens": strategy.get("chunk_tokens"),
        "status": status,
        "error": error,
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": elapsed,
        "expected_requests_per_benchmark": expected_requests,
        "workload_dir": str(workload_dir),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "delta_store_dir": str(delta_store_dir) if delta_store_dir else "",
        "options": options,
        "command": command,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    marker = "ok" if status == "ok" else "FAIL"
    print(f"[{marker:4s}] {dataset:24s} {strategy_id} ({elapsed:.1f}s)", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--workloads-root", type=Path, default=DEFAULT_WORKLOADS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--datasets", type=comma_list)
    parser.add_argument("--strategies", type=comma_list)
    parser.add_argument("--request-limit", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = read_json(args.matrix)
    datasets = args.datasets or list(matrix["datasets"])
    all_strategies = list(matrix["strategies"])
    strategy_ids = {str(strategy["id"]) for strategy in all_strategies}

    unknown_datasets = sorted(set(datasets) - set(matrix["datasets"]))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}")
    if args.strategies:
        unknown_strategies = sorted(set(args.strategies) - strategy_ids)
        if unknown_strategies:
            raise ValueError(f"unknown strategies: {unknown_strategies}")
        selected = set(args.strategies)
        strategies = [s for s in all_strategies if s["id"] in selected]
    else:
        strategies = all_strategies

    request_limit = args.request_limit
    if request_limit <= 0:
        raise ValueError("--request-limit must be positive")
    if args.quick:
        request_limit = min(request_limit, 5)
        quick_ids = {"legacy_exact", "gpu_only_128", "tiered_oracle_128"}
        strategies = [strategy for strategy in strategies if strategy["id"] in quick_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        **matrix,
        "source_matrix": str(args.matrix.resolve()),
        "binary": str(args.binary.resolve()),
        "workloads_root": str(args.workloads_root.resolve()),
        "selected_datasets": datasets,
        "selected_strategies": [strategy["id"] for strategy in strategies],
        "request_limit": request_limit,
        "quick": args.quick,
    }
    write_json(args.output_dir / "matrix_snapshot.json", snapshot)

    manifests: list[dict[str, Any]] = []
    if not args.analyze_only:
        validate_inputs(args.binary, args.workloads_root, datasets)
        total = len(datasets) * len(strategies)
        print(
            f"test7 matrix: {len(datasets)} datasets x {len(strategies)} strategies "
            f"= {total} runs",
            flush=True,
        )
        for dataset in datasets:
            for strategy in strategies:
                required = strategy.get("requires_strategy")
                if required:
                    required_dir = args.output_dir / "runs" / dataset / str(required)
                    if not completed_cell(required_dir, request_limit):
                        manifest = {
                            "dataset": dataset,
                            "strategy_id": strategy["id"],
                            "status": "failed",
                            "error": f"required strategy is incomplete: {required}",
                        }
                        manifests.append(manifest)
                        print(
                            f"[FAIL] {dataset:24s} {strategy['id']} "
                            f"(missing {required})",
                            flush=True,
                        )
                        continue
                manifests.append(
                    run_cell(
                        binary=args.binary,
                        workload_dir=args.workloads_root / dataset,
                        output_root=args.output_dir,
                        dataset=dataset,
                        strategy=strategy,
                        common_options=matrix["common_options"],
                        request_limit=request_limit,
                        timeout_seconds=args.timeout_seconds,
                        force=args.force,
                    )
                )
        write_json(
            args.output_dir / "run_summary.json",
            {
                "generated_at": utc_now(),
                "runs": manifests,
                "ok": sum(manifest.get("status") == "ok" for manifest in manifests),
                "failed": sum(manifest.get("status") != "ok" for manifest in manifests),
            },
        )

    if not args.no_analyze:
        analyze_command = [
            sys.executable,
            str(SCRIPT_DIR / "analyze_results.py"),
            "--output-dir",
            str(args.output_dir),
        ]
        print("[info] analyzing results", flush=True)
        analysis = subprocess.run(analyze_command, cwd=REPO_ROOT, check=False)
        if analysis.returncode != 0:
            return analysis.returncode

    failures = [manifest for manifest in manifests if manifest.get("status") != "ok"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
