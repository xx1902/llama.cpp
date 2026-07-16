#!/usr/bin/env python3
"""Convert the downloaded part of the 87-LoRA manifest to GGUF."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\ecnu_experiment\Model\LoRA\Qwen2.5_1.5B_87")
DEFAULT_GGUF_ROOT = Path(r"D:\ecnu_experiment\gguf-qwen2.5_1.5B")
DEFAULT_CONVERTER = Path(
    r"D:\ecnu_experiment\LLama.cpp\llama.cpp\convert_lora_to_gguf.py"
)
DEFAULT_PYTHON = r"D:\anaconda\envs\qwen2.5_vl\python.exe"


def slug(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repo_id)


def parse_id_set(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid ID range: {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("loras"), list):
        raise ValueError(f"invalid manifest format: {path}")
    return data


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

    csv_path = path.with_suffix(".csv")
    fields = [
        "lora_id",
        "app_slot",
        "repo_id",
        "group",
        "source",
        "status",
        "rank",
        "alpha",
        "base_model",
        "adapter_dir",
        "gguf_path",
        "gguf_size_mb",
        "convert_error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(data["loras"], key=lambda item: int(item["lora_id"])))


def adapter_files_valid(directory: Path) -> bool:
    return (directory / "adapter_config.json").is_file() and (
        (directory / "adapter_model.safetensors").is_file()
        or (directory / "adapter_model.bin").is_file()
    )


def gguf_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as file:
        return file.read(4) == b"GGUF"


def output_path_for(entry: dict[str, Any], gguf_root: Path, outtype: str) -> Path:
    lora_id = int(entry["lora_id"])
    repo_name = slug(entry["repo_id"])
    return gguf_root / f"lora-{lora_id:03d}-{repo_name}-{outtype}.gguf"


def validate_preconditions(args: argparse.Namespace, data: dict[str, Any]) -> Path:
    if not args.converter.is_file():
        raise FileNotFoundError(f"llama.cpp converter is missing: {args.converter}")
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Python interpreter is missing: {args.python}")
    base_config = args.root / "_base_qwen25_1.5b_instruct_config"
    if not (base_config / "config.json").is_file():
        raise FileNotFoundError(
            f"base config is missing: {base_config}; run the download script first"
        )
    expected = int(data.get("target_total") or 87)
    if len(data["loras"]) != expected:
        raise RuntimeError(
            f"manifest contains {len(data['loras'])}/{expected} LoRAs; finish downloading first"
        )
    return base_config


def run_convert(args: argparse.Namespace) -> None:
    data = load_manifest(args.manifest)
    base_config = validate_preconditions(args, data)
    selected_ids = parse_id_set(args.only_ids)
    args.gguf_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.root / "conversion_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        entry
        for entry in sorted(data["loras"], key=lambda item: int(item["lora_id"]))
        if entry.get("source") == "downloaded"
        and (selected_ids is None or int(entry["lora_id"]) in selected_ids)
    ]
    if args.max_count > 0:
        candidates = candidates[: args.max_count]

    converted = 0
    skipped = 0
    failed = 0
    for index, entry in enumerate(candidates, 1):
        lora_id = int(entry["lora_id"])
        adapter_dir = Path(entry.get("adapter_dir", ""))
        output_path = output_path_for(entry, args.gguf_root, args.outtype)
        print(f"\n[{index}/{len(candidates)}] lora_id={lora_id} {entry['repo_id']}")

        if not adapter_files_valid(adapter_dir):
            entry["status"] = "convert_failed"
            entry["convert_error"] = f"adapter files are incomplete: {adapter_dir}"
            failed += 1
            write_manifest(args.manifest, data)
            print(f"  ERROR: {entry['convert_error']}")
            if args.fail_fast:
                break
            continue

        if gguf_valid(output_path) and not args.force:
            entry["status"] = "converted"
            entry["gguf_path"] = str(output_path)
            entry["gguf_size_mb"] = round(output_path.stat().st_size / 1024**2, 6)
            entry["convert_error"] = ""
            skipped += 1
            write_manifest(args.manifest, data)
            print(f"  skip existing: {output_path}")
            continue

        command = [
            args.python,
            str(args.converter),
            "--base",
            str(base_config),
            "--outfile",
            str(output_path),
            "--outtype",
            args.outtype,
            str(adapter_dir),
        ]
        print("  " + subprocess.list2cmdline(command))
        if args.dry_run:
            continue

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        log_path = log_dir / f"lora-{lora_id:03d}.log"
        log_path.write_text(
            process.stdout + "\n--- STDERR ---\n" + process.stderr,
            encoding="utf-8",
        )

        if process.returncode == 0 and gguf_valid(output_path):
            entry["status"] = "converted"
            entry["gguf_path"] = str(output_path)
            entry["gguf_size_mb"] = round(output_path.stat().st_size / 1024**2, 6)
            entry["convert_error"] = ""
            converted += 1
            print(f"  converted: {output_path}")
        else:
            entry["status"] = "convert_failed"
            entry["convert_error"] = (
                f"returncode={process.returncode}; log={log_path}"
            )
            failed += 1
            print(f"  ERROR: {entry['convert_error']}")
        write_manifest(args.manifest, data)
        if failed and args.fail_fast:
            break

    print(
        f"\nconversion summary: converted={converted}, skipped={skipped}, "
        f"failed={failed}, selected={len(candidates)}"
    )
    if failed:
        raise SystemExit(2)


def run_audit(args: argparse.Namespace) -> None:
    data = load_manifest(args.manifest)
    expected = int(data.get("target_total") or 87)
    entries = sorted(data["loras"], key=lambda item: int(item["lora_id"]))
    errors: list[str] = []
    if len(entries) != expected:
        errors.append(f"manifest count is {len(entries)}, expected {expected}")
    ids = [int(entry["lora_id"]) for entry in entries]
    if ids != list(range(expected)):
        errors.append("LoRA IDs are not exactly 0..86")
    if len({entry["repo_id"] for entry in entries}) != len(entries):
        errors.append("repository IDs are not unique")

    group_counts: dict[str, int] = {}
    converted_count = 0
    total_bytes = 0
    for entry in entries:
        group = entry.get("group", "unknown")
        group_counts[group] = group_counts.get(group, 0) + 1
        gguf_path = Path(entry.get("gguf_path", ""))
        if not gguf_valid(gguf_path):
            errors.append(f"lora_id={entry['lora_id']} has no valid GGUF: {gguf_path}")
        else:
            converted_count += 1
            total_bytes += gguf_path.stat().st_size

    print(f"valid GGUF: {converted_count}/{expected}")
    print(f"total GGUF size: {total_bytes / 1024**3:.3f} GiB")
    print("groups:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}")
    if errors:
        print("audit errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(2)
    print("audit passed: 87 unique LoRAs all have valid GGUF files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["convert", "audit"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gguf-root", type=Path, default=DEFAULT_GGUF_ROOT)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        default="f16",
    )
    parser.add_argument("--only-ids", help="comma-separated IDs or ranges, e.g. 12-20,25")
    parser.add_argument("--max-count", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.root / "lora_87_manifest.json"
    return args


def main() -> None:
    args = parse_args()
    if args.action == "convert":
        run_convert(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()
