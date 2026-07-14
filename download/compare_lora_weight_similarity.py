#!/usr/bin/env python3
"""Compare effective LoRA weight updates without materializing large B @ A matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open


KEY_PATTERN = re.compile(
    r"(model\.layers\.\d+\.(?:mlp|self_attn)\.[^.]+)"
    r"\.lora_([AB])\.weight$"
)


def load_metadata(directory: Path) -> dict:
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    return {
        "name": directory.name,
        "directory": directory,
        "tensor_path": directory / "adapter_model.safetensors",
        "base_model": config.get("base_model_name_or_path"),
        "rank": float(config.get("r", 1)),
        "alpha": float(config.get("lora_alpha", 1)),
        "target_modules": sorted(config.get("target_modules") or []),
    }


def tensor_index(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with safe_open(path, framework="pt", device="cpu") as file:
        for key in file.keys():
            match = KEY_PATTERN.search(key)
            if match:
                result[(match.group(1), match.group(2))] = key
    return result


def compare_pair(left: dict, right: dict) -> dict:
    left_index = tensor_index(left["tensor_path"])
    right_index = tensor_index(right["tensor_path"])
    left_modules = {module for module, kind in left_index if kind == "A"}
    right_modules = {module for module, kind in right_index if kind == "A"}
    shared_modules = sorted(left_modules & right_modules)

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    left_scale = left["alpha"] / left["rank"]
    right_scale = right["alpha"] / right["rank"]

    with safe_open(left["tensor_path"], framework="pt", device="cpu") as left_file, safe_open(
        right["tensor_path"], framework="pt", device="cpu"
    ) as right_file:
        for module in shared_modules:
            a1 = left_file.get_tensor(left_index[(module, "A")]).float()
            b1 = left_file.get_tensor(left_index[(module, "B")]).float()
            a2 = right_file.get_tensor(right_index[(module, "A")]).float()
            b2 = right_file.get_tensor(right_index[(module, "B")]).float()

            # <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)).
            # This avoids allocating the large dense updates B @ A.
            dot += float(torch.trace((b1.T @ b2) @ (a2 @ a1.T))) * left_scale * right_scale
            left_norm += float(torch.trace((b1.T @ b1) @ (a1 @ a1.T))) * left_scale**2
            right_norm += float(torch.trace((b2.T @ b2) @ (a2 @ a2.T))) * right_scale**2

    cosine = dot / math.sqrt(max(left_norm * right_norm, 1e-30))
    return {
        "left": left["name"],
        "right": right["name"],
        "same_base_model": int(left["base_model"] == right["base_model"]),
        "same_rank": int(left["rank"] == right["rank"]),
        "same_target_modules": int(left["target_modules"] == right["target_modules"]),
        "shared_modules": len(shared_modules),
        "left_modules": len(left_modules),
        "right_modules": len(right_modules),
        "effective_update_cosine": cosine,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path(r"D:\ecnu_experiment\Model\LoRA\Qwen2.5_1.5B"),
    )
    parser.add_argument("--output", type=Path, default=Path("lora_weight_similarity.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapters = []
    for directory in sorted(path for path in args.root.rglob("*") if path.is_dir()):
        if (directory / "adapter_config.json").exists() and (
            directory / "adapter_model.safetensors"
        ).exists():
            adapters.append(load_metadata(directory))

    rows = []
    for left_index, left in enumerate(adapters):
        for right in adapters[left_index + 1 :]:
            row = compare_pair(left, right)
            rows.append(row)
            print(
                f"{row['left']} vs {row['right']}: "
                f"cos={row['effective_update_cosine']:.6f}, "
                f"shared={row['shared_modules']}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
