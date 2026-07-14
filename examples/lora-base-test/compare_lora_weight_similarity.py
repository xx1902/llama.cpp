#!/usr/bin/env python3
"""Compare effective LoRA updates from GGUF or PEFT safetensors adapters.

For every shared LoRA module, the script compares the effective update:

    delta_W = (alpha / rank) * B @ A

It computes the Frobenius cosine without materializing the large B @ A matrix.
GGUF input is preferred when one or more .gguf files exist under the root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


PEFT_KEY_PATTERN = re.compile(
    r"(model\.layers\.\d+\.(?:mlp|self_attn)\.[^.]+)"
    r"\.lora_([AB])\.weight$"
)


def safe_field(reader, key: str, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        return default


def load_gguf_metadata(path: Path) -> dict:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    tensor_names = {tensor.name for tensor in reader.tensors}
    modules = sorted(
        name[: -len(".lora_a")]
        for name in tensor_names
        if name.endswith(".lora_a")
        and name[: -len(".lora_a")] + ".lora_b" in tensor_names
    )

    rank = 0
    if modules:
        tensor_map = {tensor.name: tensor for tensor in reader.tensors}
        rank = int(np.asarray(tensor_map[modules[0] + ".lora_a"].data).shape[0])

    return {
        "name": path.stem,
        "format": "gguf",
        "path": path,
        "base_model": str(safe_field(reader, "general.base_model.0.name", "unknown")),
        "adapter_name": str(safe_field(reader, "general.name", path.stem)),
        "alpha": float(safe_field(reader, "adapter.lora.alpha", rank or 1)),
        "rank": float(rank or 1),
        "modules": modules,
        "target_modules": sorted({module.split(".")[-2] for module in modules}),
        "tensor_count": len(reader.tensors),
        "size_bytes": path.stat().st_size,
    }


def load_peft_metadata(directory: Path) -> dict:
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    tensor_path = directory / "adapter_model.safetensors"

    from safetensors import safe_open

    modules: list[str] = []
    with safe_open(tensor_path, framework="pt", device="cpu") as file:
        for key in file.keys():
            match = PEFT_KEY_PATTERN.search(key)
            if match and match.group(2) == "A":
                modules.append(match.group(1))

    return {
        "name": directory.name,
        "format": "peft",
        "path": tensor_path,
        "base_model": str(config.get("base_model_name_or_path", "unknown")),
        "adapter_name": directory.name,
        "alpha": float(config.get("lora_alpha", 1)),
        "rank": float(config.get("r", 1)),
        "modules": sorted(modules),
        "target_modules": sorted(config.get("target_modules") or []),
        "tensor_count": len(modules) * 2,
        "size_bytes": tensor_path.stat().st_size,
    }


def gguf_tensor_index(path: Path) -> dict[tuple[str, str], np.ndarray]:
    from gguf import GGUFReader

    result: dict[tuple[str, str], np.ndarray] = {}
    reader = GGUFReader(str(path))
    for tensor in reader.tensors:
        if tensor.name.endswith(".lora_a"):
            result[(tensor.name[: -len(".lora_a")], "A")] = np.asarray(tensor.data)
        elif tensor.name.endswith(".lora_b"):
            result[(tensor.name[: -len(".lora_b")], "B")] = np.asarray(tensor.data)
    return result


def peft_tensor_index(path: Path) -> dict[tuple[str, str], np.ndarray]:
    from safetensors import safe_open

    result: dict[tuple[str, str], np.ndarray] = {}
    with safe_open(path, framework="np") as file:
        for key in file.keys():
            match = PEFT_KEY_PATTERN.search(key)
            if match:
                result[(match.group(1), match.group(2))] = file.get_tensor(key)
    return result


def tensor_index(adapter: dict) -> dict[tuple[str, str], np.ndarray]:
    if adapter["format"] == "gguf":
        return gguf_tensor_index(adapter["path"])
    return peft_tensor_index(adapter["path"])


def effective_update_statistics(
    left_a: np.ndarray,
    left_b: np.ndarray,
    right_a: np.ndarray,
    right_b: np.ndarray,
) -> tuple[float, float, float]:
    # GGUFReader returns read-only memory maps; copy before handing arrays to
    # torch so the comparison does not emit a non-writable-array warning.
    a1 = torch.from_numpy(np.array(left_a, dtype=np.float32, copy=True))
    b1 = torch.from_numpy(np.array(left_b, dtype=np.float32, copy=True))
    a2 = torch.from_numpy(np.array(right_a, dtype=np.float32, copy=True))
    b2 = torch.from_numpy(np.array(right_b, dtype=np.float32, copy=True))

    # <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)).
    dot = float(torch.trace((b1.T @ b2) @ (a2 @ a1.T)))
    left_norm = float(torch.trace((b1.T @ b1) @ (a1 @ a1.T)))
    right_norm = float(torch.trace((b2.T @ b2) @ (a2 @ a2.T)))
    return dot, left_norm, right_norm


def compare_pair(left: dict, right: dict) -> dict:
    left_index = tensor_index(left)
    right_index = tensor_index(right)
    left_modules = {module for module, kind in left_index if kind == "A"}
    right_modules = {module for module, kind in right_index if kind == "A"}
    shared_modules = sorted(left_modules & right_modules)

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    left_scale = left["alpha"] / left["rank"]
    right_scale = right["alpha"] / right["rank"]

    for module in shared_modules:
        module_dot, module_left_norm, module_right_norm = effective_update_statistics(
            left_index[(module, "A")],
            left_index[(module, "B")],
            right_index[(module, "A")],
            right_index[(module, "B")],
        )
        dot += module_dot * left_scale * right_scale
        left_norm += module_left_norm * left_scale**2
        right_norm += module_right_norm * right_scale**2

    cosine = dot / math.sqrt(max(left_norm * right_norm, 1e-30))
    left_coverage = len(shared_modules) / max(1, len(left_modules))
    right_coverage = len(shared_modules) / max(1, len(right_modules))

    return {
        "left": left["name"],
        "right": right["name"],
        "left_format": left["format"],
        "right_format": right["format"],
        "same_base_model": int(left["base_model"] == right["base_model"]),
        "same_rank": int(left["rank"] == right["rank"]),
        "same_target_modules": int(left["target_modules"] == right["target_modules"]),
        "left_rank": int(left["rank"]),
        "right_rank": int(right["rank"]),
        "left_alpha": left["alpha"],
        "right_alpha": right["alpha"],
        "shared_modules": len(shared_modules),
        "left_modules": len(left_modules),
        "right_modules": len(right_modules),
        "left_coverage": left_coverage,
        "right_coverage": right_coverage,
        "effective_update_cosine": cosine,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(rows)


def write_metadata(root: Path, adapters: list[dict]) -> Path:
    rows = []
    for adapter in adapters:
        rows.append(
            {
                "name": adapter["name"],
                "format": adapter["format"],
                "adapter_name": adapter["adapter_name"],
                "base_model": adapter["base_model"],
                "rank": int(adapter["rank"]),
                "alpha": adapter["alpha"],
                "module_count": len(adapter["modules"]),
                "target_modules": "|".join(adapter["target_modules"]),
                "tensor_count": adapter["tensor_count"],
                "size_mb": adapter["size_bytes"] / 1024**2,
                "path": str(adapter["path"]),
            }
        )
    path = root / "lora_metadata.csv"
    write_csv(path, rows)
    return path


def write_matrix(root: Path, adapters: list[dict], rows: list[dict]) -> Path:
    names = [adapter["name"] for adapter in adapters]
    values = {(row["left"], row["right"]): row["effective_update_cosine"] for row in rows}
    matrix_rows = []
    for left in names:
        row = {"lora": left}
        for right in names:
            if left == right:
                row[right] = 1.0
            else:
                row[right] = values.get((left, right), values.get((right, left), 0.0))
        matrix_rows.append(row)
    path = root / "lora_weight_similarity_matrix.csv"
    write_csv(path, matrix_rows, ["lora", *names])
    return path


def write_ranked_pairs(root: Path, rows: list[dict]) -> Path:
    ranked = sorted(rows, key=lambda row: row["effective_update_cosine"], reverse=True)
    payload = {
        "note": (
            "Weight cosine is a screening signal only. Final grouping must use "
            "same-prefix KV cosine/L2 from llama.cpp."
        ),
        "top_pairs": ranked,
    }
    path = root / "lora_weight_similarity_ranked.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def discover_adapters(root: Path) -> list[dict]:
    gguf_paths = sorted(root.rglob("*.gguf"))
    if gguf_paths:
        return [load_gguf_metadata(path) for path in gguf_paths]

    adapters = []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        if (directory / "adapter_config.json").exists() and (
            directory / "adapter_model.safetensors"
        ).exists():
            adapters.append(load_peft_metadata(directory))
    return adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path(r"D:\ecnu_experiment\gguf-qwen2.5_1.5B"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Pairwise CSV path. Defaults to ROOT/lora_weight_similarity_pairs.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metadata, matrix, and ranked JSON. Defaults to ROOT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapters = discover_adapters(args.root)
    if len(adapters) < 2:
        raise RuntimeError(f"need at least two adapters under: {args.root}")

    print(f"found {len(adapters)} adapters under: {args.root}")
    for adapter in adapters:
        print(
            f"  {adapter['name']}: format={adapter['format']} "
            f"r={int(adapter['rank'])} alpha={adapter['alpha']} "
            f"modules={len(adapter['modules'])}"
        )

    rows = []
    for left_index, left in enumerate(adapters):
        for right in adapters[left_index + 1 :]:
            row = compare_pair(left, right)
            rows.append(row)
            print(
                f"{row['left']} vs {row['right']}: "
                f"cos={row['effective_update_cosine']:.6f}, "
                f"shared={row['shared_modules']}, "
                f"same_rank={row['same_rank']}"
            )

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output or (output_dir / "lora_weight_similarity_pairs.csv")
    write_csv(pair_path, sorted(rows, key=lambda row: row["effective_update_cosine"], reverse=True))
    metadata_path = write_metadata(output_dir, adapters)
    matrix_path = write_matrix(output_dir, adapters, rows)
    ranked_path = write_ranked_pairs(output_dir, rows)

    print("\noutputs:")
    print(f"  {pair_path}")
    print(f"  {metadata_path}")
    print(f"  {matrix_path}")
    print(f"  {ranked_path}")


if __name__ == "__main__":
    main()
