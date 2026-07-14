#!/usr/bin/env python3
"""Convert downloaded PEFT LoRA directories to llama.cpp GGUF adapters."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\ecnu_experiment\Model\LoRA\Qwen2.5_1.5B_similar"),
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(
            r"D:\ecnu_experiment\LLama.cpp\llama.cpp\convert_lora_to_gguf.py"
        ),
    )
    parser.add_argument(
        "--python",
        default=r"D:\anaconda\envs\qwen2.5_vl\python.exe",
    )
    parser.add_argument("--outtype", choices=["f16", "bf16", "f32"], default="f16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = args.root / "_base_qwen25_1.5b_instruct_config"
    output_dir = args.root / "gguf"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (base_config / "config.json").exists():
        raise FileNotFoundError(
            f"missing base config: {base_config}; run the download script first"
        )

    adapter_dirs = sorted(
        directory
        for directory in args.root.rglob("*")
        if directory.is_dir()
        and (directory / "adapter_config.json").exists()
        and (
            (directory / "adapter_model.safetensors").exists()
            or (directory / "adapter_model.bin").exists()
        )
        and directory != base_config
    )

    for index, adapter_dir in enumerate(adapter_dirs, 1):
        relative_name = "__".join(adapter_dir.relative_to(args.root).parts)
        output_path = output_dir / f"{relative_name}-{args.outtype}.gguf"
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
        print(f"\n[{index}/{len(adapter_dirs)}] {adapter_dir.name}")
        print(" ".join(command))
        subprocess.run(command, check=True)

    print(f"\nconverted {len(adapter_dirs)} adapters to: {output_dir}")


if __name__ == "__main__":
    main()
