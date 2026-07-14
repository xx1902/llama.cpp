#!/usr/bin/env python3
"""Search, validate, and download Qwen2.5-1.5B PEFT LoRA adapters."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from pathlib import Path

import requests


DEFAULT_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# These groups were checked on 2026-07-14: each repository contains
# adapter_config.json and adapter_model.safetensors.
RECOMMENDED_GROUPS = {
    "reasoning_countdown": [
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-1k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-react-1k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-optimal-1k",
    ],
    "summarization_prompt_pair": [
        "harpomaxx/qwen2.5-1.5b-slips-alert-summary-sep-prompt-8192",
        "harpomaxx/qwen2.5-1.5b-slips-alert-summary-merged-prompt-8192",
    ],
    "summarization_mixed_controls": [
        "ducnd58233/qwen2.5-1.5b-qlora-summarization",
        "phuongntc/qwen2.5-1.5b-sft-summarization-qlora",
    ],
    "mixed_general_controls": [
        "zjudai/flowertune-general-nlp-lora-qwen2.5-1.5b-instruct",
        "ligaments-dev/gst-chatbot-qwen2.5-1.5b-lora",
        "usman-isb/Qwen2.5-1.5B-Instruct-lora-ft",
    ],
}

SEARCH_QUERIES = [
    "Qwen2.5-1.5B-Instruct LoRA",
    "Qwen2.5-1.5B summarization LoRA",
    "Qwen2.5-1.5B writing LoRA",
    "Qwen2.5-1.5B chat LoRA",
    "Qwen2.5-1.5B code LoRA",
]


def api_get(session: requests.Session, endpoint: str, path: str):
    response = session.get(endpoint.rstrip("/") + path, timeout=30)
    response.raise_for_status()
    return response.json()


def inspect_repo(session: requests.Session, endpoint: str, repo_id: str) -> dict:
    info = api_get(session, endpoint, f"/api/models/{repo_id}")
    files = [item.get("rfilename", "") for item in info.get("siblings", [])]
    result = {
        "repo_id": repo_id,
        "downloads": info.get("downloads", 0),
        "likes": info.get("likes", 0),
        "has_adapter_config": "adapter_config.json" in files,
        "has_adapter_model": any(
            name in {"adapter_model.safetensors", "adapter_model.bin"}
            for name in files
        ),
        "files": files,
    }

    if not result["has_adapter_config"]:
        return result

    config_url = (
        endpoint.rstrip("/")
        + f"/{repo_id}/resolve/main/adapter_config.json"
    )
    response = session.get(config_url, timeout=30)
    response.raise_for_status()
    config = response.json()
    result.update(
        {
            "base_model": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
            "rank": config.get("r"),
            "alpha": config.get("lora_alpha"),
            "dropout": config.get("lora_dropout"),
            "task_type": config.get("task_type"),
            "target_modules": sorted(config.get("target_modules") or []),
        }
    )
    result["compatible"] = (
        result["has_adapter_model"]
        and result.get("peft_type") == "LORA"
        and result.get("base_model") == BASE_MODEL
    )
    return result


def search_repositories(endpoint: str, output_path: Path, limit: int) -> None:
    session = requests.Session()
    candidate_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        encoded = urllib.parse.quote(query)
        path = f"/api/models?search={encoded}&limit={limit}&sort=downloads&direction=-1"
        try:
            for item in api_get(session, endpoint, path):
                candidate_ids.add(item["id"])
        except requests.RequestException as error:
            print(f"warning: search failed for {query}: {error}")

    results: list[dict] = []
    for index, repo_id in enumerate(sorted(candidate_ids), 1):
        try:
            result = inspect_repo(session, endpoint, repo_id)
        except (requests.RequestException, ValueError) as error:
            print(f"[{index}/{len(candidate_ids)}] skip {repo_id}: {error}")
            continue

        if result.get("compatible"):
            results.append(result)
            print(
                f"[{index}/{len(candidate_ids)}] compatible: {repo_id} "
                f"r={result.get('rank')} alpha={result.get('alpha')} "
                f"modules={len(result.get('target_modules', []))}"
            )

    results.sort(key=lambda item: item.get("downloads", 0), reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(results)} compatible adapters to: {output_path}")


def download_group(endpoint: str, group_name: str, save_root: Path) -> None:
    if group_name == "all":
        repos = [
            repo
            for group in RECOMMENDED_GROUPS.values()
            for repo in group
        ]
    else:
        repos = RECOMMENDED_GROUPS[group_name]

    # huggingface_hub reads HF_ENDPOINT when creating download URLs.
    os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download

    base_config_dir = save_root / "_base_qwen25_1.5b_instruct_config"
    snapshot_download(
        repo_id=BASE_MODEL,
        local_dir=str(base_config_dir),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
            "chat_template.jinja",
        ],
    )
    print(f"base config for GGUF conversion: {base_config_dir}")

    allow_patterns = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "README.md",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]

    for index, repo_id in enumerate(repos, 1):
        model_name = repo_id.split("/")[-1]
        local_dir = save_root / group_name / model_name
        print(f"\n[{index}/{len(repos)}] downloading {repo_id}")
        print(f"target: {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["search", "download", "list"])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("compatible_qwen25_loras.json"),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--group",
        choices=[*RECOMMENDED_GROUPS, "all"],
        default="reasoning_countdown",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(r"D:\ecnu_experiment\Model\LoRA\Qwen2.5_1.5B_similar"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "list":
        print(json.dumps(RECOMMENDED_GROUPS, ensure_ascii=False, indent=2))
    elif args.action == "search":
        search_repositories(args.endpoint, args.output, args.limit)
    else:
        download_group(args.endpoint, args.group, args.save_root)


if __name__ == "__main__":
    main()
