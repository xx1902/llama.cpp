#!/usr/bin/env python3
"""Build a verified set of 87 Qwen2.5-1.5B-Instruct LoRA adapters.

The script treats one physical adapter as one LoRA ID. Twelve adapters that
already exist as GGUF files are registered first; compatible PEFT adapters are
then downloaded until the manifest contains exactly 87 unique LoRAs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
OFFICIAL_ENDPOINT = "https://huggingface.co"
DEFAULT_TARGET_TOTAL = 87

DEFAULT_SAVE_ROOT = Path(
    r"D:\ecnu_experiment\Model\LoRA\Qwen2.5_1.5B_87"
)
DEFAULT_GGUF_ROOT = Path(r"D:\ecnu_experiment\gguf-qwen2.5_1.5B")

# These 12 adapters are already present locally and must keep stable IDs.
EXISTING_ADAPTERS: list[dict[str, str]] = [
    {
        "repo_id": "shibing624/chinese-text-correction-1.5b-lora",
        "group": "writing",
        "gguf_name": "chinese-correction.gguf",
    },
    {
        "repo_id": "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16",
        "group": "code",
        "gguf_name": "code-r16.gguf",
    },
    {
        "repo_id": "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v2",
        "group": "code",
        "gguf_name": "code-r16v2.gguf",
    },
    {
        "repo_id": "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v3",
        "group": "code",
        "gguf_name": "code-r16v3.gguf",
    },
    {
        "repo_id": "monteri/qwen_song_lyrics_model",
        "group": "creative_writing",
        "gguf_name": "song-lyrics.gguf",
    },
    {
        "repo_id": "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-optimal-1k",
        "group": "reasoning",
        "gguf_name": "countdown-optimal.gguf",
    },
    {
        "repo_id": "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-react-1k",
        "group": "reasoning",
        "gguf_name": "countdown-search-react.gguf",
    },
    {
        "repo_id": "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-1k",
        "group": "reasoning",
        "gguf_name": "countdown-search.gguf",
    },
    {
        "repo_id": "ducnd58233/qwen2.5-1.5b-qlora-summarization",
        "group": "summarization",
        "gguf_name": "summarization_qlora.gguf",
    },
    {
        "repo_id": "phuongntc/qwen2.5-1.5b-sft-summarization-qlora",
        "group": "summarization",
        "gguf_name": "summarization_sft.gguf",
    },
    {
        "repo_id": "harpomaxx/qwen2.5-1.5b-slips-alert-summary-merged-prompt-8192",
        "group": "summarization",
        "gguf_name": "summary-merged-prompt.gguf",
    },
    {
        "repo_id": "harpomaxx/qwen2.5-1.5b-slips-alert-summary-sep-prompt-8192",
        "group": "summarization",
        "gguf_name": "summary-sep-prompt.gguf",
    },
]


def preferred_repositories() -> list[str]:
    """Return reproducible candidates before using broad Hub search."""
    repos = [
        # Controlled reasoning family.
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-react-5k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-search-5k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-optimal-5k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-sos-1k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-sos_react-1k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-deepseek-1k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-deepseek-1kx2",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-deepseek-5k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-deepseek-6k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-deepseek-correct-5k",
        "yeok/qwen-2.5-1.5B-instruct-sft-lora-countdown-mixed-10k",
        # Code-related adapters trained on the same non-Coder base model.
        "ivnle/qwen2.5-1.5b-instruct_codex-whole-v3_lora_r32-a128_sft_mts2",
        "ivnle/qwen2.5-1.5b-instruct_codex-whole-v3_lora_r32-a128_sft",
        "ivnle/qwen2.5-1.5b-instruct_codex-line-50_lora_r32-a128_sft",
        "ivnle/qwen2.5-1.5b-instruct_codex-intervals-20_lora_r32-a128_sft",
        # Cross-task controls.
        "Speeeed/Qwen2.5-1.5B-Instruct-mnli-lora",
        "zjudai/flowertune-general-nlp-lora-qwen2.5-1.5b-instruct",
        "benjaminzwhite/Qwen2.5-1.5B-Instruct_NEUDM-Senti_GRPO-64-steps_LoRA-adapters",
        "zeroconverger/dpo-Qwen2.5-1.5B-Instruct-lora",
        "lethxlity/Qwen2.5-1.5B-Instruct-RU-Jokes-Lora",
        "yusufcelebi/qwen2.5-1.5b-instruct-orchamat-sparse-lora",
        "yusufcelebi/qwen2.5-1.5b-instruct-orchamat-lora",
        "renansantosmendes/synapseai-qwen2.5-1.5B-instruct-lora-v1",
        "DEAR-Tao/Qwen2.5-1.5B-Instruct-SFT-lora-adapter",
        "nzxlu/qwen2.5-1.5b-instruct-clean-lora",
    ]

    # xw17 provides many controlled variants on mobile/personal sensing data.
    for mode in ("SFT", "COT"):
        for dataset in ("pmdata", "lifesnaps", "globem", "aw_fb", "usc-had", "wesad"):
            repos.append(f"xw17/Qwen2.5-1.5B-Instruct_{mode}_lora_{dataset}")

    for prefix in (
        "default",
        "optimal",
        "def",
        "def_lora2",
        "def_lora3",
        "def_lora4",
        "optimized",
        "optimized1",
        "optimized1_task_grouping_off",
    ):
        for index in range(1, 5):
            if prefix in {"default", "optimal"}:
                suffix = f"{index}_{prefix}_lora"
            elif prefix.startswith("def_lora"):
                number = prefix.removeprefix("def_lora")
                suffix = f"{index}_def_lora{number}"
            else:
                suffix = f"{index}_{prefix}_lora"
            repos.append(f"xw17/Qwen2.5-1.5B-Instruct_finetuned_{suffix}")

    repos.extend(
        [
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_1_def_lora_pmdata_aug",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_2_def_lora_pmdata_aug",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_3_def_lora_pmdata_aug",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_4_def_lora_pmdata_aug",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__pmdata_aug_lora",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_4_optimized_lora_activity_origin",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_4_optimized_lora_activity_updated",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__optimized_lora_globem_aug",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__optimized_lora_globem_origin",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__optimized1_lora_universal",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__optimized1_augmention_lora",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__optimized1_pmdata_augmentation_lora",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__optimized1_globem_augmentation_lora",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_1_optimized1_oversampling_lora",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned_3_optimized1_oversampling_lora",
            "xw17/Qwen2.5-1.5B-Instruct_finetuned__optimized1_universal_without_task_grouping_lora",
            "xw17/Qwen2.5-1.5B-Instruct_SFT_lora_universal",
        ]
    )
    return unique(repos)


SEARCH_QUERIES = [
    "Qwen2.5-1.5B-Instruct LoRA",
    "Qwen2.5-1.5B-Instruct SFT LoRA",
    "Qwen2.5-1.5B-Instruct COT LoRA",
    "Qwen2.5-1.5B-Instruct classification LoRA",
    "Qwen2.5-1.5B-Instruct summarization LoRA",
    "Qwen2.5-1.5B-Instruct writing LoRA",
    "Qwen2.5-1.5B-Instruct math LoRA",
    "Qwen2.5-1.5B-Instruct medical LoRA",
]

ALLOWED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def unique(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def repo_slug(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repo_id)


def infer_group(repo_id: str) -> str:
    name = repo_id.lower()
    rules = [
        ("mobile_context", ("pmdata", "lifesnaps", "globem", "aw_fb", "usc-had", "wesad")),
        ("reasoning", ("countdown", "math", "gsm8k", "reason")),
        ("code", ("code", "codex", "cpp", "sql")),
        ("summarization", ("summar", "summary")),
        ("classification", ("mnli", "class", "senti")),
        ("writing", ("writing", "joke", "lyrics", "correction")),
    ]
    for group, keywords in rules:
        if any(keyword in name for keyword in keywords):
            return group
    return "general"


def is_qwen25_15b_instruct(base_model: Any) -> bool:
    if not isinstance(base_model, str):
        return False
    normalized = base_model.replace("_", "-").lower()
    if "coder" in normalized or "math" in normalized:
        return False
    return "qwen2.5-1.5b-instruct" in normalized


class HubClient:
    def __init__(self, endpoints: list[str]) -> None:
        self.endpoints = unique(endpoint.rstrip("/") for endpoint in endpoints)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "qwen25-lora-87-builder/1.0"

    def get_json(self, path: str) -> tuple[dict | list, str]:
        last_error: Exception | None = None
        for attempt in range(3):
            for endpoint in self.endpoints:
                try:
                    response = self.session.get(
                        endpoint + path,
                        timeout=(30, 120),
                    )
                    response.raise_for_status()
                    return response.json(), endpoint
                except (requests.RequestException, ValueError) as error:
                    last_error = error
                    print(
                        f"  metadata attempt {attempt + 1}/3 failed on "
                        f"{endpoint}: {error}"
                    )
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"all endpoints failed for {path}: {last_error}")

    def inspect_repo(self, repo_id: str) -> dict[str, Any]:
        info, endpoint = self.get_json(f"/api/models/{repo_id}")
        assert isinstance(info, dict)
        files = sorted(item.get("rfilename", "") for item in info.get("siblings", []))
        has_weight = any(
            name in {"adapter_model.safetensors", "adapter_model.bin"}
            for name in files
        )
        result: dict[str, Any] = {
            "repo_id": repo_id,
            "endpoint": endpoint,
            "downloads": int(info.get("downloads") or 0),
            "likes": int(info.get("likes") or 0),
            "files": files,
            "has_adapter_config": "adapter_config.json" in files,
            "has_adapter_model": has_weight,
            "compatible": False,
            "reject_reason": "",
        }
        if not result["has_adapter_config"] or not has_weight:
            result["reject_reason"] = "missing root adapter config or weight"
            return result

        config, _ = self.get_json(f"/{repo_id}/resolve/main/adapter_config.json")
        assert isinstance(config, dict)
        target_modules = sorted(config.get("target_modules") or [])
        result.update(
            {
                "base_model": config.get("base_model_name_or_path"),
                "peft_type": config.get("peft_type"),
                "task_type": config.get("task_type"),
                "bias": config.get("bias", "none"),
                "rank": config.get("r"),
                "alpha": config.get("lora_alpha"),
                "target_modules": target_modules,
                "modules_to_save": config.get("modules_to_save"),
            }
        )

        reasons: list[str] = []
        if str(result.get("peft_type") or "").upper() != "LORA":
            reasons.append("peft_type is not LORA")
        if str(result.get("task_type") or "").upper() != "CAUSAL_LM":
            reasons.append("task_type is not CAUSAL_LM")
        if not is_qwen25_15b_instruct(result.get("base_model")):
            reasons.append("base model is not Qwen2.5-1.5B-Instruct")
        if result.get("modules_to_save"):
            reasons.append("contains unsupported modules_to_save")
        if str(result.get("bias") or "none").lower() != "none":
            reasons.append(f"unsupported trained bias: {result.get('bias')}")
        unsupported = set(target_modules) - ALLOWED_TARGET_MODULES
        if unsupported:
            reasons.append(f"unsupported targets: {sorted(unsupported)}")

        result["reject_reason"] = "; ".join(reasons)
        result["compatible"] = not reasons
        return result

    def search(self, query: str, limit: int) -> list[str]:
        params = requests.models.PreparedRequest()
        params.prepare_url("https://placeholder.invalid", {"search": query, "limit": limit})
        query_string = params.url.split("?", 1)[1]
        data, _ = self.get_json(f"/api/models?{query_string}")
        assert isinstance(data, list)
        return [item["id"] for item in data if item.get("id")]


def build_existing_entries(gguf_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for lora_id, adapter in enumerate(EXISTING_ADAPTERS):
        gguf_path = gguf_root / adapter["gguf_name"]
        if not gguf_path.is_file():
            raise FileNotFoundError(f"existing GGUF is missing: {gguf_path}")
        entries.append(
            {
                "lora_id": lora_id,
                "app_slot": lora_id,
                "repo_id": adapter["repo_id"],
                "group": adapter["group"],
                "source": "existing",
                "status": "converted",
                "adapter_dir": "",
                "gguf_path": str(gguf_path),
                "gguf_size_mb": round(gguf_path.stat().st_size / 1024**2, 6),
            }
        )
    return entries


def local_adapter_valid(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_config.json").is_file() and (
        (adapter_dir / "adapter_model.safetensors").is_file()
        or (adapter_dir / "adapter_model.bin").is_file()
    )


def inspect_local_adapter(adapter_dir: Path, repo_id: str) -> dict[str, Any]:
    """Validate a downloaded adapter without relying on the Hub metadata API."""
    if not local_adapter_valid(adapter_dir):
        return {
            "repo_id": repo_id,
            "compatible": False,
            "reject_reason": f"local adapter files are incomplete: {adapter_dir}",
        }
    config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    target_modules = sorted(config.get("target_modules") or [])
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "base_model": config.get("base_model_name_or_path"),
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "bias": config.get("bias", "none"),
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "target_modules": target_modules,
        "modules_to_save": config.get("modules_to_save"),
        "compatible": False,
        "reject_reason": "",
    }
    reasons: list[str] = []
    if str(result.get("peft_type") or "").upper() != "LORA":
        reasons.append("peft_type is not LORA")
    if str(result.get("task_type") or "").upper() != "CAUSAL_LM":
        reasons.append("task_type is not CAUSAL_LM")
    if not is_qwen25_15b_instruct(result.get("base_model")):
        reasons.append("base model is not Qwen2.5-1.5B-Instruct")
    if result.get("modules_to_save"):
        reasons.append("contains unsupported modules_to_save")
    if str(result.get("bias") or "none").lower() != "none":
        reasons.append(f"unsupported trained bias: {result.get('bias')}")
    unsupported = set(target_modules) - ALLOWED_TARGET_MODULES
    if unsupported:
        reasons.append(f"unsupported targets: {sorted(unsupported)}")
    result["reject_reason"] = "; ".join(reasons)
    result["compatible"] = not reasons
    return result


def load_resumable_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("loras", []) if isinstance(data, dict) else []
    resumed = [
        entry
        for entry in entries
        if entry.get("source") == "downloaded"
        and local_adapter_valid(Path(entry.get("adapter_dir", "")))
    ]
    resumed.sort(key=lambda item: int(item.get("lora_id", 10**9)))
    for offset, entry in enumerate(resumed, start=len(EXISTING_ADAPTERS)):
        entry["lora_id"] = offset
        entry["app_slot"] = offset
    return resumed


def write_manifest(
    manifest_path: Path,
    entries: list[dict[str, Any]],
    target_total: int,
) -> None:
    entries = sorted(entries, key=lambda item: int(item["lora_id"]))
    payload = {
        "schema_version": 1,
        "base_model": BASE_MODEL,
        "target_total": target_total,
        "actual_total": len(entries),
        "complete": len(entries) == target_total,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loras": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = manifest_path.with_suffix(".csv")
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
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)


def discover_candidates(
    client: HubClient,
    search_limit: int,
    excluded: set[str],
) -> list[str]:
    candidates = preferred_repositories()
    for query in SEARCH_QUERIES:
        try:
            candidates.extend(client.search(query, search_limit))
        except RuntimeError as error:
            print(f"warning: search failed for {query!r}: {error}")
    return [repo_id for repo_id in unique(candidates) if repo_id not in excluded]


def download_repo(repo_id: str, local_dir: Path, endpoints: list[str]) -> str:
    from huggingface_hub import snapshot_download

    allow_patterns = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "README.md",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    last_error: Exception | None = None
    for attempt in range(3):
        for endpoint in endpoints:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    allow_patterns=allow_patterns,
                    endpoint=endpoint,
                    max_workers=4,
                    etag_timeout=60,
                )
                if not local_adapter_valid(local_dir):
                    raise RuntimeError("download completed but adapter files are incomplete")
                return endpoint
            except Exception as error:  # Continue with the next endpoint/repository.
                last_error = error
                print(
                    f"  download attempt {attempt + 1}/3 failed on "
                    f"{endpoint}: {error}"
                )
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed on all endpoints: {last_error}")


def download_base_config(save_root: Path, endpoints: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    base_dir = save_root / "_base_qwen25_1.5b_instruct_config"
    if (base_dir / "config.json").is_file() and (base_dir / "tokenizer.json").is_file():
        return base_dir
    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
    ]
    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            snapshot_download(
                repo_id=BASE_MODEL,
                local_dir=str(base_dir),
                allow_patterns=patterns,
                endpoint=endpoint,
            )
            return base_dir
        except Exception as error:
            last_error = error
    raise RuntimeError(f"failed to download base config: {last_error}")


def run_plan(args: argparse.Namespace, client: HubClient) -> None:
    existing_ids = {item["repo_id"] for item in EXISTING_ADAPTERS}
    candidates = discover_candidates(client, args.search_limit, existing_ids)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    required = args.target_total - len(EXISTING_ADAPTERS)
    for index, repo_id in enumerate(candidates, 1):
        try:
            result = client.inspect_repo(repo_id)
        except RuntimeError as error:
            result = {"repo_id": repo_id, "compatible": False, "reject_reason": str(error)}
        (accepted if result.get("compatible") else rejected).append(result)
        state = "accept" if result.get("compatible") else "reject"
        print(f"[{index}/{len(candidates)}] {state}: {repo_id}")
        if len(accepted) >= required + args.spare_candidates:
            break

    output = {
        "required_new": required,
        "accepted_count": len(accepted),
        "accepted": accepted,
        "rejected": rejected,
    }
    args.plan_path.parent.mkdir(parents=True, exist_ok=True)
    args.plan_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidate plan saved to: {args.plan_path}")
    if len(accepted) < required:
        raise RuntimeError(f"only {len(accepted)} compatible candidates; need {required}")


def run_download(args: argparse.Namespace, client: HubClient) -> None:
    if args.target_total < len(EXISTING_ADAPTERS):
        raise ValueError("target total is smaller than the existing adapter count")

    endpoints = unique([args.endpoint, args.fallback_endpoint])
    existing = build_existing_entries(args.existing_gguf_root)
    resumed = load_resumable_entries(args.manifest)
    entries = existing + sorted(resumed, key=lambda item: int(item["lora_id"]))
    entries = entries[: args.target_total]
    used_repo_ids = {entry["repo_id"] for entry in entries}
    write_manifest(args.manifest, entries, args.target_total)

    download_base_config(args.save_root, endpoints)
    candidates = discover_candidates(client, args.search_limit, used_repo_ids)
    failures: list[dict[str, str]] = []

    for index, repo_id in enumerate(candidates, 1):
        if len(entries) >= args.target_total:
            break
        print(f"\n[{index}/{len(candidates)}] inspecting {repo_id}")
        try:
            metadata = client.inspect_repo(repo_id)
        except RuntimeError as error:
            failures.append({"repo_id": repo_id, "stage": "inspect", "error": str(error)})
            print(f"  rejected: {error}")
            continue
        if not metadata.get("compatible"):
            print(f"  rejected: {metadata.get('reject_reason')}")
            continue

        group = infer_group(repo_id)
        adapter_dir = args.save_root / "adapters" / group / repo_slug(repo_id)
        try:
            endpoint = download_repo(repo_id, adapter_dir, endpoints)
        except RuntimeError as error:
            failures.append({"repo_id": repo_id, "stage": "download", "error": str(error)})
            print(f"  failed: {error}")
            continue

        lora_id = len(entries)
        entry = {
            "lora_id": lora_id,
            "app_slot": lora_id,
            "repo_id": repo_id,
            "group": group,
            "source": "downloaded",
            "status": "downloaded",
            "adapter_dir": str(adapter_dir),
            "gguf_path": "",
            "gguf_size_mb": "",
            "base_model": metadata.get("base_model"),
            "rank": metadata.get("rank"),
            "alpha": metadata.get("alpha"),
            "target_modules": metadata.get("target_modules"),
            "downloads": metadata.get("downloads"),
            "download_endpoint": endpoint,
        }
        entries.append(entry)
        used_repo_ids.add(repo_id)
        write_manifest(args.manifest, entries, args.target_total)
        print(f"  accepted as lora_id={lora_id}; total={len(entries)}/{args.target_total}")

    failure_path = args.manifest.with_name("lora_87_failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(entries) != args.target_total:
        raise RuntimeError(
            f"download incomplete: {len(entries)}/{args.target_total}; "
            f"see {failure_path}"
        )
    print(f"\n87-LoRA download set is complete: {args.manifest}")


def run_verify(args: argparse.Namespace) -> None:
    existing = build_existing_entries(args.existing_gguf_root)
    resumed = load_resumable_entries(args.manifest)
    entries = existing + sorted(resumed, key=lambda item: int(item["lora_id"]))
    ids = [int(entry["lora_id"]) for entry in entries]
    repo_ids = [entry["repo_id"] for entry in entries]
    errors: list[str] = []
    if ids != list(range(len(entries))):
        errors.append("LoRA IDs are not contiguous from zero")
    if len(repo_ids) != len(set(repo_ids)):
        errors.append("duplicate repository IDs exist")
    if len(entries) != args.target_total:
        errors.append(f"count is {len(entries)}, expected {args.target_total}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(2)
    print(f"verified {len(entries)} unique physical LoRAs")


def run_replace(args: argparse.Namespace, client: HubClient) -> None:
    """Replace one failed manifest entry without changing its LoRA ID."""
    if args.replace_id is None or not args.replace_repo:
        raise ValueError("replace requires --replace-id and --replace-repo")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {args.manifest}")

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = data.get("loras", [])
    target = next(
        (entry for entry in entries if int(entry.get("lora_id", -1)) == args.replace_id),
        None,
    )
    if target is None:
        raise ValueError(f"lora_id={args.replace_id} is not present in the manifest")

    duplicate = next(
        (
            entry
            for entry in entries
            if entry.get("repo_id") == args.replace_repo
            and int(entry.get("lora_id", -1)) != args.replace_id
        ),
        None,
    )
    if duplicate is not None:
        raise ValueError(
            f"replacement repository is already lora_id={duplicate['lora_id']}"
        )

    print(f"replacing lora_id={args.replace_id}")
    print(f"  old: {target.get('repo_id')}")
    print(f"  new: {args.replace_repo}")
    endpoints = unique([args.endpoint, args.fallback_endpoint])
    group = infer_group(args.replace_repo)
    adapter_dir = args.save_root / "adapters" / group / repo_slug(args.replace_repo)
    remote_metadata: dict[str, Any] | None = None
    if args.skip_remote_inspect:
        print("skipping remote metadata inspection; validating downloaded files locally")
    else:
        try:
            remote_metadata = client.inspect_repo(args.replace_repo)
            if not remote_metadata.get("compatible"):
                raise RuntimeError(
                    f"replacement is incompatible: {remote_metadata.get('reject_reason')}"
                )
        except RuntimeError as error:
            print(f"warning: remote inspection failed; validating after download: {error}")

    endpoint = download_repo(args.replace_repo, adapter_dir, endpoints)
    metadata = inspect_local_adapter(adapter_dir, args.replace_repo)
    if not metadata.get("compatible"):
        raise RuntimeError(
            f"downloaded replacement is incompatible: {metadata.get('reject_reason')}"
        )
    if remote_metadata:
        metadata["downloads"] = remote_metadata.get("downloads")

    old_repo = target.get("repo_id", "")
    old_adapter_dir = target.get("adapter_dir", "")
    target.update(
        {
            "repo_id": args.replace_repo,
            "group": group,
            "source": "downloaded",
            "status": "downloaded",
            "adapter_dir": str(adapter_dir),
            "gguf_path": "",
            "gguf_size_mb": "",
            "base_model": metadata.get("base_model"),
            "rank": metadata.get("rank"),
            "alpha": metadata.get("alpha"),
            "target_modules": metadata.get("target_modules"),
            "downloads": metadata.get("downloads"),
            "download_endpoint": endpoint,
            "convert_error": "",
            "replaced_repo_id": old_repo,
            "replaced_adapter_dir": old_adapter_dir,
        }
    )
    write_manifest(args.manifest, entries, int(data.get("target_total") or 87))
    print(f"replacement downloaded and manifest updated: {args.manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "download", "verify", "replace"])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--fallback-endpoint", default=OFFICIAL_ENDPOINT)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--search-limit", type=int, default=200)
    parser.add_argument("--spare-candidates", type=int, default=10)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--existing-gguf-root", type=Path, default=DEFAULT_GGUF_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--replace-id", type=int)
    parser.add_argument("--replace-repo")
    parser.add_argument("--skip-remote-inspect", action="store_true")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.save_root / "lora_87_manifest.json"
    if args.plan_path is None:
        args.plan_path = args.save_root / "lora_87_candidate_plan.json"
    return args


def main() -> None:
    args = parse_args()
    endpoints = unique([args.endpoint, args.fallback_endpoint])
    client = HubClient(endpoints)
    if args.action == "plan":
        run_plan(args, client)
    elif args.action == "download":
        run_download(args, client)
    elif args.action == "verify":
        run_verify(args)
    else:
        run_replace(args, client)


if __name__ == "__main__":
    main()
