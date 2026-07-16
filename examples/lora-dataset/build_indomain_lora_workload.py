"""Build task-specific KV-delta workloads from local datasets.

Summary adapters use XSum articles, code adapters use programming requests
filtered from ShareGPT, and countdown adapters use locally generated puzzles
with known valid expressions. Tokens remain identical inside each LoRA pair.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import load_from_disk


SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
CODE_HINTS = (
    "python",
    "javascript",
    "java ",
    "c++",
    "function",
    "algorithm",
    "code",
    "program",
    "debug",
    "bug",
    "sql",
    "class ",
    "implement",
    "leetcode",
    "```",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\ecnu_experiment\datasets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\ecnu_experiment\datasets\mobilora_workloads_indomain"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).parent / "profiles" / "lora_indomain_experiment.json",
    )
    parser.add_argument("--contexts-per-group", type=int, default=50)
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars]
    boundary = max(shortened.rfind(". "), shortened.rfind("\n"))
    if boundary >= max_chars // 2:
        shortened = shortened[: boundary + 1]
    return shortened.strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimated_tokens(text: str) -> int:
    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4 + non_ascii_count / 1.5))


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def xsum_contexts(dataset_root: Path, count: int, max_chars: int, seed: int) -> list[dict]:
    dataset = load_from_disk(str(dataset_root / "xsum_dataset"))
    rows = []
    start = (seed * 9973) % len(dataset)
    for offset in range(len(dataset)):
        item = dataset[(start + offset) % len(dataset)]
        context = truncate_text(item["document"], max_chars)
        if len(context) < 200:
            continue
        rows.append(
            {
                "source_dataset": "xsum",
                "source_id": str(item["id"]),
                "context": context,
                "reference": clean_text(item["summary"]),
            }
        )
        if len(rows) >= count:
            break
    return rows


def first_user_message(conversations: object) -> str:
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", turn.get("role", ""))).lower()
        if role in {"human", "user"}:
            return clean_text(turn.get("value", turn.get("content", "")))
    return ""


def sharegpt_code_contexts(
    dataset_root: Path,
    count: int,
    max_chars: int,
    seed: int,
) -> list[dict]:
    dataset = load_from_disk(str(dataset_root / "sharegpt52k_dataset"))
    rows = []
    start = (seed * 7919) % len(dataset)
    for offset in range(len(dataset)):
        item = dataset[(start + offset) % len(dataset)]
        message = first_user_message(item.get("conversations"))
        normalized = message.lower()
        if len(message) < 80 or not any(hint in normalized for hint in CODE_HINTS):
            continue
        rows.append(
            {
                "source_dataset": "sharegpt_code",
                "source_id": str(item.get("id", len(rows))),
                "context": truncate_text(message, max_chars),
                "reference": "",
            }
        )
        if len(rows) >= count:
            break
    return rows


def countdown_contexts(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    operations = ["+", "-", "*"]
    while len(rows) < count:
        numbers = [rng.randint(2, 25) for _ in range(4)]
        op1, op2, op3 = [rng.choice(operations) for _ in range(3)]
        expression = f"(({numbers[0]} {op1} {numbers[1]}) {op2} {numbers[2]}) {op3} {numbers[3]}"
        try:
            target = int(eval(expression, {"__builtins__": {}}, {}))
        except Exception:
            continue
        key = (tuple(sorted(numbers)), target)
        if target <= 0 or target > 2000 or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "source_dataset": "countdown_synthetic",
                "source_id": f"countdown_{len(rows):06d}",
                "context": (
                    f"Use the numbers {', '.join(map(str, numbers))} exactly once "
                    f"to reach the target {target}. Allowed operations are +, -, * and /."
                ),
                "reference": expression,
            }
        )
    return rows


def make_prefix(context: str) -> str:
    return (
        "You are an on-device assistant.\n"
        "Context:\n"
        f"{context}\n\n"
        "Task:\n"
    )


def save_lora_config(output_dir: Path, profile: dict) -> None:
    config = {
        group_name: [
            {
                "lora_id": int(lora["lora_id"]),
                "logical_name": str(lora["name"]),
                "actual_adapter_path": str(lora["actual_adapter_path"]),
                "task": str(group["task"]),
                "is_anchor": bool(lora.get("is_anchor", False)),
            }
            for lora in group["loras"]
        ]
        for group_name, group in profile["groups"].items()
    }
    (output_dir / "lora_groups.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    xsum = xsum_contexts(
        args.dataset_root, args.contexts_per_group, args.max_context_chars, args.seed
    )
    code = sharegpt_code_contexts(
        args.dataset_root,
        args.contexts_per_group,
        args.max_context_chars,
        args.seed,
    )
    countdown = countdown_contexts(args.contexts_per_group, args.seed)
    pools = {
        "xsum": xsum,
        "sharegpt_code": code,
        "countdown_synthetic": countdown,
    }
    for source, pool in pools.items():
        if len(pool) < args.contexts_per_group:
            raise RuntimeError(
                f"Only {len(pool)} contexts were found for {source}; "
                f"requested {args.contexts_per_group}."
            )

    requests = []
    pairs = []
    request_id = 0
    for group_name, group in profile["groups"].items():
        contexts = pools[group["context_source"]]
        loras = group["loras"]
        anchors = [lora for lora in loras if lora.get("is_anchor", False)]
        if len(anchors) != 1:
            raise ValueError(f"Group {group_name} must contain exactly one anchor")
        anchor = anchors[0]

        for context_index, item in enumerate(contexts):
            context_id = f"{group_name}_{context_index:06d}"
            prefix = make_prefix(item["context"])
            prompt = prefix + group["task"]
            local_ids = {}
            for lora in loras:
                local_ids[int(lora["lora_id"])] = request_id
                requests.append(
                    {
                        "request_id": request_id,
                        "experiment": "cross_lora_delta_indomain",
                        "group_name": group_name,
                        "context_id": context_id,
                        "source_dataset": item["source_dataset"],
                        "source_id": item["source_id"],
                        "lora_id": int(lora["lora_id"]),
                        "lora_name": str(lora["name"]),
                        "is_anchor_lora": bool(lora.get("is_anchor", False)),
                        "task": str(group["task"]),
                        "common_prefix_text": prefix,
                        "common_prefix_hash": stable_hash(prefix),
                        "estimated_common_prefix_tokens": estimated_tokens(prefix),
                        "prompt": prompt,
                        "estimated_prompt_tokens": estimated_tokens(prompt),
                        "reference": item["reference"],
                    }
                )
                request_id += 1

            anchor_id = int(anchor["lora_id"])
            for child in loras:
                child_id = int(child["lora_id"])
                if child_id == anchor_id:
                    continue
                pairs.append(
                    {
                        "pair_name": (
                            f"{context_id}_{anchor['name']}_vs_{child['name']}"
                        ),
                        "group_name": group_name,
                        "context_id": context_id,
                        "common_prefix_hash": stable_hash(prefix),
                        "estimated_common_prefix_tokens": estimated_tokens(prefix),
                        "anchor_request_id": local_ids[anchor_id],
                        "child_request_id": local_ids[child_id],
                        "anchor_lora_id": anchor_id,
                        "child_lora_id": child_id,
                        "probe_type": "same_prefix_different_lora_indomain",
                    }
                )

    delta_dir = args.output_dir / "delta"
    grouped_dir = args.output_dir / "grouped"
    request_count = write_jsonl(delta_dir / "delta_requests.jsonl", requests)
    pair_count = write_jsonl(delta_dir / "delta_pairs.jsonl", pairs)
    write_jsonl(grouped_dir / "grouped_requests.jsonl", [])
    write_jsonl(grouped_dir / "app_state_trace.jsonl", [])
    save_lora_config(args.output_dir, profile)

    summary = {
        "profile_name": profile["name"],
        "contexts_per_group": args.contexts_per_group,
        "xsum_contexts": len(xsum),
        "sharegpt_code_contexts": len(code),
        "countdown_contexts": len(countdown),
        "delta_requests": request_count,
        "delta_pairs": pair_count,
        "grouped_requests": 0,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
