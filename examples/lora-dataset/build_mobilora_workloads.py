#!/usr/bin/env python3
"""Build MobiLoRA-style delta and grouped request workloads.

The script creates two complementary experiments:

1. delta/: exact shared token prefixes processed by different logical LoRAs.
2. grouped/: LSApp-driven arrival traces where only LoRAs from the routed
   context group are considered as cache-reuse candidates.

Logical LoRA IDs are intentionally separated from actual adapter file paths.
Replace the generated mapping later without rebuilding the text workload.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from datasets import load_from_disk


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class LogicalLora:
    lora_id: int
    name: str
    task: str
    is_anchor: bool = False


GROUPS: dict[str, list[LogicalLora]] = {
    "writing": [
        LogicalLora(0, "writing_summary_anchor", "Summarize the context.", True),
        LogicalLora(1, "writing_rewrite", "Rewrite the context clearly."),
        LogicalLora(2, "writing_keypoints", "Extract the key points from the context."),
    ],
    "conversation": [
        LogicalLora(3, "conversation_reply_anchor", "Write a helpful reply.", True),
        LogicalLora(4, "conversation_summary", "Summarize the conversation."),
        LogicalLora(5, "conversation_intent", "Extract the user's main intent."),
    ],
    "information": [
        LogicalLora(6, "information_summary_anchor", "Summarize the information.", True),
        LogicalLora(7, "information_facts", "Extract the important facts."),
        LogicalLora(8, "information_queries", "Generate useful search queries.", False),
    ],
}


WRITING_APP_KEYWORDS = {
    "gmail",
    "email",
    "mail",
    "office",
    "word",
    "docs",
    "notes",
    "keep",
}

CONVERSATION_APP_KEYWORDS = {
    "messenger",
    "messages",
    "messaging",
    "whatsapp",
    "telegram",
    "phone",
    "contacts",
}

INFORMATION_APP_KEYWORDS = {
    "google",
    "chrome",
    "browser",
    "internet",
    "maps",
    "youtube",
    "twitter",
    "facebook",
    "instagram",
    "play store",
}


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimated_tokens(text: str) -> int:
    """Coarse estimate only; llama.cpp should record exact token counts later."""
    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4 + non_ascii_count / 1.5))


def truncate_context(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    boundary = max(truncated.rfind(". "), truncated.rfind("\n"))
    if boundary >= max_chars // 2:
        truncated = truncated[: boundary + 1]
    return truncated.strip()


def make_prefix(group_name: str, context: str) -> str:
    # The task is deliberately placed at the end so every LoRA sees an exact
    # shared token prefix before branching into a different instruction.
    return (
        "You are an on-device assistant.\n"
        f"Context type: {group_name}.\n"
        "Context:\n"
        f"{context}\n\n"
        "Task:\n"
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def xsum_contexts(dataset, max_contexts: int, max_chars: int) -> list[dict]:
    contexts: list[dict] = []
    for row in dataset:
        context = truncate_context(row["document"], max_chars)
        if len(context) < 400:
            continue
        contexts.append(
            {
                "source_dataset": "xsum",
                "source_id": str(row["id"]),
                "context": context,
                "reference_summary": clean_text(row["summary"]),
            }
        )
        if len(contexts) >= max_contexts:
            break
    return contexts


def sharegpt_contexts(dataset, max_contexts: int, max_chars: int) -> list[dict]:
    contexts: list[dict] = []

    for row in dataset:
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            continue

        parts: list[str] = []
        for turn in conversations[:6]:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("from", "unknown")).lower()
            value = clean_text(turn.get("value", ""))
            if not value:
                continue

            role = "User" if speaker in {"human", "user"} else "Assistant"
            parts.append(f"{role}: {value}")

            if sum(map(len, parts)) >= max_chars:
                break

        context = truncate_context("\n".join(parts), max_chars)
        if len(context) < 400:
            continue

        contexts.append(
            {
                "source_dataset": "sharegpt52k",
                "source_id": str(row["id"]),
                "context": context,
                "reference_summary": None,
            }
        )
        if len(contexts) >= max_contexts:
            break

    return contexts


def build_delta_workload(
    output_dir: Path,
    writing_contexts: list[dict],
    conversation_contexts_data: list[dict],
) -> dict[str, int]:
    requests: list[dict] = []
    pairs: list[dict] = []
    request_id = 0

    sources = {
        "writing": writing_contexts,
        "conversation": conversation_contexts_data,
    }

    for group_name, contexts in sources.items():
        loras = GROUPS[group_name]
        anchor_lora = next(lora for lora in loras if lora.is_anchor)

        for context_index, item in enumerate(contexts):
            context_id = f"{group_name}_{context_index:06d}"
            prefix = make_prefix(group_name, item["context"])
            prefix_hash = stable_hash(prefix)
            local_requests: dict[int, int] = {}

            for lora in loras:
                prompt = prefix + lora.task
                row = {
                    "request_id": request_id,
                    "experiment": "cross_lora_delta",
                    "group_name": group_name,
                    "context_id": context_id,
                    "source_dataset": item["source_dataset"],
                    "source_id": item["source_id"],
                    "lora_id": lora.lora_id,
                    "lora_name": lora.name,
                    "is_anchor_lora": lora.is_anchor,
                    "task": lora.task,
                    "common_prefix_text": prefix,
                    "common_prefix_hash": prefix_hash,
                    "estimated_common_prefix_tokens": estimated_tokens(prefix),
                    "prompt": prompt,
                    "estimated_prompt_tokens": estimated_tokens(prompt),
                    "reference_summary": item["reference_summary"],
                }
                requests.append(row)
                local_requests[lora.lora_id] = request_id
                request_id += 1

            anchor_request_id = local_requests[anchor_lora.lora_id]
            for child_lora in loras:
                if child_lora.is_anchor:
                    continue
                pairs.append(
                    {
                        "pair_name": (
                            f"{context_id}_{anchor_lora.name}_vs_{child_lora.name}"
                        ),
                        "group_name": group_name,
                        "context_id": context_id,
                        "common_prefix_hash": prefix_hash,
                        "estimated_common_prefix_tokens": estimated_tokens(prefix),
                        "anchor_request_id": anchor_request_id,
                        "child_request_id": local_requests[child_lora.lora_id],
                        "anchor_lora_id": anchor_lora.lora_id,
                        "child_lora_id": child_lora.lora_id,
                        "probe_type": "same_prefix_different_lora",
                    }
                )

    request_count = write_jsonl(output_dir / "delta_requests.jsonl", requests)
    pair_count = write_jsonl(output_dir / "delta_pairs.jsonl", pairs)
    return {"delta_requests": request_count, "delta_pairs": pair_count}


def app_to_group(app_name: str) -> str:
    normalized = clean_text(app_name).lower()
    if any(keyword in normalized for keyword in WRITING_APP_KEYWORDS):
        return "writing"
    if any(keyword in normalized for keyword in CONVERSATION_APP_KEYWORDS):
        return "conversation"
    if any(keyword in normalized for keyword in INFORMATION_APP_KEYWORDS):
        return "information"
    return "conversation"


def pareto_choice(rng: np.random.Generator, items: list[LogicalLora], alpha: float) -> LogicalLora:
    ranks = np.arange(1, len(items) + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    return items[int(rng.choice(len(items), p=weights))]


def stable_app_lora(app_name: str, group_name: str) -> LogicalLora:
    """Map one App to a stable logical LoRA inside its routed group.

    LSApp contains App traces rather than real LoRA traces. A deterministic
    mapping keeps the synthesized LoRA label reproducible and lets a GRU learn
    App-transition patterns instead of trying to predict independently sampled
    random labels.
    """
    loras = GROUPS[group_name]
    index = int(stable_hash(f"{group_name}:{app_name}"), 16) % len(loras)
    return loras[index]


def context_pool_for_group(
    group_name: str,
    writing_contexts: list[dict],
    conversation_contexts_data: list[dict],
) -> list[dict]:
    if group_name == "conversation":
        return conversation_contexts_data
    return writing_contexts


def select_lsapp_user(events: pd.DataFrame, requested_user_id: int | None) -> int:
    if requested_user_id is not None:
        if requested_user_id not in set(events["user_id"].unique()):
            raise ValueError(f"LSApp does not contain user_id={requested_user_id}")
        return requested_user_id
    return int(events["user_id"].value_counts().idxmax())


def build_grouped_workload(
    output_dir: Path,
    lsapp_path: Path,
    writing_contexts: list[dict],
    conversation_contexts_data: list[dict],
    num_events: int,
    seed: int,
    pareto_alpha: float,
    context_reuse_probability: float,
    requested_user_id: int | None,
    time_scale: float,
    kill_gap_seconds: int,
) -> dict[str, int | float]:
    rng = np.random.default_rng(seed)
    python_rng = random.Random(seed)

    events = pd.read_csv(lsapp_path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events = events.dropna(subset=["timestamp", "app_name", "user_id"])
    events = events[events["event_type"].isin(["Opened", "User Interaction"])]

    user_id = select_lsapp_user(events, requested_user_id)
    user_events = events[events["user_id"] == user_id].sort_values("timestamp").head(num_events)
    if user_events.empty:
        raise ValueError("No LSApp events were selected")

    first_timestamp = user_events.iloc[0]["timestamp"]
    last_context_by_group: dict[str, dict] = {}
    context_counters: defaultdict[str, int] = defaultdict(int)
    requests: list[dict] = []
    state_events: list[dict] = []
    previous_app: str | None = None
    previous_timestamp: pd.Timestamp | None = None
    previous_session_id: int | None = None

    for request_id, (_, event) in enumerate(user_events.iterrows()):
        timestamp: pd.Timestamp = event["timestamp"]
        app_name = clean_text(event["app_name"])
        session_id = int(event["session_id"])

        # User-focused contexts are hot only inside the current mobile session.
        # Starting a new session invalidates the previous session's context pool.
        if previous_session_id is not None and session_id != previous_session_id:
            last_context_by_group.clear()

        group_name = app_to_group(app_name)
        lora = pareto_choice(rng, GROUPS[group_name], pareto_alpha)

        pool = context_pool_for_group(
            group_name,
            writing_contexts,
            conversation_contexts_data,
        )
        reuse_context = (
            group_name in last_context_by_group
            and rng.random() < context_reuse_probability
        )
        if reuse_context:
            context_item = last_context_by_group[group_name]
        else:
            context_item = python_rng.choice(pool)
            last_context_by_group[group_name] = context_item
            context_counters[group_name] += 1

        context_id = (
            f"session_{session_id}_{group_name}_"
            f"{context_counters[group_name]:05d}_{context_item['source_id']}"
        )
        prefix = make_prefix(group_name, context_item["context"])
        prompt = prefix + lora.task
        original_elapsed_ms = int((timestamp - first_timestamp).total_seconds() * 1000)
        arrival_ms = int(original_elapsed_ms * time_scale)

        requests.append(
            {
                "request_id": request_id,
                "experiment": "grouped_lora_routing",
                "arrival_ms": arrival_ms,
                "original_timestamp": timestamp.isoformat(),
                "user_id": int(event["user_id"]),
                "session_id": session_id,
                "app_name": app_name,
                "app_state": "foreground",
                "group_name": group_name,
                "context_id": context_id,
                "source_dataset": context_item["source_dataset"],
                "source_id": context_item["source_id"],
                "lora_id": lora.lora_id,
                "lora_name": lora.name,
                "is_anchor_lora": lora.is_anchor,
                "task": lora.task,
                "common_prefix_hash": stable_hash(prefix),
                "common_prefix_text": prefix,
                "estimated_common_prefix_tokens": estimated_tokens(prefix),
                "prompt": prompt,
                "estimated_prompt_tokens": estimated_tokens(prompt),
            }
        )

        if previous_app is not None and previous_app != app_name:
            gap_seconds = int((timestamp - previous_timestamp).total_seconds())
            state_events.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "arrival_ms": arrival_ms,
                    "app_name": previous_app,
                    "new_state": "killed" if gap_seconds >= kill_gap_seconds else "background",
                    "reason": f"next_app_opened_gap_{gap_seconds}s",
                }
            )
        state_events.append(
            {
                "timestamp": timestamp.isoformat(),
                "arrival_ms": arrival_ms,
                "app_name": app_name,
                "new_state": "foreground",
                "reason": "app_event",
            }
        )

        previous_app = app_name
        previous_timestamp = timestamp
        previous_session_id = session_id

    request_count = write_jsonl(output_dir / "grouped_requests.jsonl", requests)
    state_count = write_jsonl(output_dir / "app_state_trace.jsonl", state_events)
    return {
        "grouped_requests": request_count,
        "app_state_events": state_count,
        "selected_lsapp_user_id": user_id,
        "pareto_alpha": pareto_alpha,
    }


def build_gru_workload(
    output_dir: Path,
    lsapp_path: Path,
    user_limit: int,
    events_per_user: int,
    history_length: int,
    min_history: int,
) -> dict[str, int]:
    """Create real App sequences and synthesized LoRA prediction samples.

    Each training sample uses previous events in the same LSApp session to
    predict the next App and its deterministically associated logical LoRA.
    Histories never cross session boundaries.
    """
    events = pd.read_csv(lsapp_path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events = events.dropna(subset=["timestamp", "app_name", "user_id", "session_id"])
    events = events[events["event_type"].isin(["Opened", "User Interaction"])]

    user_counts = events["user_id"].value_counts()
    selected_users = list(user_counts.index)
    if user_limit > 0:
        selected_users = selected_users[:user_limit]
    events = events[events["user_id"].isin(selected_users)].copy()

    app_names = sorted(clean_text(name) for name in events["app_name"].unique())
    app_to_id = {name: index for index, name in enumerate(app_names)}
    group_to_id = {name: index for index, name in enumerate(GROUPS)}

    raw_events: list[dict] = []
    samples: list[dict] = []
    app_lora_map: dict[str, dict] = {}
    global_event_id = 0
    sample_id = 0

    for user_id in selected_users:
        user_events = events[events["user_id"] == user_id].sort_values("timestamp")
        if events_per_user > 0:
            user_events = user_events.head(events_per_user)

        for session_id, session_frame in user_events.groupby("session_id", sort=False):
            session_rows: list[dict] = []
            previous_timestamp: pd.Timestamp | None = None

            for session_index, (_, event) in enumerate(session_frame.iterrows()):
                timestamp: pd.Timestamp = event["timestamp"]
                app_name = clean_text(event["app_name"])
                group_name = app_to_group(app_name)
                lora = stable_app_lora(app_name, group_name)
                delta_ms = 0
                if previous_timestamp is not None:
                    delta_ms = max(
                        0,
                        int((timestamp - previous_timestamp).total_seconds() * 1000),
                    )

                row = {
                    "event_id": global_event_id,
                    "user_id": int(user_id),
                    "session_id": int(session_id),
                    "session_index": session_index,
                    "timestamp": timestamp.isoformat(),
                    "delta_ms": delta_ms,
                    "event_type": clean_text(event["event_type"]),
                    "app_name": app_name,
                    "app_id": app_to_id[app_name],
                    "group_name": group_name,
                    "group_id": group_to_id[group_name],
                    "lora_id": lora.lora_id,
                    "lora_name": lora.name,
                    "is_session_start": session_index == 0,
                    "lora_label_source": "synthetic_stable_app_mapping",
                }
                raw_events.append(row)
                session_rows.append(row)
                app_lora_map[app_name] = {
                    "app_id": app_to_id[app_name],
                    "group_name": group_name,
                    "group_id": group_to_id[group_name],
                    "lora_id": lora.lora_id,
                    "lora_name": lora.name,
                }
                global_event_id += 1
                previous_timestamp = timestamp

            # History [i-k, i) predicts event i. Do not cross sessions.
            for target_index in range(min_history, len(session_rows)):
                history = session_rows[
                    max(0, target_index - history_length) : target_index
                ]
                target = session_rows[target_index]
                samples.append(
                    {
                        "sample_id": sample_id,
                        "user_id": int(user_id),
                        "session_id": int(session_id),
                        "history_event_ids": [item["event_id"] for item in history],
                        "history_app_ids": [item["app_id"] for item in history],
                        "history_group_ids": [item["group_id"] for item in history],
                        "history_lora_ids": [item["lora_id"] for item in history],
                        "history_delta_ms": [item["delta_ms"] for item in history],
                        "target_event_id": target["event_id"],
                        "target_app_id": target["app_id"],
                        "target_app_name": target["app_name"],
                        "target_group_id": target["group_id"],
                        "target_group_name": target["group_name"],
                        "target_lora_id": target["lora_id"],
                        "target_lora_name": target["lora_name"],
                    }
                )
                sample_id += 1

    event_count = write_jsonl(output_dir / "gru_lora_events.jsonl", raw_events)
    sample_count = write_jsonl(output_dir / "gru_lora_samples.jsonl", samples)
    vocabulary = {
        "label_notice": (
            "LSApp provides real App sequences but no real LoRA labels. "
            "lora_id is synthesized with a stable App-to-LoRA mapping."
        ),
        "apps": app_to_id,
        "groups": group_to_id,
        "loras": {
            lora.name: lora.lora_id
            for loras in GROUPS.values()
            for lora in loras
        },
        "app_lora_map": app_lora_map,
    }
    (output_dir / "gru_vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "gru_users": len(selected_users),
        "gru_events": event_count,
        "gru_samples": sample_count,
    }


def save_group_config(output_dir: Path) -> None:
    config = {
        group_name: [
            {
                "lora_id": lora.lora_id,
                "logical_name": lora.name,
                "actual_adapter_path": None,
                "task": lora.task,
                "is_anchor": lora.is_anchor,
            }
            for lora in loras
        ]
        for group_name, loras in GROUPS.items()
    }
    path = output_dir / "lora_groups.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


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
        default=Path(r"D:\ecnu_experiment\datasets\mobilora_workloads"),
    )
    parser.add_argument("--contexts-per-source", type=int, default=100)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--lsapp-events", type=int, default=300)
    parser.add_argument("--lsapp-user-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pareto-alpha", type=float, default=1.2)
    parser.add_argument("--context-reuse-probability", type=float, default=0.70)
    parser.add_argument("--time-scale", type=float, default=0.01)
    parser.add_argument("--kill-gap-seconds", type=int, default=600)
    parser.add_argument("--gru-user-limit", type=int, default=50)
    parser.add_argument("--gru-events-per-user", type=int, default=1000)
    parser.add_argument("--gru-history-length", type=int, default=10)
    parser.add_argument("--gru-min-history", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    xsum = load_from_disk(str(args.dataset_root / "xsum_dataset"))
    sharegpt = load_from_disk(str(args.dataset_root / "sharegpt52k_dataset"))

    writing_contexts = xsum_contexts(
        xsum,
        args.contexts_per_source,
        args.max_context_chars,
    )
    conversation_contexts_data = sharegpt_contexts(
        sharegpt,
        args.contexts_per_source,
        args.max_context_chars,
    )

    if not writing_contexts or not conversation_contexts_data:
        raise RuntimeError("No valid XSum or ShareGPT contexts were produced")

    delta_stats = build_delta_workload(
        args.output_dir / "delta",
        writing_contexts,
        conversation_contexts_data,
    )
    grouped_stats = build_grouped_workload(
        args.output_dir / "grouped",
        args.dataset_root / "lsapp" / "df_start.csv",
        writing_contexts,
        conversation_contexts_data,
        args.lsapp_events,
        args.seed,
        args.pareto_alpha,
        args.context_reuse_probability,
        args.lsapp_user_id,
        args.time_scale,
        args.kill_gap_seconds,
    )
    gru_stats = build_gru_workload(
        args.output_dir / "gru",
        args.dataset_root / "lsapp" / "df_start.csv",
        args.gru_user_limit,
        args.gru_events_per_user,
        args.gru_history_length,
        args.gru_min_history,
    )
    save_group_config(args.output_dir)

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "xsum_contexts": len(writing_contexts),
        "sharegpt_contexts": len(conversation_contexts_data),
        **delta_stats,
        **grouped_stats,
        **gru_stats,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
