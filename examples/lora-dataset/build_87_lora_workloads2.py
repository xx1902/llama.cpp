#!/usr/bin/env python3
"""Build 87-App/87-LoRA workloads for the llama.cpp experiments.

The builder keeps three concepts separate:

1. App -> LoRA is a stable one-to-one mapping.
2. LoRA groups are computed from semantic family and LSApp transition affinity.
3. Prefix text is shared; LoRA-specific instructions remain in the suffix.

Run the script once for each duration slicing policy (0, 5, and 10 minutes).
Only the GRU event stream changes with duration slicing. The App/LoRA mapping,
groups, delta workload, and online request trace stay fixed for fair comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Adapter:
    lora_id: int
    repo_id: str
    gguf_path: str
    manifest_group: str
    family: str
    logical_name: str
    task: str


APP_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "communication": (
        "message", "messenger", "telegram", "whatsapp", "wechat", "kik",
        "text", "phone", "contact", "hangout", "discord", "badoo",
    ),
    "productivity": (
        "gmail", "email", "outlook", "notes", "calendar", "drive", "office",
        "survey",
    ),
    "information": (
        "google", "chrome", "browser", "bing", "maps", "quora", "reddit",
        "twitter", "flipboard", "internet", "play store",
    ),
    "media": (
        "youtube", "netflix", "hulu", "music", "spotify", "pandora", "podcast",
        "instagram", "snapchat", "pinterest", "photos", "gallery", "camera",
        "movie", "faceu", "pixlr",
    ),
    "commerce": (
        "amazon", "walmart", "offerup", "paypal", "pay", "ibotta", "receipt",
        "reward", "swagbucks", "robinhood", "shopping",
    ),
    "health_context": (
        "calorie", "speedometer", "health", "fitness", "muiq",
    ),
    "system": (
        "settings", "clean", "clock", "calculator", "android", "launcher",
        "slidejoy",
    ),
    "games": (
        "game", "baseball", "minesweeper", "luck", "pch",
    ),
}


FAMILY_PREFERENCES: dict[str, tuple[str, ...]] = {
    "communication": ("writing", "summarization", "classification", "general"),
    "productivity": ("writing", "summarization", "code", "general"),
    "information": ("reasoning", "classification", "summarization", "general"),
    "media": ("creative_writing", "summarization", "classification", "general"),
    "commerce": ("reasoning", "classification", "general"),
    "health_context": ("mobile_context", "classification", "general"),
    "system": ("code", "mobile_context", "reasoning", "general"),
    "games": ("reasoning", "creative_writing", "general"),
    "general": ("general", "reasoning", "summarization", "mobile_context"),
}


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimated_tokens(text: str) -> int:
    ascii_count = sum(ord(char) < 128 for char in text)
    return max(1, math.ceil(ascii_count / 4 + (len(text) - ascii_count) / 1.5))


def truncate_context(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    boundary = max(truncated.rfind(". "), truncated.rfind("\n"))
    if boundary >= max_chars // 2:
        truncated = truncated[: boundary + 1]
    return truncated.strip()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_split(path: Path) -> Dataset:
    data = load_from_disk(str(path))
    if isinstance(data, DatasetDict):
        for split in ("train", "validation", "test"):
            if split in data:
                return data[split]
        return next(iter(data.values()))
    if not isinstance(data, Dataset):
        raise TypeError(f"unsupported dataset object at {path}: {type(data)}")
    return data


def load_context_pools(dataset_root: Path, count: int, max_chars: int) -> dict[str, list[dict]]:
    xsum = dataset_split(dataset_root / "xsum_dataset")
    sharegpt = dataset_split(dataset_root / "sharegpt52k_dataset")
    articles: list[dict] = []
    conversations: list[dict] = []

    for index, row in enumerate(xsum):
        context = truncate_context(row.get("document", ""), max_chars)
        if len(context) < 400:
            continue
        articles.append(
            {
                "source_dataset": "xsum",
                "source_id": str(row.get("id", index)),
                "context": context,
            }
        )
        if len(articles) >= count:
            break

    for index, row in enumerate(sharegpt):
        turns = row.get("conversations", [])
        if not isinstance(turns, list):
            continue
        parts: list[str] = []
        for turn in turns[:8]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "user")).lower()
            role = "User" if role in {"human", "user"} else "Assistant"
            value = clean_text(turn.get("value", ""))
            if value:
                parts.append(f"{role}: {value}")
        context = truncate_context("\n".join(parts), max_chars)
        if len(context) < 400:
            continue
        conversations.append(
            {
                "source_dataset": "sharegpt52k",
                "source_id": str(row.get("id", index)),
                "context": context,
            }
        )
        if len(conversations) >= count:
            break

    if not articles or not conversations:
        raise RuntimeError("failed to load sufficient XSum/ShareGPT contexts")
    return {"article": articles, "conversation": conversations}


def adapter_family(entry: dict[str, Any]) -> str:
    repo = str(entry.get("repo_id", "")).lower()
    group = str(entry.get("group", "general"))
    # Most xw17 adapters are variants trained on mobile/personal sensing data.
    if repo.startswith("xw17/"):
        return "mobile_context"
    return group


def adapter_task(family: str, repo_id: str) -> str:
    tasks = {
        "writing": "Correct and rewrite the user content clearly.",
        "code": "Analyze the content and provide a concise technical solution.",
        "creative_writing": "Create a short engaging response based on the context.",
        "reasoning": "Reason step by step and give the final answer.",
        "summarization": "Summarize the important information in the context.",
        "classification": "Classify the intent or sentiment and explain briefly.",
        "mobile_context": "Infer the user's current activity or context and provide assistance.",
        "general": "Respond helpfully to the user's request using the context.",
    }
    base = tasks.get(family, tasks["general"])
    return f"{base} Adapter: {repo_id.split('/')[-1]}."


def load_adapters(manifest_path: Path) -> list[Adapter]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = sorted(manifest["loras"], key=lambda item: int(item["lora_id"]))
    if len(entries) != 87:
        raise ValueError(f"manifest has {len(entries)} LoRAs; expected 87")
    ids = [int(entry["lora_id"]) for entry in entries]
    if ids != list(range(87)):
        raise ValueError("manifest LoRA IDs must be exactly 0..86")

    adapters: list[Adapter] = []
    for entry in entries:
        gguf_path = Path(entry.get("gguf_path", ""))
        if entry.get("status") != "converted" or not gguf_path.is_file():
            raise FileNotFoundError(
                f"lora_id={entry['lora_id']} has no converted GGUF: {gguf_path}"
            )
        family = adapter_family(entry)
        repo_id = str(entry["repo_id"])
        adapters.append(
            Adapter(
                lora_id=int(entry["lora_id"]),
                repo_id=repo_id,
                gguf_path=str(gguf_path),
                manifest_group=str(entry.get("group", "general")),
                family=family,
                logical_name=re.sub(r"[^A-Za-z0-9._-]+", "__", repo_id),
                task=adapter_task(family, repo_id),
            )
        )
    return adapters


def classify_app(app_name: str) -> str:
    name = clean_text(app_name).lower()
    for family, keywords in APP_FAMILY_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            return family
    return "general"


def assignment_score(app_family: str, adapter: Adapter) -> float:
    preferred = FAMILY_PREFERENCES[app_family]
    if adapter.family in preferred:
        score = 100.0 - 15.0 * preferred.index(adapter.family)
    else:
        score = 5.0
    # Deterministic tie-breaking without changing the semantic priority.
    score -= adapter.lora_id * 1e-5
    return score


def assign_apps_to_loras(app_names: list[str], adapters: list[Adapter]) -> dict[str, Adapter]:
    scores = np.array(
        [
            [assignment_score(classify_app(app), adapter) for adapter in adapters]
            for app in app_names
        ],
        dtype=float,
    )
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-scores)
        mapping = {app_names[row]: adapters[column] for row, column in zip(rows, columns)}
    except ImportError:
        remaining = set(range(len(adapters)))
        mapping = {}
        for row, app in sorted(
            enumerate(app_names),
            key=lambda item: -float(scores[item[0]].max()),
        ):
            column = max(remaining, key=lambda index: scores[row, index])
            remaining.remove(column)
            mapping[app] = adapters[column]
    if len(mapping) != 87 or len({item.lora_id for item in mapping.values()}) != 87:
        raise RuntimeError("failed to create a one-to-one 87 App/LoRA mapping")
    return mapping


def filtered_lsapp(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events["open_time"] = pd.to_datetime(events["open_time"], errors="coerce")
    events["close_time"] = pd.to_datetime(events["close_time"], errors="coerce")
    events = events.dropna(subset=["timestamp", "app_name", "user_id", "session_id"])
    events = events[events["event_type"].isin(["Opened", "User Interaction"])]
    events["app_name"] = events["app_name"].map(clean_text)
    return events.sort_values(["user_id", "session_id", "timestamp"])


def transition_statistics(
    events: pd.DataFrame,
    app_names: list[str],
) -> tuple[np.ndarray, Counter[str]]:
    app_to_id = {name: index for index, name in enumerate(app_names)}
    matrix = np.zeros((len(app_names), len(app_names)), dtype=np.int64)
    frequency: Counter[str] = Counter(events["app_name"])
    for (_, _), frame in events.groupby(["user_id", "session_id"], sort=False):
        sequence = [app_to_id[name] for name in frame["app_name"] if name in app_to_id]
        for left, right in zip(sequence, sequence[1:]):
            matrix[left, right] += 1
    return matrix, frequency


def cluster_apps(
    app_names: list[str],
    app_mapping: dict[str, Adapter],
    transitions: np.ndarray,
    target_groups: int,
    max_group_size: int,
    transition_weight: float,
) -> list[list[str]]:
    if target_groups * max_group_size < len(app_names):
        raise ValueError("group capacity is smaller than 87 apps")
    symmetric = transitions + transitions.T
    normalized = np.log1p(symmetric.astype(float))
    if normalized.max() > 0:
        normalized /= normalized.max()
    app_to_id = {name: index for index, name in enumerate(app_names)}

    def pair_affinity(left: str, right: str) -> float:
        transition = normalized[app_to_id[left], app_to_id[right]]
        same_adapter_family = float(app_mapping[left].family == app_mapping[right].family)
        same_app_family = float(classify_app(left) == classify_app(right))
        semantic = 0.7 * same_adapter_family + 0.3 * same_app_family
        return transition_weight * transition + (1.0 - transition_weight) * semantic

    affinity = {
        (left, right): pair_affinity(left, right)
        for left in app_names
        for right in app_names
        if left != right
    }
    degree = {
        app: float(symmetric[app_to_id[app]].sum())
        for app in app_names
    }

    # Farthest-first seeds avoid collapsing all high-frequency Apps into one group.
    seeds = [max(app_names, key=lambda app: (degree[app], app))]
    while len(seeds) < target_groups:
        candidates = [app for app in app_names if app not in seeds]
        next_seed = min(
            candidates,
            key=lambda app: (
                max(affinity.get((app, seed), 0.0) for seed in seeds),
                -degree[app],
                app,
            ),
        )
        seeds.append(next_seed)

    base_size, remainder = divmod(len(app_names), target_groups)
    capacities = [base_size + (1 if index < remainder else 0) for index in range(target_groups)]
    if max(capacities) > max_group_size:
        raise ValueError("balanced group size exceeds max_group_size")
    groups = [[seed] for seed in seeds]

    remaining = [app for app in app_names if app not in seeds]
    # Assign difficult Apps first; each App chooses the best non-full group.
    remaining.sort(
        key=lambda app: (
            max(affinity.get((app, seed), 0.0) for seed in seeds),
            degree[app],
            app,
        )
    )
    for app in remaining:
        candidates = [index for index, group in enumerate(groups) if len(group) < capacities[index]]
        group_index = max(
            candidates,
            key=lambda index: (
                float(np.mean([affinity.get((app, member), 0.0) for member in groups[index]])),
                capacities[index] - len(groups[index]),
                -index,
            ),
        )
        groups[group_index].append(app)

    return [sorted(group) for group in groups]


def choose_anchor(group: list[str], transitions: np.ndarray, app_names: list[str], frequency: Counter[str]) -> str:
    app_to_id = {name: index for index, name in enumerate(app_names)}
    members = [app_to_id[name] for name in group]
    return max(
        group,
        key=lambda name: (
            int(transitions[app_to_id[name], members].sum())
            + int(transitions[members, app_to_id[name]].sum()),
            frequency[name],
            -app_to_id[name],
        ),
    )


def build_group_metadata(
    app_names: list[str],
    app_mapping: dict[str, Adapter],
    clusters: list[list[str]],
    transitions: np.ndarray,
    frequency: Counter[str],
) -> tuple[dict[str, list[dict]], dict[str, str], list[dict]]:
    config: dict[str, list[dict]] = {}
    app_to_group: dict[str, str] = {}
    mapping_rows: list[dict] = []
    for index, apps in enumerate(clusters):
        family_counts = Counter(app_mapping[app].family for app in apps)
        dominant_family = family_counts.most_common(1)[0][0]
        group_name = f"group_{index:02d}_{dominant_family}"
        anchor_app = choose_anchor(apps, transitions, app_names, frequency)
        entries: list[dict] = []
        for app in sorted(apps, key=lambda name: app_mapping[name].lora_id):
            adapter = app_mapping[app]
            app_to_group[app] = group_name
            entries.append(
                {
                    "lora_id": adapter.lora_id,
                    "logical_name": adapter.logical_name,
                    "actual_adapter_path": adapter.gguf_path,
                    "task": adapter.task,
                    "is_anchor": app == anchor_app,
                    "app_name": app,
                    "app_family": classify_app(app),
                    "adapter_family": adapter.family,
                    "repo_id": adapter.repo_id,
                }
            )
            mapping_rows.append(
                {
                    "app_id": app_names.index(app),
                    "app_name": app,
                    "app_family": classify_app(app),
                    "lora_id": adapter.lora_id,
                    "lora_name": adapter.logical_name,
                    "adapter_family": adapter.family,
                    "repo_id": adapter.repo_id,
                    "group_name": group_name,
                    "is_anchor": int(app == anchor_app),
                    "gguf_path": adapter.gguf_path,
                }
            )
        config[group_name] = entries
    return config, app_to_group, sorted(mapping_rows, key=lambda row: row["app_id"])


def make_prefix(context: str) -> str:
    return (
        "You are an on-device assistant.\n"
        "The following user context may be shared by several applications.\n"
        "User context:\n"
        f"{context}\n\n"
        "Application task:\n"
    )


def task_suffix(app_name: str, adapter: Adapter) -> str:
    return f"App: {app_name}. {adapter.task}"


def context_pool_for_app(app_name: str, pools: dict[str, list[dict]]) -> list[dict]:
    return pools["conversation"] if classify_app(app_name) == "communication" else pools["article"]


def build_delta_workload(
    output_dir: Path,
    group_config: dict[str, list[dict]],
    pools: dict[str, list[dict]],
    contexts_per_group: int,
    chunk_tokens: int,
) -> dict[str, int]:
    requests: list[dict] = []
    pairs: list[dict] = []
    request_id = 0
    article_contexts = pools["article"]
    for group_index, (group_name, entries) in enumerate(group_config.items()):
        anchor = next(entry for entry in entries if entry["is_anchor"])
        contexts = article_contexts[
            group_index * contexts_per_group : (group_index + 1) * contexts_per_group
        ]
        if len(contexts) < contexts_per_group:
            contexts = article_contexts[:contexts_per_group]
        for context_index, context_item in enumerate(contexts):
            context_id = f"{group_name}_context_{context_index:04d}"
            prefix = make_prefix(context_item["context"])
            prefix_hash = stable_hash(prefix)
            local_ids: dict[int, int] = {}
            for entry in entries:
                suffix = task_suffix(entry["app_name"], Adapter(
                    entry["lora_id"], entry["repo_id"], entry["actual_adapter_path"],
                    entry["adapter_family"], entry["adapter_family"],
                    entry["logical_name"], entry["task"],
                ))
                requests.append(
                    {
                        "request_id": request_id,
                        "experiment": "same_group_same_prefix_delta",
                        "group_name": group_name,
                        "context_id": context_id,
                        "source_dataset": context_item["source_dataset"],
                        "source_id": context_item["source_id"],
                        "lora_id": entry["lora_id"],
                        "lora_name": entry["logical_name"],
                        "app_name": entry["app_name"],
                        "is_anchor_lora": entry["is_anchor"],
                        "common_prefix_text": prefix,
                        "common_prefix_hash": prefix_hash,
                        "prompt": prefix + suffix,
                        "task": suffix,
                        "chunk_token_size": chunk_tokens,
                        "prefix_segment_types": ["shared_system", "user_context"],
                        "estimated_common_prefix_tokens": estimated_tokens(prefix),
                    }
                )
                local_ids[entry["lora_id"]] = request_id
                request_id += 1
            for child in entries:
                if child["is_anchor"]:
                    continue
                pairs.append(
                    {
                        "pair_name": f"{context_id}_lora_{anchor['lora_id']}_vs_{child['lora_id']}",
                        "pair_type": "intra_group",
                        "group_name": group_name,
                        "context_id": context_id,
                        "common_prefix_hash": prefix_hash,
                        "anchor_request_id": local_ids[anchor["lora_id"]],
                        "child_request_id": local_ids[child["lora_id"]],
                        "anchor_lora_id": anchor["lora_id"],
                        "child_lora_id": child["lora_id"],
                    }
                )
    return {
        "delta_requests": write_jsonl(output_dir / "delta_requests.jsonl", requests),
        "delta_pairs": write_jsonl(output_dir / "delta_pairs.jsonl", pairs),
    }


def build_cross_group_control(
    output_root: Path,
    mapping_rows: list[dict],
    pools: dict[str, list[dict]],
    pair_count: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    by_family: defaultdict[str, list[dict]] = defaultdict(list)
    for row in mapping_rows:
        by_family[row["adapter_family"]].append(row)
    families = sorted(by_family)
    pairs_to_run: list[tuple[dict, dict]] = []
    while len(pairs_to_run) < pair_count:
        left_family, right_family = rng.sample(families, 2)
        left = rng.choice(by_family[left_family])
        right = rng.choice(by_family[right_family])
        key = tuple(sorted((left["lora_id"], right["lora_id"])))
        if any(tuple(sorted((a["lora_id"], b["lora_id"]))) == key for a, b in pairs_to_run):
            continue
        pairs_to_run.append((left, right))

    control_dir = output_root / "cross_group_control"
    all_group = []
    for row in mapping_rows:
        all_group.append(
            {
                "lora_id": row["lora_id"],
                "logical_name": row["lora_name"],
                "actual_adapter_path": row["gguf_path"],
                "task": "Cross-group delta control.",
                "is_anchor": False,
            }
        )
    (control_dir / "lora_groups.json").parent.mkdir(parents=True, exist_ok=True)
    (control_dir / "lora_groups.json").write_text(
        json.dumps({"all_control": all_group}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    requests: list[dict] = []
    pairs: list[dict] = []
    request_id = 0
    for index, (left, right) in enumerate(pairs_to_run):
        context_item = pools["article"][index % len(pools["article"])]
        context_id = f"cross_group_{index:05d}"
        prefix = make_prefix(context_item["context"])
        prefix_hash = stable_hash(prefix)
        local_request_ids = []
        for row in (left, right):
            suffix = f"App: {row['app_name']}. Cross-group control request."
            requests.append(
                {
                    "request_id": request_id,
                    "experiment": "cross_group_control",
                    "group_name": "all_control",
                    "context_id": context_id,
                    "source_dataset": context_item["source_dataset"],
                    "lora_id": row["lora_id"],
                    "lora_name": row["lora_name"],
                    "common_prefix_text": prefix,
                    "common_prefix_hash": prefix_hash,
                    "prompt": prefix + suffix,
                }
            )
            local_request_ids.append(request_id)
            request_id += 1
        pairs.append(
            {
                "pair_name": f"cross_{left['lora_id']}_vs_{right['lora_id']}_{index}",
                "pair_type": "cross_group_control",
                "group_name": "all_control",
                "context_id": context_id,
                "common_prefix_hash": prefix_hash,
                "anchor_request_id": local_request_ids[0],
                "child_request_id": local_request_ids[1],
                "anchor_lora_id": left["lora_id"],
                "child_lora_id": right["lora_id"],
            }
        )
    write_jsonl(control_dir / "delta" / "delta_requests.jsonl", requests)
    write_jsonl(control_dir / "delta" / "delta_pairs.jsonl", pairs)
    write_jsonl(control_dir / "grouped" / "grouped_requests.jsonl", [])
    return {"cross_group_pairs": len(pairs)}


def build_online_workload(
    output_dir: Path,
    group_config: dict[str, list[dict]],
    events: pd.DataFrame,
    app_mapping: dict[str, Adapter],
    app_to_group: dict[str, str],
    pools: dict[str, list[dict]],
    user_id: int | None,
    event_limit: int,
    context_reuse_probability: float,
    time_scale: float,
    chunk_tokens: int,
    seed: int,
) -> dict[str, int]:
    if user_id is None:
        user_id = int(events["user_id"].value_counts().idxmax())
    trace = events[events["user_id"] == user_id].sort_values("timestamp").head(event_limit)
    if trace.empty:
        raise ValueError(f"no events for user_id={user_id}")
    rng = random.Random(seed)
    first_timestamp = trace.iloc[0]["timestamp"]
    current_context: dict | None = None
    current_context_id = ""
    previous_session: int | None = None
    previous_app: str | None = None
    previous_timestamp: pd.Timestamp | None = None
    requests: list[dict] = []
    states: list[dict] = []

    for request_id, (_, event) in enumerate(trace.iterrows()):
        app_name = event["app_name"]
        if app_name not in app_mapping:
            continue
        session_id = int(event["session_id"])
        timestamp: pd.Timestamp = event["timestamp"]
        if previous_session != session_id:
            current_context = None
        reuse = current_context is not None and rng.random() < context_reuse_probability
        if not reuse:
            current_context = rng.choice(context_pool_for_app(app_name, pools))
            current_context_id = (
                f"user_{user_id}_session_{session_id}_request_{request_id}_"
                f"{current_context['source_id']}"
            )
        adapter = app_mapping[app_name]
        group_name = app_to_group[app_name]
        prefix = make_prefix(current_context["context"])
        suffix = task_suffix(app_name, adapter)
        arrival_ms = int((timestamp - first_timestamp).total_seconds() * 1000 * time_scale)
        requests.append(
            {
                "request_id": request_id,
                "experiment": "lsapp_87_online",
                "arrival_ms": arrival_ms,
                "original_timestamp": timestamp.isoformat(),
                "user_id": int(user_id),
                "session_id": session_id,
                "app_name": app_name,
                "app_state": "foreground",
                "group_name": group_name,
                "context_id": current_context_id,
                "source_dataset": current_context["source_dataset"],
                "source_id": current_context["source_id"],
                "lora_id": adapter.lora_id,
                "lora_name": adapter.logical_name,
                "common_prefix_text": prefix,
                "common_prefix_hash": stable_hash(prefix),
                "prompt": prefix + suffix,
                "task": suffix,
                "chunk_token_size": chunk_tokens,
                "prefix_segment_types": ["shared_system", "user_context"],
                "estimated_common_prefix_tokens": estimated_tokens(prefix),
            }
        )
        if previous_app and previous_app != app_name and previous_timestamp is not None:
            gap = int((timestamp - previous_timestamp).total_seconds())
            states.append(
                {
                    "arrival_ms": arrival_ms,
                    "app_name": previous_app,
                    "new_state": "killed" if gap >= 300 else "background",
                    "reason": f"next_app_gap_{gap}s",
                }
            )
        states.append(
            {
                "arrival_ms": arrival_ms,
                "app_name": app_name,
                "new_state": "foreground",
                "reason": "app_event",
            }
        )
        previous_session = session_id
        previous_app = app_name
        previous_timestamp = timestamp

    request_count = write_jsonl(output_dir / "grouped_requests.jsonl", requests)
    state_count = write_jsonl(output_dir / "app_state_trace.jsonl", states)

    # One common context for all 87 adapters: loading and routing smoke test.
    smoke_context = pools["article"][0]
    smoke_prefix = make_prefix(smoke_context["context"])
    smoke_rows = []
    for request_id, app_name in enumerate(sorted(app_mapping)):
        adapter = app_mapping[app_name]
        smoke_rows.append(
            {
                "request_id": request_id,
                "experiment": "all_87_same_prefix_smoke",
                "arrival_ms": request_id * 100,
                "user_id": -1,
                "session_id": 0,
                "app_name": app_name,
                "group_name": app_to_group[app_name],
                "context_id": "all_87_shared_context",
                "source_dataset": smoke_context["source_dataset"],
                "lora_id": adapter.lora_id,
                "lora_name": adapter.logical_name,
                "common_prefix_text": smoke_prefix,
                "common_prefix_hash": stable_hash(smoke_prefix),
                "prompt": smoke_prefix + task_suffix(app_name, adapter),
                "chunk_token_size": chunk_tokens,
            }
        )
    write_jsonl(output_dir / "all_apps_same_prefix.jsonl", smoke_rows)
    smoke_root = output_dir.parent / "all_apps_smoke"
    (smoke_root / "lora_groups.json").parent.mkdir(parents=True, exist_ok=True)
    (smoke_root / "lora_groups.json").write_text(
        json.dumps(group_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(smoke_root / "delta" / "delta_requests.jsonl", [])
    write_jsonl(smoke_root / "delta" / "delta_pairs.jsonl", [])
    write_jsonl(smoke_root / "grouped" / "grouped_requests.jsonl", smoke_rows)
    return {
        "grouped_requests": request_count,
        "app_state_events": state_count,
        "online_user_id": int(user_id),
    }


def expand_duration(events: pd.DataFrame, step_min: int, max_duration_min: int) -> pd.DataFrame:
    if step_min <= 0:
        result = events.copy()
        result["slice_index"] = 0
        result["is_duration_slice"] = False
        return result
    rows: list[pd.Series] = []
    for _, row in events.iterrows():
        start, end = row["open_time"], row["close_time"]
        if pd.isna(start) or pd.isna(end):
            copied = row.copy()
            copied["slice_index"] = 0
            copied["is_duration_slice"] = False
            rows.append(copied)
            continue
        duration = (end - start).total_seconds() / 60.0
        if duration <= step_min:
            copied = row.copy()
            copied["slice_index"] = 0
            copied["is_duration_slice"] = False
            rows.append(copied)
            continue
        end_limit = start + pd.Timedelta(minutes=min(duration, max_duration_min))
        current = start
        slice_index = 0
        while current < end_limit:
            copied = row.copy()
            copied["timestamp"] = current
            copied["open_time"] = current
            copied["close_time"] = min(current + pd.Timedelta(minutes=step_min), end_limit)
            copied["slice_index"] = slice_index
            copied["is_duration_slice"] = slice_index > 0
            rows.append(copied)
            current = copied["close_time"]
            slice_index += 1
    return pd.DataFrame(rows).sort_values(["user_id", "session_id", "timestamp"])


def build_gru_workload(
    output_dir: Path,
    events: pd.DataFrame,
    app_names: list[str],
    app_mapping: dict[str, Adapter],
    app_to_group: dict[str, str],
    user_limit: int,
    events_per_user: int,
    history_length: int,
    min_history: int,
    duration_slice_min: int,
    max_duration_min: int,
) -> dict[str, int]:
    selected_users = list(events["user_id"].value_counts().index)
    if user_limit > 0:
        selected_users = selected_users[:user_limit]
    app_to_id = {name: index for index, name in enumerate(app_names)}
    group_names = sorted(set(app_to_group.values()))
    group_to_id = {name: index for index, name in enumerate(group_names)}
    raw_rows: list[dict] = []
    samples: list[dict] = []
    event_id = 0
    sample_id = 0

    for user_id in selected_users:
        user_events = events[events["user_id"] == user_id].sort_values("timestamp")
        if events_per_user > 0:
            user_events = user_events.head(events_per_user)
        user_events = expand_duration(user_events, duration_slice_min, max_duration_min)
        for session_id, frame in user_events.groupby("session_id", sort=False):
            session_rows: list[dict] = []
            previous_timestamp: pd.Timestamp | None = None
            for session_index, (_, event) in enumerate(frame.iterrows()):
                app_name = event["app_name"]
                adapter = app_mapping[app_name]
                group_name = app_to_group[app_name]
                timestamp: pd.Timestamp = event["timestamp"]
                delta_ms = 0 if previous_timestamp is None else max(
                    0, int((timestamp - previous_timestamp).total_seconds() * 1000)
                )
                row = {
                    "event_id": event_id,
                    "user_id": int(user_id),
                    "session_id": int(session_id),
                    "session_index": session_index,
                    "timestamp": timestamp.isoformat(),
                    "delta_ms": delta_ms,
                    "app_name": app_name,
                    "app_id": app_to_id[app_name],
                    "group_name": group_name,
                    "group_id": group_to_id[group_name],
                    "lora_id": adapter.lora_id,
                    "lora_name": adapter.logical_name,
                    "slice_index": int(event.get("slice_index", 0)),
                    "is_duration_slice": bool(event.get("is_duration_slice", False)),
                    "is_session_start": session_index == 0,
                    "lora_label_source": "physical_87_app_one_to_one",
                }
                raw_rows.append(row)
                session_rows.append(row)
                event_id += 1
                previous_timestamp = timestamp

            for target_index in range(min_history, len(session_rows)):
                history = session_rows[max(0, target_index - history_length) : target_index]
                target = session_rows[target_index]
                samples.append(
                    {
                        "sample_id": sample_id,
                        "user_id": int(user_id),
                        "session_id": int(session_id),
                        "history_event_ids": [item["event_id"] for item in history],
                        "history_app_ids": [item["app_id"] for item in history],
                        "history_lora_ids": [item["lora_id"] for item in history],
                        "history_group_ids": [item["group_id"] for item in history],
                        "history_delta_ms": [item["delta_ms"] for item in history],
                        "history_is_duration_slice": [item["is_duration_slice"] for item in history],
                        "target_event_id": target["event_id"],
                        "target_app_id": target["app_id"],
                        "target_app_name": target["app_name"],
                        "target_lora_id": target["lora_id"],
                        "target_lora_name": target["lora_name"],
                        "target_group_id": target["group_id"],
                        "target_group_name": target["group_name"],
                        "target_is_duration_slice": target["is_duration_slice"],
                    }
                )
                sample_id += 1

    write_jsonl(output_dir / "gru_lora_events.jsonl", raw_rows)
    write_jsonl(output_dir / "gru_lora_samples.jsonl", samples)
    vocab = {
        "label_notice": "87 LSApp Apps are mapped one-to-one to 87 physical GGUF LoRAs.",
        "apps": app_to_id,
        "loras": {app: app_mapping[app].lora_id for app in app_names},
        "groups": group_to_id,
        "duration_slice_min": duration_slice_min,
    }
    (output_dir / "gru_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"gru_events": len(raw_rows), "gru_samples": len(samples)}


def save_transition_matrix(path: Path, app_names: list[str], matrix: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, index=app_names, columns=app_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"D:\ecnu_experiment\Model\LoRA\Qwen2.5_1.5B_87\lora_87_manifest.json"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\ecnu_experiment\datasets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-slice-min", type=int, choices=[0, 5, 10], default=0)
    parser.add_argument("--max-duration-min", type=int, default=200)
    parser.add_argument("--context-count", type=int, default=160)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--delta-contexts-per-group", type=int, default=5)
    parser.add_argument("--cross-group-pairs", type=int, default=100)
    parser.add_argument("--group-count", type=int, default=10)
    parser.add_argument("--max-group-size", type=int, default=12)
    parser.add_argument("--transition-weight", type=float, default=0.65)
    parser.add_argument("--online-user-id", type=int, default=None)
    parser.add_argument("--online-events", type=int, default=1000)
    parser.add_argument("--context-reuse-probability", type=float, default=0.70)
    parser.add_argument("--time-scale", type=float, default=0.001)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--gru-user-limit", type=int, default=50)
    parser.add_argument("--gru-events-per-user", type=int, default=1000)
    parser.add_argument("--gru-history-length", type=int, default=10)
    parser.add_argument("--gru-min-history", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapters = load_adapters(args.manifest)
    events = filtered_lsapp(args.dataset_root / "lsapp" / "df_start.csv")
    app_names = sorted(events["app_name"].unique())
    if len(app_names) != 87:
        raise ValueError(f"LSApp contains {len(app_names)} Apps; expected 87")
    app_mapping = assign_apps_to_loras(app_names, adapters)
    transitions, frequency = transition_statistics(events, app_names)
    clusters = cluster_apps(
        app_names,
        app_mapping,
        transitions,
        args.group_count,
        args.max_group_size,
        args.transition_weight,
    )
    group_config, app_to_group, mapping_rows = build_group_metadata(
        app_names, app_mapping, clusters, transitions, frequency
    )
    pools = load_context_pools(args.dataset_root, args.context_count, args.max_context_chars)

    (args.output_dir / "lora_groups.json").write_text(
        json.dumps(group_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "app_lora_mapping.csv", mapping_rows)
    (args.output_dir / "app_lora_mapping.json").write_text(
        json.dumps(mapping_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_transition_matrix(args.output_dir / "app_transition_matrix.csv", app_names, transitions)

    summary: dict[str, Any] = {
        "apps": len(app_names),
        "physical_loras": len(adapters),
        "one_to_one_mapping": True,
        "groups": len(group_config),
        "duration_slice_min": args.duration_slice_min,
        "chunk_tokens": args.chunk_tokens,
        "transition_weight": args.transition_weight,
    }
    summary.update(
        build_delta_workload(
            args.output_dir / "delta",
            group_config,
            pools,
            args.delta_contexts_per_group,
            args.chunk_tokens,
        )
    )
    summary.update(
        build_cross_group_control(
            args.output_dir,
            mapping_rows,
            pools,
            args.cross_group_pairs,
            args.seed,
        )
    )
    summary.update(
        build_online_workload(
            args.output_dir / "grouped",
            group_config,
            events,
            app_mapping,
            app_to_group,
            pools,
            args.online_user_id,
            args.online_events,
            args.context_reuse_probability,
            args.time_scale,
            args.chunk_tokens,
            args.seed,
        )
    )
    summary.update(
        build_gru_workload(
            args.output_dir / "gru",
            events,
            app_names,
            app_mapping,
            app_to_group,
            args.gru_user_limit,
            args.gru_events_per_user,
            args.gru_history_length,
            args.gru_min_history,
            args.duration_slice_min,
            args.max_duration_min,
        )
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
