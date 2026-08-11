#!/usr/bin/env python3
"""Build five 30-request workloads from locally downloaded real datasets.

The logical task or language attached to a request is deliberately separate
from the physical GGUF adapter used to exercise the LoRA runtime. The adapters
in the LSApp trace are system-test artifacts; they are not claimed to have
been trained for MSC, Taskmaster, LMSYS, OPUS-100, or XSum.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
from datasets import load_from_disk


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS_DIR = SCRIPT_DIR / "datasets"
DEFAULT_ROUTE_JSONL = SCRIPT_DIR / "output" / "lsapp_30" / "source_trace_30.jsonl"
DEFAULT_LORA_GROUPS_JSON = (
    SCRIPT_DIR / "output" / "lsapp_30" / "parallel" / "lora_groups.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "real_30"

REQUEST_COUNT = 30
CHUNK_TOKEN_SIZE = 128
PHYSICAL_ADAPTER_NOTICE = (
    "The logical workload role is dataset metadata. The physical GGUF adapter "
    "is reused only to exercise LoRA switching and prefix reuse; no semantic "
    "task-training match is implied."
)

WORKLOADS = {
    "msc_continuous": {
        "dataset": "MSC",
        "form": "continuous",
        "focus": "multi-session departure, return, and non-contiguous LoRA hits",
    },
    "taskmaster_continuous": {
        "dataset": "Taskmaster-1",
        "form": "continuous",
        "focus": "personal-agent dialogue with growing task state",
    },
    "lmsys_continuous": {
        "dataset": "LMSYS-33K",
        "form": "continuous",
        "focus": "ordinary multi-turn chat baseline",
    },
    "opus100_parallel": {
        "dataset": "OPUS-100",
        "form": "parallel",
        "focus": "same shallow English prefix dispatched to language LoRAs",
    },
    "xsum_parallel": {
        "dataset": "XSum",
        "form": "parallel",
        "focus": "same article dispatched to summary, QA, and rewrite LoRAs",
    },
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimated_tokens(text: str) -> int:
    ascii_count = sum(ord(char) < 128 for char in text)
    return max(1, math.ceil(ascii_count / 4 + (len(text) - ascii_count) / 1.5))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def prefix_fields(segments: list[dict[str, str]]) -> dict[str, Any]:
    prefix = "".join(segment["text"] for segment in segments)
    return {
        "common_prefix_text": prefix,
        "common_prefix_hash": stable_hash(prefix),
        "prefix_segment_types": [segment["type"] for segment in segments],
        "prefix_segments": segments,
        "estimated_common_prefix_tokens": estimated_tokens(prefix),
        "chunk_token_size": CHUNK_TOKEN_SIZE,
    }


def route_arrival_ms(route: list[dict[str, Any]], index: int) -> int:
    from datetime import datetime

    first = datetime.fromisoformat(route[0]["timestamp"])
    current = datetime.fromisoformat(route[index]["timestamp"])
    return max(0, int((current - first).total_seconds() * 1000))


def load_adapter_inputs(
    route_path: Path, groups_path: Path
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    route = read_jsonl(route_path)
    if len(route) < REQUEST_COUNT:
        raise ValueError(f"LSApp route has {len(route)} rows; {REQUEST_COUNT} are required")

    groups = read_json(groups_path)
    catalog = {
        int(entry["lora_id"]): entry
        for entries in groups.values()
        for entry in entries
    }
    missing = sorted({int(row["lora_id"]) for row in route[:REQUEST_COUNT]} - set(catalog))
    if missing:
        raise ValueError(f"LoRAs missing from source lora_groups.json: {missing}")
    return route[:REQUEST_COUNT], catalog


def build_lora_groups(
    group_name: str,
    lora_roles: dict[int, str],
    catalog: dict[int, dict[str, Any]],
    requested_loras: list[int],
) -> dict[str, list[dict[str, Any]]]:
    counts = Counter(requested_loras)
    anchor_id = counts.most_common(1)[0][0]
    entries: list[dict[str, Any]] = []
    for lora_id in sorted(set(requested_loras)):
        entry = copy.deepcopy(catalog[lora_id])
        entry["is_anchor"] = lora_id == anchor_id
        entry["workload_role"] = lora_roles.get(lora_id, "routed_system_test_adapter")
        entry["physical_adapter_notice"] = PHYSICAL_ADAPTER_NOTICE
        entries.append(entry)
    return {group_name: entries}


def route_metadata(
    route: list[dict[str, Any]],
    catalog: dict[int, dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    row = route[index]
    lora_id = int(row["lora_id"])
    return {
        "arrival_ms": route_arrival_ms(route, index),
        "lora_id": lora_id,
        "lora_name": str(row["lora_name"]),
        "adapter_path": str(catalog[lora_id].get("actual_adapter_path", "")),
        "lsapp_route_index": index,
        "lsapp_app_name": row["app_name"],
        "lsapp_timestamp": row["timestamp"],
        "lsapp_interaction_id": row["interaction_id"],
        "physical_adapter_notice": PHYSICAL_ADAPTER_NOTICE,
    }


def make_request(
    *,
    request_id: int,
    experiment: str,
    group_name: str,
    context_id: str,
    source_dataset: str,
    segments: list[dict[str, str]],
    task: str,
    reference_response: str,
    lora_id: int,
    lora_name: str,
    adapter_path: str,
    arrival_ms: int,
    logical_role: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = prefix_fields(segments)
    request = {
        "request_id": request_id,
        "experiment": experiment,
        "group_name": group_name,
        "context_id": context_id,
        "source_dataset": source_dataset,
        "lora_id": lora_id,
        "lora_name": lora_name,
        "adapter_path": adapter_path,
        "arrival_ms": arrival_ms,
        "logical_lora_role": logical_role,
        "task": task,
        "reference_response": reference_response,
        "reference_is_real_dataset_text": True,
        "physical_adapter_notice": PHYSICAL_ADAPTER_NOTICE,
        **fields,
    }
    request["prompt"] = request["common_prefix_text"] + task
    if extra:
        request.update(extra)
    return request


def add_continuous_reuse_metadata(requests: list[dict[str, Any]]) -> dict[str, Any]:
    previous_lora_by_context: dict[str, int] = {}
    seen_loras_by_context: dict[str, set[int]] = {}
    previous_hash_by_context: dict[str, str] = {}
    same_lora = 0
    returns = 0
    transitions = 0

    for request in requests:
        context_id = str(request["context_id"])
        lora_id = int(request["lora_id"])
        previous = previous_lora_by_context.get(context_id)
        seen = seen_loras_by_context.setdefault(context_id, set())
        request["previous_lora_id"] = previous
        request["same_lora_as_previous"] = previous == lora_id if previous is not None else False
        request["return_to_lora_after_gap"] = (
            previous is not None and previous != lora_id and lora_id in seen
        )
        request["parent_prefix_hash"] = previous_hash_by_context.get(context_id, "")
        if previous is not None:
            transitions += 1
            same_lora += int(request["same_lora_as_previous"])
            returns += int(request["return_to_lora_after_gap"])
        previous_lora_by_context[context_id] = lora_id
        previous_hash_by_context[context_id] = request["common_prefix_hash"]
        seen.add(lora_id)

    return {
        "contexts": len({request["context_id"] for request in requests}),
        "lora_transitions": transitions,
        "same_lora_transitions": same_lora,
        "same_lora_transition_rate": same_lora / transitions if transitions else 0.0,
        "return_after_gap_transitions": returns,
        "return_after_gap_rate": returns / transitions if transitions else 0.0,
    }


def alternating_pairs(dialog: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    pairs: list[tuple[int, str, str]] = []
    for index in range(0, len(dialog) - 1, 2):
        user_text = clean_text(dialog[index].get("text"))
        assistant_text = clean_text(dialog[index + 1].get("text"))
        if user_text and assistant_text:
            pairs.append((index // 2, user_text, assistant_text))
    return pairs


def build_msc(
    datasets_dir: Path,
    route: list[dict[str, Any]],
    catalog: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    source = datasets_dir / "msc" / "msc" / "msc" / "msc_dialogue" / "session_5" / "valid.txt"
    selected: dict[str, Any] | None = None
    with source.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            sessions = list(record.get("previous_dialogs", [])) + [
                {"dialog": record.get("dialog", []), "time_back": "current session"}
            ]
            pair_count = sum(len(alternating_pairs(session["dialog"])) for session in sessions)
            if pair_count >= REQUEST_COUNT:
                selected = record
                break
    if selected is None:
        raise ValueError("MSC validation split contains no five-session record with 30 pairs")

    source_id = str(selected["metadata"]["initial_data_id"])
    sessions = list(selected["previous_dialogs"]) + [
        {"dialog": selected["dialog"], "time_back": "current session", "time_num": 0, "time_unit": ""}
    ]
    persona_groups = selected.get("init_personas") or selected.get("personas") or []
    persona_text = "Participant profiles:\n" + "\n".join(
        f"Participant {group_index + 1}: " + " ".join(clean_text(item) for item in group)
        for group_index, group in enumerate(persona_groups)
    ) + "\n"
    system = (
        "Continue this long-term conversation naturally. Use the participant profiles "
        "and all available earlier sessions when answering the current user.\n"
    )
    history = ""
    requests: list[dict[str, Any]] = []
    selected_session_counts: Counter[int] = Counter()

    for session_index, session in enumerate(sessions, start=1):
        time_back = clean_text(session.get("time_back")) or "unspecified gap"
        history += f"[Session {session_index} begins; source time gap: {time_back}]\n"
        for turn_index, user_text, assistant_text in alternating_pairs(session["dialog"]):
            if len(requests) == REQUEST_COUNT:
                break
            request_id = len(requests)
            route_meta = route_metadata(route, catalog, request_id)
            segments = [
                {"type": "shared_system", "text": system},
                {"type": "persona_context", "text": persona_text},
            ]
            if history:
                segments.append({"type": "conversation_history", "text": history})
            task = f"User: {user_text}\nAssistant:"
            requests.append(
                make_request(
                    request_id=request_id,
                    experiment="msc_multi_session_continuous",
                    group_name="msc_lsapp_route",
                    context_id=f"msc_{source_id}",
                    source_dataset="MSC session_5 valid",
                    segments=segments,
                    task=task,
                    reference_response=assistant_text,
                    lora_id=route_meta["lora_id"],
                    lora_name=route_meta["lora_name"],
                    adapter_path=route_meta["adapter_path"],
                    arrival_ms=route_meta["arrival_ms"],
                    logical_role="long_term_persona_chat",
                    extra={
                        **route_meta,
                        "source_record_id": source_id,
                        "source_session_index": session_index,
                        "source_session_time_back": time_back,
                        "source_turn_index": turn_index,
                    },
                )
            )
            selected_session_counts[session_index] += 1
            history += f"User: {user_text}\nAssistant: {assistant_text}\n"
        if len(requests) == REQUEST_COUNT:
            break

    reuse = add_continuous_reuse_metadata(requests)
    requested_loras = [int(request["lora_id"]) for request in requests]
    groups = build_lora_groups(
        "msc_lsapp_route",
        {lora_id: "long_term_persona_chat" for lora_id in set(requested_loras)},
        catalog,
        requested_loras,
    )
    summary = {
        **WORKLOADS["msc_continuous"],
        "requests": len(requests),
        "source_file": str(source),
        "selected_source_ids": [source_id],
        "requests_by_session": dict(sorted(selected_session_counts.items())),
        "construction": "five-session growing conversation prefix",
        "direct_lora_base_test5_compatible": False,
        **reuse,
    }
    return requests, [], summary, groups


def taskmaster_pairs(utterances: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index in range(len(utterances) - 1):
        user = utterances[index]
        assistant = utterances[index + 1]
        if user.get("speaker") == "USER" and assistant.get("speaker") == "ASSISTANT":
            if clean_text(user.get("text")) and clean_text(assistant.get("text")):
                pairs.append((len(pairs), user, assistant))
    return pairs


def annotation_updates(utterance: dict[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    for segment in utterance.get("segments", []) or []:
        segment_text = clean_text(segment.get("text"))
        for annotation in segment.get("annotations", []) or []:
            name = clean_text(annotation.get("name"))
            value = clean_text(annotation.get("value")) or segment_text
            if name and value:
                updates[name] = value
    return updates


def build_taskmaster(
    datasets_dir: Path,
    route: list[dict[str, Any]],
    catalog: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    source = datasets_dir / "taskmaster" / "TM-1-2019" / "self-dialogs.json"
    conversations = read_json(source)
    selected = [
        conversation
        for conversation in conversations
        if len(taskmaster_pairs(conversation.get("utterances", []))) >= 15
    ][:2]
    if len(selected) != 2:
        raise ValueError("Taskmaster self-dialogs needs two conversations with 15 pairs")

    system = (
        "You are a personal task assistant. Continue the task using the conversation "
        "history and the structured task-state updates appended after completed turns.\n"
    )
    requests: list[dict[str, Any]] = []
    for conversation in selected:
        conversation_id = str(conversation["conversation_id"])
        instruction_id = str(conversation.get("instruction_id", "unknown"))
        history = ""
        state: dict[str, str] = {}
        for turn_index, user, assistant in taskmaster_pairs(conversation["utterances"])[:15]:
            request_id = len(requests)
            route_meta = route_metadata(route, catalog, request_id)
            segments = [
                {"type": "shared_system", "text": system},
                {"type": "task_context", "text": f"Task instruction: {instruction_id}\n"},
            ]
            if history:
                segments.append({"type": "conversation_history", "text": history})
            user_text = clean_text(user["text"])
            assistant_text = clean_text(assistant["text"])
            current_updates = annotation_updates(user)
            task = f"User: {user_text}\nAssistant:"
            requests.append(
                make_request(
                    request_id=request_id,
                    experiment="taskmaster_task_state_continuous",
                    group_name="taskmaster_lsapp_route",
                    context_id=f"taskmaster_{conversation_id}",
                    source_dataset="Taskmaster-1 self-dialogs",
                    segments=segments,
                    task=task,
                    reference_response=assistant_text,
                    lora_id=route_meta["lora_id"],
                    lora_name=route_meta["lora_name"],
                    adapter_path=route_meta["adapter_path"],
                    arrival_ms=route_meta["arrival_ms"],
                    logical_role="personal_agent_task_state",
                    extra={
                        **route_meta,
                        "source_conversation_id": conversation_id,
                        "source_instruction_id": instruction_id,
                        "source_turn_index": turn_index,
                        "task_state_before_turn": dict(state),
                        "current_user_annotation_updates": current_updates,
                    },
                )
            )
            state.update(current_updates)
            state.update(annotation_updates(assistant))
            state_text = "; ".join(
                f"{key}={value}" for key, value in sorted(state.items())
            ) or "not specified yet"
            history += (
                f"User: {user_text}\nAssistant: {assistant_text}\n"
                f"[Task state after turn {turn_index}: {state_text}]\n"
            )

    reuse = add_continuous_reuse_metadata(requests)
    requested_loras = [int(request["lora_id"]) for request in requests]
    groups = build_lora_groups(
        "taskmaster_lsapp_route",
        {lora_id: "personal_agent_task_state" for lora_id in set(requested_loras)},
        catalog,
        requested_loras,
    )
    summary = {
        **WORKLOADS["taskmaster_continuous"],
        "requests": len(requests),
        "source_file": str(source),
        "selected_source_ids": [conversation["conversation_id"] for conversation in selected],
        "selected_instruction_ids": [conversation["instruction_id"] for conversation in selected],
        "construction": "two 15-turn growing task-state conversations",
        "direct_lora_base_test5_compatible": False,
        **reuse,
    }
    return requests, [], summary, groups


def lmsys_is_clean(record: dict[str, Any]) -> bool:
    if str(record.get("language", "")).lower() != "english":
        return False
    if record.get("winner") not in {"model_a", "model_b"}:
        return False
    if bool((record.get("openai_moderation") or {}).get("flagged")):
        return False
    for tag in (record.get("toxic_chat_tag") or {}).values():
        if bool((tag or {}).get("flagged")):
            return False
    return True


def role_pairs(conversation: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    pairs: list[tuple[int, str, str]] = []
    for index in range(len(conversation) - 1):
        user = conversation[index]
        assistant = conversation[index + 1]
        if user.get("role") == "user" and assistant.get("role") == "assistant":
            user_text = clean_text(user.get("content"))
            assistant_text = clean_text(assistant.get("content"))
            if user_text and assistant_text:
                pairs.append((len(pairs), user_text, assistant_text))
    return pairs


def conversation_is_mostly_ascii(conversation: list[dict[str, Any]]) -> bool:
    text = "".join(clean_text(message.get("content")) for message in conversation)
    if not text:
        return False
    lowered = text.lower()
    platform_prompt_markers = (
        "chatbot arena",
        "two anonymous models side-by-side",
        "you must give response as two model",
    )
    if any(marker in lowered for marker in platform_prompt_markers):
        return False
    printable_ascii = sum(character.isascii() and character.isprintable() for character in text)
    return printable_ascii / len(text) >= 0.98


def build_lmsys(
    datasets_dir: Path,
    route: list[dict[str, Any]],
    catalog: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    source = next((datasets_dir / "lmsys_33k" / "data").glob("*.parquet"))
    records = pq.read_table(source).to_pylist()
    system = "Continue the ordinary chat conversation and answer the latest user message.\n"
    requests: list[dict[str, Any]] = []
    selected_ids: list[str] = []

    for record in records:
        if not lmsys_is_clean(record):
            continue
        winner = str(record["winner"])
        branch = "conversation_a" if winner == "model_a" else "conversation_b"
        winning_conversation = record.get(branch) or []
        if not conversation_is_mostly_ascii(winning_conversation):
            continue
        pairs = role_pairs(winning_conversation)
        if len(pairs) < 2:
            continue
        question_id = str(record["question_id"])
        winner_model = str(record["model_a"] if winner == "model_a" else record["model_b"])
        selected_ids.append(question_id)
        history = ""
        for turn_index, user_text, assistant_text in pairs:
            if len(requests) == REQUEST_COUNT:
                break
            request_id = len(requests)
            route_meta = route_metadata(route, catalog, request_id)
            segments = [{"type": "shared_system", "text": system}]
            if history:
                segments.append({"type": "conversation_history", "text": history})
            task = f"User: {user_text}\nAssistant:"
            requests.append(
                make_request(
                    request_id=request_id,
                    experiment="lmsys_ordinary_chat_continuous",
                    group_name="lmsys_lsapp_route",
                    context_id=f"lmsys_{question_id}",
                    source_dataset="LMSYS Chatbot Arena Conversations 33K",
                    segments=segments,
                    task=task,
                    reference_response=assistant_text,
                    lora_id=route_meta["lora_id"],
                    lora_name=route_meta["lora_name"],
                    adapter_path=route_meta["adapter_path"],
                    arrival_ms=route_meta["arrival_ms"],
                    logical_role="general_chat_baseline",
                    extra={
                        **route_meta,
                        "source_question_id": question_id,
                        "source_turn_index": turn_index,
                        "source_winner": winner,
                        "source_winner_model": winner_model,
                        "source_timestamp": record.get("tstamp"),
                        "source_moderation_flagged": False,
                    },
                )
            )
            history += f"User: {user_text}\nAssistant: {assistant_text}\n"
        if len(requests) == REQUEST_COUNT:
            break
    if len(requests) != REQUEST_COUNT:
        raise ValueError(f"LMSYS selection produced only {len(requests)} requests")

    reuse = add_continuous_reuse_metadata(requests)
    requested_loras = [int(request["lora_id"]) for request in requests]
    groups = build_lora_groups(
        "lmsys_lsapp_route",
        {lora_id: "general_chat_baseline" for lora_id in set(requested_loras)},
        catalog,
        requested_loras,
    )
    summary = {
        **WORKLOADS["lmsys_continuous"],
        "requests": len(requests),
        "source_file": str(source),
        "selected_source_ids": selected_ids,
        "selection": "clean English conversations using the winning model branch",
        "construction": "multiple independent growing chat contexts",
        "direct_lora_base_test5_compatible": False,
        **reuse,
    }
    return requests, [], summary, groups


def iter_translations(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    row_index = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["translation"], batch_size=65536):
        for translation in batch.column(0).to_pylist():
            yield row_index, translation or {}
            row_index += 1


def quality_english_source(text: str) -> bool:
    if not 50 <= len(text) <= 300 or "\n" in text or "http" in text.lower():
        return False
    if len(text.split()) < 8 or text.startswith(("-", "[", "(")):
        return False
    letters = sum(character.isalpha() for character in text)
    return letters / max(1, len(text)) >= 0.55


def select_opus_intersections(datasets_dir: Path) -> list[dict[str, Any]]:
    root = datasets_dir / "opus100"
    ja_path = root / "en-ja" / "train-00000-of-00001.parquet"
    japanese_by_english: dict[str, tuple[str, int]] = {}
    for row_index, translation in iter_translations(ja_path):
        english = clean_text(translation.get("en"))
        japanese = clean_text(translation.get("ja"))
        if quality_english_source(english) and japanese and japanese != english:
            japanese_by_english.setdefault(english, (japanese, row_index))
            if len(japanese_by_english) >= 200000:
                break

    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for language, config in (("de", "de-en"), ("es", "en-es"), ("fr", "en-fr"), ("ru", "en-ru"), ("zh", "en-zh")):
        path = root / config / "train-00000-of-00001.parquet"
        found = 0
        for row_index, translation in iter_translations(path):
            english = clean_text(translation.get("en"))
            target = clean_text(translation.get(language))
            if english not in japanese_by_english or english in used_sources:
                continue
            if not target or target == english or len(target) < 4:
                continue
            japanese, ja_row_index = japanese_by_english[english]
            selected.append(
                {
                    "english": english,
                    "ja": japanese,
                    language: target,
                    "other_language": language,
                    "other_config": config,
                    "ja_row_index": ja_row_index,
                    "other_row_index": row_index,
                }
            )
            used_sources.add(english)
            found += 1
            if found == 3:
                break
        if found != 3:
            raise ValueError(f"OPUS-100 found only {found}/3 en-ja/en-{language} intersections")
    return selected


def make_pair(
    context_id: str,
    group_name: str,
    anchor: dict[str, Any],
    child: dict[str, Any],
    pair_type: str,
) -> dict[str, Any]:
    return {
        "pair_name": f"{context_id}_lora_{anchor['lora_id']}_vs_{child['lora_id']}",
        "pair_type": pair_type,
        "group_name": group_name,
        "context_id": context_id,
        "common_prefix_hash": anchor["common_prefix_hash"],
        "anchor_request_id": anchor["request_id"],
        "child_request_id": child["request_id"],
        "anchor_lora_id": anchor["lora_id"],
        "child_lora_id": child["lora_id"],
    }


def build_opus100(
    datasets_dir: Path,
    catalog: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    language_loras = {"ja": 31, "de": 60, "es": 21, "fr": 18, "ru": 54, "zh": 24}
    language_names = {
        "ja": "Japanese",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "ru": "Russian",
        "zh": "Chinese",
    }
    selected = select_opus_intersections(datasets_dir)
    system = (
        "You are a multilingual translation assistant. Translate the supplied English "
        "source faithfully while preserving its meaning and tone.\n"
    )
    requests: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for pair_index, record in enumerate(selected):
        other_language = str(record["other_language"])
        context_id = f"opus100_shared_source_{pair_index:02d}"
        segments = [
            {"type": "shared_system", "text": system},
            {"type": "source_text", "text": f"English source:\n{record['english']}\n"},
        ]
        local: list[dict[str, Any]] = []
        for pair_role, language in (("anchor", "ja"), ("child", other_language)):
            request_id = len(requests)
            lora_id = language_loras[language]
            entry = catalog[lora_id]
            task = f"Translate the English source into {language_names[language]}.\nTranslation:"
            request = make_request(
                request_id=request_id,
                experiment="opus100_parallel_language_switch",
                group_name="opus100_language_loras",
                context_id=context_id,
                source_dataset="OPUS-100 train exact English intersections",
                segments=segments,
                task=task,
                reference_response=clean_text(record[language]),
                lora_id=lora_id,
                lora_name=str(entry["logical_name"]),
                adapter_path=str(entry["actual_adapter_path"]),
                arrival_ms=pair_index * 1000 + (1 if pair_role == "child" else 0),
                logical_role=f"translate_en_to_{language}",
                extra={
                    "pair_index": pair_index,
                    "pair_role": pair_role,
                    "source_language": "en",
                    "target_language": language,
                    "source_config": "en-ja" if language == "ja" else record["other_config"],
                    "source_row_index": record["ja_row_index"] if language == "ja" else record["other_row_index"],
                    "exact_shared_english_intersection": True,
                },
            )
            requests.append(request)
            local.append(request)
        pairs.append(
            make_pair(
                context_id,
                "opus100_language_loras",
                local[0],
                local[1],
                "opus100_same_english_different_target_language",
            )
        )

    requested_loras = [int(request["lora_id"]) for request in requests]
    groups = build_lora_groups(
        "opus100_language_loras",
        {lora_id: f"translate_en_to_{language}" for language, lora_id in language_loras.items()},
        catalog,
        requested_loras,
    )
    summary = {
        **WORKLOADS["opus100_parallel"],
        "requests": len(requests),
        "pairs": len(pairs),
        "contexts": len(selected),
        "context_reuses": len(requests) - len(selected),
        "request_level_context_reuse_rate": (len(requests) - len(selected)) / len(requests),
        "different_lora_pairs": len(pairs),
        "target_language_counts": dict(Counter(request["target_language"] for request in requests)),
        "selected_source_ids": [stable_hash(record["english"]) for record in selected],
        "construction": "15 exact English intersections, each translated by Japanese and one other language role",
        "direct_lora_base_test5_compatible": True,
    }
    return requests, pairs, summary, groups


def build_xsum(
    datasets_dir: Path,
    catalog: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    source = datasets_dir / "xsum_dataset"
    dataset = load_from_disk(str(source))
    selected: list[dict[str, str]] = []
    for record in dataset:
        document = clean_text(record["document"])
        summary = clean_text(record["summary"])
        if 1600 <= len(document) <= 4000 and 40 <= len(summary) <= 500:
            selected.append({"document": document, "summary": summary, "id": str(record["id"])})
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise ValueError(f"XSum selection produced only {len(selected)} documents")

    task_roles = [
        ("summarization", 31, "Write a one-sentence news summary of the article.\nSummary:"),
        ("main_event_qa", 60, "Answer the question: What is the main event described in this article?\nAnswer:"),
        ("headline_rewrite", 21, "Rewrite the article as a concise news headline.\nHeadline:"),
    ]
    system = "Read the following news article before carrying out the requested task.\n"
    requests: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for context_index, record in enumerate(selected):
        context_id = f"xsum_article_{record['id']}"
        segments = [
            {"type": "shared_system", "text": system},
            {"type": "source_article", "text": f"Article:\n{record['document']}\n"},
        ]
        local: list[dict[str, Any]] = []
        for role_index, (logical_role, lora_id, task) in enumerate(task_roles):
            entry = catalog[lora_id]
            request = make_request(
                request_id=len(requests),
                experiment="xsum_parallel_multi_task",
                group_name="xsum_task_loras",
                context_id=context_id,
                source_dataset="XSum train",
                segments=segments,
                task=task,
                reference_response=record["summary"],
                lora_id=lora_id,
                lora_name=str(entry["logical_name"]),
                adapter_path=str(entry["actual_adapter_path"]),
                arrival_ms=context_index * 1000 + role_index,
                logical_role=logical_role,
                extra={
                    "pair_role": "anchor" if role_index == 0 else "child",
                    "source_document_id": record["id"],
                    "source_document_characters": len(record["document"]),
                    "reference_usage": (
                        "gold XSum summary" if logical_role == "summarization" else "gold XSum summary used as a proxy target"
                    ),
                },
            )
            requests.append(request)
            local.append(request)
        for child in local[1:]:
            pairs.append(
                make_pair(
                    context_id,
                    "xsum_task_loras",
                    local[0],
                    child,
                    "xsum_same_article_different_task",
                )
            )

    requested_loras = [int(request["lora_id"]) for request in requests]
    groups = build_lora_groups(
        "xsum_task_loras",
        {lora_id: role for role, lora_id, _ in task_roles},
        catalog,
        requested_loras,
    )
    summary = {
        **WORKLOADS["xsum_parallel"],
        "requests": len(requests),
        "pairs": len(pairs),
        "contexts": len(selected),
        "context_reuses": len(requests) - len(selected),
        "request_level_context_reuse_rate": (len(requests) - len(selected)) / len(requests),
        "different_lora_pairs": len(pairs),
        "logical_task_counts": dict(Counter(request["logical_lora_role"] for request in requests)),
        "selected_source_ids": [record["id"] for record in selected],
        "construction": "10 articles, each dispatched to three logical task LoRAs",
        "reference_note": "QA and headline requests use the real XSum summary as a proxy reference, not a task-specific gold label.",
        "direct_lora_base_test5_compatible": True,
    }
    return requests, pairs, summary, groups


def validate_workload(
    name: str,
    requests: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(requests) != REQUEST_COUNT:
        errors.append(f"request count is {len(requests)}")
    if [request["request_id"] for request in requests] != list(range(REQUEST_COUNT)):
        errors.append("request IDs are not contiguous")
    for request in requests:
        prefix = request["common_prefix_text"]
        if request["prompt"] != prefix + request["task"]:
            errors.append(f"request {request['request_id']} prompt mismatch")
        if stable_hash(prefix) != request["common_prefix_hash"]:
            errors.append(f"request {request['request_id']} prefix hash mismatch")
        if "".join(segment["text"] for segment in request["prefix_segments"]) != prefix:
            errors.append(f"request {request['request_id']} segment mismatch")
        if not clean_text(request.get("reference_response")):
            errors.append(f"request {request['request_id']} has no real reference")

    prefix_growth_validated = WORKLOADS[name]["form"] == "continuous"
    if prefix_growth_validated:
        previous_by_context: dict[str, dict[str, Any]] = {}
        for request in requests:
            context_id = str(request["context_id"])
            previous = previous_by_context.get(context_id)
            if previous and not request["common_prefix_text"].startswith(previous["prompt"]):
                errors.append(
                    f"request {request['request_id']} is not an append-only extension "
                    f"of request {previous['request_id']}"
                )
            previous_by_context[context_id] = request

    by_id = {int(request["request_id"]): request for request in requests}
    for pair in pairs:
        anchor = by_id[int(pair["anchor_request_id"])]
        child = by_id[int(pair["child_request_id"])]
        if anchor["common_prefix_text"] != child["common_prefix_text"]:
            errors.append(f"pair {pair['pair_name']} prefix mismatch")
        if int(anchor["lora_id"]) == int(child["lora_id"]):
            errors.append(f"pair {pair['pair_name']} uses the same LoRA")

    configured_ids = {
        int(entry["lora_id"])
        for entries in groups.values()
        for entry in entries
    }
    requested_ids = {int(request["lora_id"]) for request in requests}
    if configured_ids != requested_ids:
        errors.append("lora_groups.json does not exactly cover requested LoRAs")
    missing_paths = [
        str(entry.get("actual_adapter_path", ""))
        for entries in groups.values()
        for entry in entries
        if not Path(str(entry.get("actual_adapter_path", ""))).is_file()
    ]
    if missing_paths:
        errors.append(f"{len(missing_paths)} physical adapter paths do not exist")
    if errors:
        raise ValueError(f"{name} validation failed: " + "; ".join(errors[:10]))
    return {
        "status": "ok",
        "requests": len(requests),
        "pairs": len(pairs),
        "configured_loras": len(configured_ids),
        "adapter_paths_present": not missing_paths,
        "real_references_present": True,
        "prompt_prefix_consistency": True,
        "append_only_prefix_growth": prefix_growth_validated,
    }


def write_workload(
    output_dir: Path,
    name: str,
    requests: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    summary: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    validation = validate_workload(name, requests, pairs, groups)
    workload_dir = output_dir / name
    write_json(workload_dir / "lora_groups.json", groups)
    write_jsonl(workload_dir / "grouped" / "grouped_requests.jsonl", requests)
    if WORKLOADS[name]["form"] == "parallel":
        write_jsonl(workload_dir / "delta" / "delta_requests.jsonl", requests)
        write_jsonl(workload_dir / "delta" / "delta_pairs.jsonl", pairs)
    else:
        write_jsonl(workload_dir / "delta" / "delta_requests.jsonl", [])
        write_jsonl(workload_dir / "delta" / "delta_pairs.jsonl", [])
    final_summary = {
        **summary,
        "validation": validation,
        "physical_adapter_notice": PHYSICAL_ADAPTER_NOTICE,
    }
    write_json(workload_dir / "summary.json", final_summary)
    return final_summary


def preview_excerpt(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def write_preview(
    output_dir: Path,
    summaries: dict[str, dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Real 30-request workload preview",
        "",
        "All source prompts and reference responses come from the local dataset copies. ",
        "Task instructions are constructed for the experiment. Physical LoRAs are routing artifacts, ",
        "not semantically matched trained adapters.",
        "",
        "| Workload | Form | Requests | Contexts | Pairs | Focus |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| `{name}` | {summary['form']} | {summary['requests']} | "
            f"{summary.get('contexts', 0)} | {summary.get('pairs', 0)} | {summary['focus']} |"
        )
    lines.extend(["", "## First request from each workload", ""])
    for name, requests in samples.items():
        request = requests[0]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Source/context: `{request['context_id']}`",
                f"- Logical role: `{request['logical_lora_role']}`; physical LoRA: `{request['lora_id']}`",
                f"- Task: {preview_excerpt(request['task'])}",
                f"- Real reference: {preview_excerpt(request['reference_response'])}",
                "",
            ]
        )
    (output_dir / "preview.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--route-jsonl", type=Path, default=DEFAULT_ROUTE_JSONL)
    parser.add_argument("--lora-groups-json", type=Path, default=DEFAULT_LORA_GROUPS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route, catalog = load_adapter_inputs(args.route_jsonl, args.lora_groups_json)
    builders = [
        ("msc_continuous", lambda: build_msc(args.datasets_dir, route, catalog)),
        ("taskmaster_continuous", lambda: build_taskmaster(args.datasets_dir, route, catalog)),
        ("lmsys_continuous", lambda: build_lmsys(args.datasets_dir, route, catalog)),
        ("opus100_parallel", lambda: build_opus100(args.datasets_dir, catalog)),
        ("xsum_parallel", lambda: build_xsum(args.datasets_dir, catalog)),
    ]
    summaries: dict[str, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for name, builder in builders:
        print(f"Building {name}...", flush=True)
        requests, pairs, summary, groups = builder()
        summaries[name] = write_workload(
            args.output_dir, name, requests, pairs, summary, groups
        )
        samples[name] = requests

    root_summary = {
        "request_count_per_workload": REQUEST_COUNT,
        "workloads": summaries,
        "all_valid": all(summary["validation"]["status"] == "ok" for summary in summaries.values()),
        "physical_adapter_notice": PHYSICAL_ADAPTER_NOTICE,
    }
    write_json(args.output_dir / "summary.json", root_summary)
    write_preview(args.output_dir, summaries, samples)
    print(json.dumps(root_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
