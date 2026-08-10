#!/usr/bin/env python3
"""Build small LSApp-derived workloads for local LoRA reuse experiments.

The output contains two constructions:

* continuous: one session whose prefix grows with assistant history;
* parallel: pairs of different LoRAs that receive the exact same prefix.

The parallel workload follows the JSONL schema consumed by lora-base-test5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LSAPP_CSV = Path(r"D:\ecnu_experiment\datasets_lastest\lsapp\df_start.csv")
DEFAULT_MAPPING_JSON = Path(
    r"D:\ecnu_experiment\datasets\mobilora_workloads_87_original\app_lora_mapping.json"
)
DEFAULT_LORA_GROUPS_JSON = Path(
    r"D:\ecnu_experiment\datasets\mobilora_workloads_87_original\lora_groups.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "lsapp_30"

SYSTEM_PREFIX = (
    "You are an on-device assistant.\n"
    "Use the shared user and device context before handling the current application task.\n"
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimated_tokens(text: str) -> int:
    ascii_count = sum(ord(char) < 128 for char in text)
    return max(1, math.ceil(ascii_count / 4 + (len(text) - ascii_count) / 1.5))


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


def process_long_duration_apps(
    frame: pd.DataFrame,
    time_step_min: int = 5,
    max_duration_min: int = 200,
) -> pd.DataFrame:
    """Expand a long foreground interval into fixed-duration request slices."""
    rows: list[pd.Series] = []
    frame = frame.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], errors="coerce")

    for _, row in frame.iterrows():
        start_time = row["open_time"]
        end_time = row["close_time"]
        duration_min = (
            float("nan")
            if pd.isna(start_time) or pd.isna(end_time)
            else (end_time - start_time).total_seconds() / 60.0
        )

        def append_slice(
            current_start: pd.Timestamp | Any,
            current_end: pd.Timestamp | Any,
            slice_index: int,
        ) -> None:
            copied = row.copy()
            copied["source_open_time"] = start_time
            copied["source_close_time"] = end_time
            copied["source_duration_min"] = duration_min
            copied["open_time"] = current_start
            copied["close_time"] = current_end
            if not pd.isna(current_start):
                copied["timestamp"] = current_start
            copied["slice_idx"] = slice_index
            copied["is_slice"] = slice_index > 0
            copied["slice_duration_min"] = (
                float("nan")
                if pd.isna(current_start) or pd.isna(current_end)
                else max(0.0, (current_end - current_start).total_seconds() / 60.0)
            )
            rows.append(copied)

        if math.isnan(duration_min) or duration_min <= time_step_min:
            append_slice(start_time, end_time, 0)
            continue

        duration_limit = min(duration_min, max_duration_min)
        real_end_limit = start_time + pd.Timedelta(minutes=duration_limit)
        current_start = start_time
        slice_index = 0
        while current_start < real_end_limit:
            current_end = min(
                current_start + pd.Timedelta(minutes=time_step_min),
                real_end_limit,
            )
            append_slice(current_start, current_end, slice_index)
            current_start = current_end
            slice_index += 1

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["timestamp", "interaction_id", "slice_idx"], kind="stable"
    ).reset_index(drop=True)


def duration_slice_counts(
    frame: pd.DataFrame,
    time_step_min: int,
    max_duration_min: int,
) -> pd.Series:
    duration = (frame["close_time"] - frame["open_time"]).dt.total_seconds() / 60.0
    long_duration = duration.clip(upper=max_duration_min)
    counts = long_duration.map(
        lambda value: math.ceil(value / time_step_min) if pd.notna(value) else 1
    )
    return counts.where(duration > time_step_min, 1).clip(lower=1).astype(int)


def load_source_events(path: Path, known_apps: set[str]) -> pd.DataFrame:
    columns = [
        "interaction_id",
        "user_id",
        "session_id",
        "timestamp",
        "app_name",
        "event_type",
        "open_time",
        "close_time",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame["app_name"] = frame["app_name"].astype(str).str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], errors="coerce")
    frame = frame[
        frame["app_name"].isin(known_apps)
        & frame["event_type"].isin(["Opened", "User Interaction"])
    ].dropna(subset=["timestamp", "user_id", "session_id"])
    return frame.sort_values(
        ["user_id", "session_id", "timestamp", "interaction_id"], kind="stable"
    ).reset_index(drop=True)


def select_trace(
    events: pd.DataFrame,
    request_count: int,
    time_step_min: int,
    max_duration_min: int,
    user_id: int | None,
    session_id: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    events = events.copy()
    events["_slice_count"] = duration_slice_counts(
        events, time_step_min, max_duration_min
    )

    if (user_id is None) != (session_id is None):
        raise ValueError("--user-id and --session-id must be supplied together")

    if user_id is not None and session_id is not None:
        selected = events[
            (events["user_id"] == user_id) & (events["session_id"] == session_id)
        ]
        if selected.empty:
            raise ValueError(f"no LSApp rows for user={user_id}, session={session_id}")
        raw_rows = len(selected)
        app_count = int(selected["app_name"].nunique())
        max_repeat = int(selected["_slice_count"].max())
        expanded_rows = int(selected["_slice_count"].sum())
    else:
        stats = events.groupby(["user_id", "session_id"], sort=False).agg(
            expanded_rows=("_slice_count", "sum"),
            raw_rows=("app_name", "size"),
            app_count=("app_name", "nunique"),
            max_repeat=("_slice_count", "max"),
        )
        candidates = stats[
            (stats["expanded_rows"] >= request_count)
            & (stats["app_count"] >= 2)
        ].copy()
        if candidates.empty:
            raise ValueError(f"no LSApp session can provide {request_count} requests")

        preferred = candidates[
            (candidates["app_count"] >= min(4, request_count))
            & (candidates["max_repeat"] >= 2)
        ]
        if not preferred.empty:
            candidates = preferred.copy()

        target_repeat = min(5, max(2, request_count // 6))
        candidates["overflow"] = candidates["expanded_rows"] - request_count
        candidates["repeat_distance"] = (
            candidates["max_repeat"] - target_repeat
        ).abs()
        candidates = candidates.reset_index().sort_values(
            [
                "overflow",
                "repeat_distance",
                "app_count",
                "raw_rows",
                "user_id",
                "session_id",
            ],
            ascending=[True, True, False, False, True, True],
            kind="stable",
        )
        best = candidates.iloc[0]
        user_id = int(best["user_id"])
        session_id = int(best["session_id"])
        raw_rows = int(best["raw_rows"])
        app_count = int(best["app_count"])
        max_repeat = int(best["max_repeat"])
        expanded_rows = int(best["expanded_rows"])
        selected = events[
            (events["user_id"] == user_id) & (events["session_id"] == session_id)
        ]

    expanded = process_long_duration_apps(
        selected.drop(columns=["_slice_count"]),
        time_step_min=time_step_min,
        max_duration_min=max_duration_min,
    )
    if len(expanded) < request_count:
        raise ValueError(
            f"selected session expands to {len(expanded)} rows, fewer than {request_count}"
        )

    trace = expanded.head(request_count).copy()
    trace["trace_index"] = range(len(trace))
    selection = {
        "user_id": int(user_id),
        "session_id": int(session_id),
        "raw_session_rows": raw_rows,
        "expanded_session_rows": expanded_rows,
        "selected_rows": len(trace),
        "session_app_count": app_count,
        "max_duration_slices_in_source_row": max_repeat,
    }
    return trace, selection


def load_metadata(
    mapping_path: Path,
    groups_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]], dict[int, str]]:
    mapping_rows = json.loads(mapping_path.read_text(encoding="utf-8"))
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    mapping_by_app = {
        str(row["app_name"]).strip(): row
        for row in mapping_rows
    }
    entry_by_lora: dict[int, dict[str, Any]] = {}
    source_group_by_lora: dict[int, str] = {}
    for group_name, entries in groups.items():
        for entry in entries:
            lora_id = int(entry["lora_id"])
            entry_by_lora[lora_id] = entry
            source_group_by_lora[lora_id] = group_name
    return mapping_by_app, entry_by_lora, source_group_by_lora


def build_trace_group(
    trace: pd.DataFrame,
    mapping_by_app: dict[str, dict[str, Any]],
    entry_by_lora: dict[int, dict[str, Any]],
    source_group_by_lora: dict[int, str],
    group_name: str,
) -> dict[str, list[dict[str, Any]]]:
    lora_sequence = [
        int(mapping_by_app[str(app_name)]["lora_id"])
        for app_name in trace["app_name"]
    ]
    anchor_lora_id = Counter(lora_sequence).most_common(1)[0][0]
    entries: list[dict[str, Any]] = []
    for lora_id in sorted(set(lora_sequence)):
        if lora_id not in entry_by_lora:
            raise ValueError(f"LoRA {lora_id} is missing from the source lora_groups.json")
        entry = dict(entry_by_lora[lora_id])
        entry["source_group_name"] = source_group_by_lora[lora_id]
        entry["is_anchor"] = lora_id == anchor_lora_id
        entries.append(entry)
    return {group_name: entries}


def iso_timestamp(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def finite_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def make_task(app_name: str, lora_entry: dict[str, Any], turn_index: int) -> str:
    task = str(
        lora_entry.get(
            "task",
            "Infer the user's current activity or context and provide assistance.",
        )
    )
    return f"Task turn {turn_index}: App: {app_name}. {task}\n"


def common_request_fields(
    row: pd.Series,
    mapping: dict[str, Any],
    lora_entry: dict[str, Any],
    source_group_name: str,
) -> dict[str, Any]:
    return {
        "user_id": int(row["user_id"]),
        "session_id": int(row["session_id"]),
        "app_name": str(row["app_name"]),
        "lora_id": int(mapping["lora_id"]),
        "lora_name": str(mapping["lora_name"]),
        "source_group_name": source_group_name,
        "source_interaction_id": int(row["interaction_id"]),
        "source_trace_index": int(row["trace_index"]),
        "source_timestamp": iso_timestamp(row["timestamp"]),
        "source_open_time": iso_timestamp(row["source_open_time"]),
        "source_close_time": iso_timestamp(row["source_close_time"]),
        "slice_idx": int(row["slice_idx"]),
        "is_duration_slice": bool(row["is_slice"]),
        "source_duration_min": finite_float(row["source_duration_min"]),
        "adapter_path": str(lora_entry.get("actual_adapter_path", "")),
    }


def build_continuous_requests(
    trace: pd.DataFrame,
    mapping_by_app: dict[str, dict[str, Any]],
    entry_by_lora: dict[int, dict[str, Any]],
    source_group_by_lora: dict[int, str],
    group_name: str,
    chunk_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_timestamp = pd.Timestamp(trace.iloc[0]["timestamp"])
    user_id = int(trace.iloc[0]["user_id"])
    session_id = int(trace.iloc[0]["session_id"])
    context_id = f"lsapp_u{user_id}_s{session_id}_continuous"
    session_context = (
        "LSApp session context:\n"
        f"User {user_id}, session {session_id}. Requests follow the observed foreground "
        "application path; prior assistant results remain in the conversation context.\n"
    )
    history: list[str] = []
    requests: list[dict[str, Any]] = []
    previous_lora_id: int | None = None
    previous_prefix_hash = ""

    for request_id, (_, row) in enumerate(trace.iterrows()):
        app_name = str(row["app_name"])
        mapping = mapping_by_app[app_name]
        lora_id = int(mapping["lora_id"])
        lora_entry = entry_by_lora[lora_id]
        segments = [
            {"type": "shared_system", "text": SYSTEM_PREFIX},
            {"type": "user_context", "text": session_context},
        ]
        if history:
            segments.append(
                {"type": "assistant_history", "text": "".join(history)}
            )
        common_prefix = "".join(segment["text"] for segment in segments)
        prefix_hash = stable_hash(common_prefix)
        task = make_task(app_name, lora_entry, request_id)
        arrival_ms = max(
            0,
            int((pd.Timestamp(row["timestamp"]) - first_timestamp).total_seconds() * 1000),
        )
        request = {
            "request_id": request_id,
            "experiment": "lsapp_continuous_dialogue",
            "arrival_ms": arrival_ms,
            "original_timestamp": iso_timestamp(row["timestamp"]),
            "group_name": group_name,
            "context_id": context_id,
            "source_dataset": "lsapp_df_start_5min_sliced",
            "common_prefix_text": common_prefix,
            "common_prefix_hash": prefix_hash,
            "parent_prefix_hash": previous_prefix_hash,
            "prompt": common_prefix + task,
            "task": task,
            "chunk_token_size": chunk_tokens,
            "prefix_segment_types": [segment["type"] for segment in segments],
            "prefix_segments": segments,
            "estimated_common_prefix_tokens": estimated_tokens(common_prefix),
            "turn_index": request_id,
            "previous_lora_id": previous_lora_id,
            "same_lora_as_previous": (
                previous_lora_id is not None and previous_lora_id == lora_id
            ),
        }
        request.update(
            common_request_fields(
                row,
                mapping,
                lora_entry,
                source_group_by_lora[lora_id],
            )
        )
        requests.append(request)
        history.append(
            f"Assistant turn {request_id}: The {app_name} request was completed "
            f"with LoRA {lora_id}.\n"
        )
        previous_lora_id = lora_id
        previous_prefix_hash = prefix_hash

    same_lora_transitions = sum(
        bool(request["same_lora_as_previous"]) for request in requests
    )
    transition_count = max(0, len(requests) - 1)
    summary = {
        "construction": "continuous_dialogue_growing_prefix",
        "requests": len(requests),
        "contexts": 1,
        "lora_transitions": transition_count,
        "same_lora_transitions": same_lora_transitions,
        "same_lora_transition_rate": (
            same_lora_transitions / transition_count if transition_count else 0.0
        ),
        "direct_lora_base_test5_compatible": False,
        "compatibility_note": (
            "The common prefix grows after every turn, while lora-base-test5 currently "
            "requires an anchor and child with identical common_prefix_text."
        ),
    }
    return requests, summary


def different_lora_transitions(
    trace: pd.DataFrame,
    mapping_by_app: dict[str, dict[str, Any]],
    pair_count: int,
) -> list[tuple[pd.Series, pd.Series]]:
    rows = [row for _, row in trace.iterrows()]
    pairs: list[tuple[pd.Series, pd.Series]] = []
    used_indices: set[tuple[int, int]] = set()

    for offset in range(1, len(rows)):
        for left_index in range(len(rows) - offset):
            right_index = left_index + offset
            left = rows[left_index]
            right = rows[right_index]
            left_lora = int(mapping_by_app[str(left["app_name"])]["lora_id"])
            right_lora = int(mapping_by_app[str(right["app_name"])]["lora_id"])
            if left_lora == right_lora:
                continue
            key = (int(left["trace_index"]), int(right["trace_index"]))
            if key in used_indices:
                continue
            pairs.append((left, right))
            used_indices.add(key)
            if len(pairs) == pair_count:
                return pairs

    raise ValueError(
        f"the selected trace provides only {len(pairs)} different-LoRA pairs; "
        f"{pair_count} are required"
    )


def build_parallel_requests(
    trace: pd.DataFrame,
    mapping_by_app: dict[str, dict[str, Any]],
    entry_by_lora: dict[int, dict[str, Any]],
    source_group_by_lora: dict[int, str],
    group_name: str,
    request_count: int,
    chunk_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_count = request_count // 2
    transitions = different_lora_transitions(trace, mapping_by_app, pair_count)
    first_timestamp = pd.Timestamp(trace.iloc[0]["timestamp"])
    user_id = int(trace.iloc[0]["user_id"])
    session_id = int(trace.iloc[0]["session_id"])
    requests: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for pair_index, (left, right) in enumerate(transitions):
        context_id = f"lsapp_u{user_id}_s{session_id}_parallel_{pair_index:02d}"
        context = (
            "Parallel LSApp task context:\n"
            f"User {user_id}, session {session_id}, observed transition window "
            f"{int(left['trace_index'])}->{int(right['trace_index'])} from "
            f"{iso_timestamp(left['timestamp'])} to {iso_timestamp(right['timestamp'])}. "
            "The same device observation must be handled by two specialized adapters.\n"
        )
        segments = [
            {"type": "shared_system", "text": SYSTEM_PREFIX},
            {"type": "user_context", "text": context},
        ]
        common_prefix = "".join(segment["text"] for segment in segments)
        prefix_hash = stable_hash(common_prefix)
        local_request_ids: list[int] = []

        for pair_role, row in (("anchor", left), ("child", right)):
            request_id = len(requests)
            app_name = str(row["app_name"])
            mapping = mapping_by_app[app_name]
            lora_id = int(mapping["lora_id"])
            lora_entry = entry_by_lora[lora_id]
            task = make_task(app_name, lora_entry, pair_index)
            base_arrival_ms = max(
                0,
                int(
                    (pd.Timestamp(left["timestamp"]) - first_timestamp).total_seconds()
                    * 1000
                ),
            )
            request = {
                "request_id": request_id,
                "experiment": "lsapp_parallel_same_context",
                "arrival_ms": base_arrival_ms + (1 if pair_role == "child" else 0),
                "original_timestamp": iso_timestamp(row["timestamp"]),
                "group_name": group_name,
                "context_id": context_id,
                "source_dataset": "lsapp_df_start_5min_sliced",
                "common_prefix_text": common_prefix,
                "common_prefix_hash": prefix_hash,
                "prompt": common_prefix + task,
                "task": task,
                "chunk_token_size": chunk_tokens,
                "prefix_segment_types": [segment["type"] for segment in segments],
                "prefix_segments": segments,
                "estimated_common_prefix_tokens": estimated_tokens(common_prefix),
                "pair_index": pair_index,
                "pair_role": pair_role,
            }
            request.update(
                common_request_fields(
                    row,
                    mapping,
                    lora_entry,
                    source_group_by_lora[lora_id],
                )
            )
            requests.append(request)
            local_request_ids.append(request_id)

        anchor_request = requests[local_request_ids[0]]
        child_request = requests[local_request_ids[1]]
        pairs.append(
            {
                "pair_name": (
                    f"{context_id}_lora_{anchor_request['lora_id']}_vs_"
                    f"{child_request['lora_id']}"
                ),
                "pair_type": "lsapp_parallel_intra_trace",
                "group_name": group_name,
                "context_id": context_id,
                "common_prefix_hash": prefix_hash,
                "anchor_request_id": anchor_request["request_id"],
                "child_request_id": child_request["request_id"],
                "anchor_lora_id": anchor_request["lora_id"],
                "child_lora_id": child_request["lora_id"],
                "source_trace_indices": [
                    anchor_request["source_trace_index"],
                    child_request["source_trace_index"],
                ],
            }
        )

    unique_contexts = len({request["context_id"] for request in requests})
    context_reuses = len(requests) - unique_contexts
    summary = {
        "construction": "parallel_two_loras_same_prefix",
        "requests": len(requests),
        "pairs": len(pairs),
        "contexts": unique_contexts,
        "context_reuses": context_reuses,
        "request_level_context_reuse_rate": (
            context_reuses / len(requests) if requests else 0.0
        ),
        "different_lora_pairs": sum(
            pair["anchor_lora_id"] != pair["child_lora_id"] for pair in pairs
        ),
        "direct_lora_base_test5_compatible": True,
    }
    return requests, pairs, summary


def source_trace_rows(
    trace: pd.DataFrame,
    mapping_by_app: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in trace.iterrows():
        mapping = mapping_by_app[str(row["app_name"])]
        rows.append(
            {
                "trace_index": int(row["trace_index"]),
                "interaction_id": int(row["interaction_id"]),
                "user_id": int(row["user_id"]),
                "session_id": int(row["session_id"]),
                "timestamp": iso_timestamp(row["timestamp"]),
                "app_name": str(row["app_name"]),
                "lora_id": int(mapping["lora_id"]),
                "lora_name": str(mapping["lora_name"]),
                "open_time": iso_timestamp(row["open_time"]),
                "close_time": iso_timestamp(row["close_time"]),
                "source_open_time": iso_timestamp(row["source_open_time"]),
                "source_close_time": iso_timestamp(row["source_close_time"]),
                "source_duration_min": finite_float(row["source_duration_min"]),
                "slice_duration_min": finite_float(row["slice_duration_min"]),
                "slice_idx": int(row["slice_idx"]),
                "is_duration_slice": bool(row["is_slice"]),
            }
        )
    return rows


def validate_outputs(
    request_count: int,
    lora_groups: dict[str, list[dict[str, Any]]],
    continuous: list[dict[str, Any]],
    parallel: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(continuous) != request_count:
        errors.append(f"continuous request count is {len(continuous)}")
    if len(parallel) != request_count:
        errors.append(f"parallel request count is {len(parallel)}")
    if len(pairs) != request_count // 2:
        errors.append(f"parallel pair count is {len(pairs)}")

    for label, requests in (("continuous", continuous), ("parallel", parallel)):
        expected_ids = list(range(request_count))
        actual_ids = [int(request["request_id"]) for request in requests]
        if actual_ids != expected_ids:
            errors.append(f"{label} request IDs are not contiguous")
        for request in requests:
            if request["prompt"] != request["common_prefix_text"] + request["task"]:
                errors.append(f"{label} request {request['request_id']} prompt mismatch")
                break
            if stable_hash(request["common_prefix_text"]) != request["common_prefix_hash"]:
                errors.append(f"{label} request {request['request_id']} hash mismatch")
                break

    parallel_by_id = {int(request["request_id"]): request for request in parallel}
    for pair in pairs:
        anchor = parallel_by_id[int(pair["anchor_request_id"])]
        child = parallel_by_id[int(pair["child_request_id"])]
        if anchor["common_prefix_text"] != child["common_prefix_text"]:
            errors.append(f"pair {pair['pair_name']} does not share an exact prefix")
        if int(anchor["lora_id"]) == int(child["lora_id"]):
            errors.append(f"pair {pair['pair_name']} uses the same LoRA twice")

    configured_ids = {
        int(entry["lora_id"])
        for entries in lora_groups.values()
        for entry in entries
    }
    requested_ids = {
        int(request["lora_id"]) for request in continuous + parallel
    }
    if configured_ids != requested_ids:
        errors.append("lora_groups.json does not exactly cover requested LoRAs")

    missing_adapters = [
        str(entry.get("actual_adapter_path", ""))
        for entries in lora_groups.values()
        for entry in entries
        if not Path(str(entry.get("actual_adapter_path", ""))).is_file()
    ]
    if missing_adapters:
        errors.append(f"{len(missing_adapters)} configured adapter paths do not exist")

    if errors:
        raise ValueError("output validation failed: " + "; ".join(errors))
    return {
        "status": "ok",
        "configured_loras": len(configured_ids),
        "adapter_paths_present": len(missing_adapters) == 0,
        "continuous_requests": len(continuous),
        "parallel_requests": len(parallel),
        "parallel_pairs": len(pairs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lsapp-csv", type=Path, default=DEFAULT_LSAPP_CSV)
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--lora-groups-json", type=Path, default=DEFAULT_LORA_GROUPS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-count", type=int, default=30)
    parser.add_argument("--time-step-min", type=int, default=5)
    parser.add_argument("--max-duration-min", type=int, default=200)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--session-id", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request_count <= 0 or args.request_count % 2:
        raise ValueError("--request-count must be a positive even number")
    if args.time_step_min <= 0 or args.max_duration_min <= 0:
        raise ValueError("duration limits must be positive")

    mapping_by_app, entry_by_lora, source_group_by_lora = load_metadata(
        args.mapping_json,
        args.lora_groups_json,
    )
    events = load_source_events(args.lsapp_csv, set(mapping_by_app))
    trace, selection = select_trace(
        events,
        request_count=args.request_count,
        time_step_min=args.time_step_min,
        max_duration_min=args.max_duration_min,
        user_id=args.user_id,
        session_id=args.session_id,
    )
    group_name = (
        f"lsapp_trace_u{selection['user_id']}_s{selection['session_id']}"
    )
    lora_groups = build_trace_group(
        trace,
        mapping_by_app,
        entry_by_lora,
        source_group_by_lora,
        group_name,
    )
    continuous, continuous_summary = build_continuous_requests(
        trace,
        mapping_by_app,
        entry_by_lora,
        source_group_by_lora,
        group_name,
        args.chunk_tokens,
    )
    parallel, pairs, parallel_summary = build_parallel_requests(
        trace,
        mapping_by_app,
        entry_by_lora,
        source_group_by_lora,
        group_name,
        args.request_count,
        args.chunk_tokens,
    )
    validation = validate_outputs(
        args.request_count,
        lora_groups,
        continuous,
        parallel,
        pairs,
    )

    output_dir = args.output_dir
    write_jsonl(
        output_dir / f"source_trace_{args.request_count}.jsonl",
        source_trace_rows(trace, mapping_by_app),
    )
    for construction in ("continuous", "parallel"):
        write_json(output_dir / construction / "lora_groups.json", lora_groups)
    write_jsonl(
        output_dir / "continuous" / "grouped" / "grouped_requests.jsonl",
        continuous,
    )
    write_jsonl(output_dir / "continuous" / "delta" / "delta_requests.jsonl", [])
    write_jsonl(output_dir / "continuous" / "delta" / "delta_pairs.jsonl", [])
    write_json(output_dir / "continuous" / "summary.json", continuous_summary)
    write_jsonl(
        output_dir / "parallel" / "delta" / "delta_requests.jsonl",
        parallel,
    )
    write_jsonl(
        output_dir / "parallel" / "delta" / "delta_pairs.jsonl",
        pairs,
    )
    write_jsonl(
        output_dir / "parallel" / "grouped" / "grouped_requests.jsonl",
        parallel,
    )
    write_json(output_dir / "parallel" / "summary.json", parallel_summary)

    summary = {
        "source": {
            "lsapp_csv": str(args.lsapp_csv),
            "mapping_json": str(args.mapping_json),
            "lora_groups_json": str(args.lora_groups_json),
        },
        "time_step_min": args.time_step_min,
        "max_duration_min": args.max_duration_min,
        "request_count_per_construction": args.request_count,
        "selection": selection,
        "trace_app_sequence": trace["app_name"].astype(str).tolist(),
        "trace_lora_sequence": [
            int(mapping_by_app[str(app_name)]["lora_id"])
            for app_name in trace["app_name"]
        ],
        "continuous": continuous_summary,
        "parallel": parallel_summary,
        "validation": validation,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"output_dir={output_dir}")
    print(
        f"selected user={selection['user_id']} session={selection['session_id']} "
        f"requests={args.request_count}"
    )
    print(
        f"continuous same_lora_rate={continuous_summary['same_lora_transition_rate']:.4f}"
    )
    print(
        "parallel "
        f"pairs={parallel_summary['pairs']} "
        f"context_reuse_rate={parallel_summary['request_level_context_reuse_rate']:.4f}"
    )
    print("validation=ok")


if __name__ == "__main__":
    main()
