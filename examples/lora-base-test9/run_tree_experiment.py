#!/usr/bin/env python3
"""Small, reproducible comparison of LoRA prefix-tree storage policies.

The script intentionally does not load a language model.  It evaluates the
part of the proposed design that is independent of logits: root-to-leaf
matching, incremental branch creation, and the unit used for eviction.

Compared policies:
  flat_leaf_lru       Store one complete prefix per LoRA and evict leaves.
  trie_leaf_lru       Build a chunked trie incrementally, evict individual leaves.
  trie_sequence_lru   Build the same trie, but evict conversation x LoRA paths
                      atomically (the edge-device proposal).

The workload contains a deterministic WorkBuddy-style return-after-a-gap trace,
two server-style shared-template traces, and optional real JSONL workloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Request:
    request_id: int
    workload: str
    user_id: str
    conversation_id: str
    lora_id: str
    prefix: tuple[str, ...]
    arrival_ms: int

    @property
    def sequence_id(self) -> str:
        return f"{self.conversation_id}:{self.lora_id}"


@dataclass
class TraceRow:
    workload: str
    strategy: str
    request_id: int
    sequence_id: str
    lora_id: str
    prefix_tokens: int
    reused_tokens: int
    hit: int
    complete_hit: int
    created_nodes: int
    evictions: int
    cache_nodes: int
    active_sequences: int
    walk_steps: int


@dataclass
class Summary:
    workload: str
    strategy: str
    requests: int
    request_hit_rate: float
    complete_hit_rate: float
    token_coverage: float
    mean_reused_tokens: float
    peak_nodes: int
    peak_active_sequences: int
    evictions: int
    mean_walk_steps: float
    storage_node_visits: int


@dataclass
class ChunkingRow:
    workload: str
    text_tokens: int
    chunks: int
    split_ms: float
    inference_ms: float
    serial_ttft_ms: float
    parallel_ttft_ms: float
    overlap_saved_ms: float
    split_hint_hit: int


@dataclass
class TrieNode:
    node_id: int
    parent: int | None
    token: str
    depth: int
    children: dict[str, int] = field(default_factory=dict)
    owners: set[str] = field(default_factory=set)
    last_access: int = -1


def proxy_tokens(text: str) -> tuple[str, ...]:
    """Stable text tokens for the storage experiment, not a model tokenizer."""
    # Keep Chinese characters separate so the proxy reflects token-count growth.
    pieces = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\w\s]", text.lower(), flags=re.UNICODE)
    return tuple(pieces)


def make_request(
    workload: str,
    request_id: int,
    user: str,
    conversation: str,
    lora: str,
    prefix: Sequence[str],
    arrival_ms: int,
) -> Request:
    return Request(
        request_id,
        workload,
        user,
        conversation,
        lora,
        tuple(prefix),
        arrival_ms,
    )


def synthetic_workloads() -> dict[str, list[Request]]:
    """Construct traces matching the four design questions in the feedback."""
    workbuddy: list[Request] = []
    system = ("workbuddy", "assistant", "remember", "the", "trip")
    profile_a = ("user", "alice", "home", "shanghai", "preference", "window")
    profile_b = ("user", "bob", "home", "hangzhou", "preference", "aisle")
    trip = ("trip", "beijing", "outbound", "friday")
    requests = [
        ("alice", "conv-a", "L1", system + profile_a + trip + ("outbound_date",)),
        ("alice", "conv-a", "L2", system + profile_a + trip + ("outbound_date",)),
        ("alice", "conv-a", "L1", system + profile_a + trip + ("outbound_date", "return_date")),
        ("bob", "conv-b", "L1", system + profile_b + ("trip", "hangzhou", "outbound", "monday")),
        ("alice", "conv-a", "L1", system + profile_a + trip + ("outbound_date", "return_date")),
        ("alice", "conv-a", "L2", system + profile_a + trip + ("outbound_date", "return_date")),
        ("bob", "conv-b", "L2", system + profile_b + ("trip", "hangzhou", "outbound", "monday")),
        ("alice", "conv-a", "L1", system + profile_a + trip + ("outbound_date", "return_date", "seat")),
        ("alice", "conv-a", "L2", system + profile_a + trip + ("outbound_date", "return_date", "seat")),
        ("alice", "conv-a", "L1", system + profile_a + trip + ("outbound_date", "return_date")),
    ]
    for i, (user, conv, lora, prefix) in enumerate(requests):
        workbuddy.append(make_request("workbuddy_return", i, user, conv, lora, prefix, i * 300_000))

    server: list[Request] = []
    shared = ("booking", "workflow", "station", "date", "passenger", "policy")
    users = [("u1", "beijing", "L1"), ("u2", "shanghai", "L1"), ("u3", "beijing", "L2"), ("u1", "beijing", "L2")]
    for i in range(16):
        user, city, lora = users[i % len(users)]
        prefix = shared + ("origin", city, "day", str(10 + (i % 3)), "traveler", user)
        server.append(make_request("server_shared_template", i, user, f"session-{user}", lora, prefix, i * 10_000))

    long_doc: list[Request] = []
    article = ("article", "climate", "policy", "energy", "transition", "evidence", "summary")
    for i in range(12):
        length = 4 + (i % 4) * 2
        prefix = article + tuple(f"paragraph_{j}" for j in range(length))
        long_doc.append(make_request("long_document", i, "doc-user", "doc-1", "L" + str((i % 3) + 1), prefix, i * 60_000))
    return {
        "workbuddy_return": workbuddy,
        "server_shared_template": server,
        "long_document": long_doc,
    }


def load_real_jsonl(path: Path, workload_name: str, limit: int) -> list[Request]:
    rows: list[Request] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            item = json.loads(line)
            text = str(item.get("common_prefix_text", ""))
            if not text:
                continue
            rows.append(
                make_request(
                    workload_name,
                    int(item.get("request_id", len(rows))),
                    str(item.get("user_id", item.get("app_name", "user"))),
                    str(item.get("context_id", "context")),
                    str(item.get("lora_id", "unknown")),
                    proxy_tokens(text),
                    int(item.get("arrival_ms", len(rows))),
                )
            )
    return rows


def key_for(request: Request) -> tuple[str, tuple[str, ...]]:
    return request.lora_id, request.prefix


def simulate_flat(requests: list[Request], capacity: int) -> tuple[list[TraceRow], Summary]:
    cache: dict[tuple[str, tuple[str, ...]], int] = {}
    rows: list[TraceRow] = []
    evictions = 0
    for index, request in enumerate(requests):
        key = key_for(request)
        hit = key in cache
        if hit:
            cache[key] = index
        else:
            cache[key] = index
            if len(cache) > capacity:
                victim = min(cache, key=lambda candidate: cache[candidate])
                del cache[victim]
                evictions += 1
        reused = len(request.prefix) if hit else 0
        rows.append(TraceRow(request.workload, "flat_leaf_lru", request.request_id, request.sequence_id,
                             request.lora_id, len(request.prefix), reused, int(hit), int(hit),
                             1 if not hit else 0, evictions, len(cache), len(cache), 1))
    return rows, summarize(rows)


def chunk_prefix(prefix: tuple[str, ...], chunk_size: int) -> list[tuple[str, ...]]:
    return [prefix[start : start + chunk_size] for start in range(0, len(prefix), chunk_size)]


class IncrementalTrie:
    def __init__(self, strategy: str, capacity: int, chunk_size: int, background_chunks: int = 0) -> None:
        self.strategy = strategy
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.nodes: dict[int, TrieNode] = {0: TrieNode(0, None, "<root>", 0)}
        self.next_id = 1
        self.sequence_paths: dict[str, list[int]] = {}
        self.sequence_last_access: dict[str, int] = {}
        self.evictions = 0
        self.background_chunks = max(0, background_chunks)
        self.pending_builds: list[tuple[str, int, list[tuple[str, ...]], int]] = []

    def _prune_unowned(self) -> None:
        changed = True
        while changed:
            changed = False
            for node_id, node in list(self.nodes.items()):
                if node_id == 0 or node.children or node.owners:
                    continue
                parent = self.nodes.get(node.parent) if node.parent is not None else None
                if parent is not None:
                    parent.children.pop(node.token, None)
                del self.nodes[node_id]
                changed = True

    def _evict_leaf(self) -> None:
        candidates = [node for node in self.nodes.values() if node.node_id != 0 and not node.children]
        if not candidates:
            return
        victim = min(candidates, key=lambda node: (node.last_access, node.depth, node.node_id))
        parent = self.nodes.get(victim.parent) if victim.parent is not None else None
        if parent is not None:
            parent.children.pop(victim.token, None)
        for path in self.sequence_paths.values():
            if victim.node_id in path:
                path.remove(victim.node_id)
        del self.nodes[victim.node_id]
        self.evictions += 1
        self._prune_unowned()

    def _evict_sequence(self) -> None:
        if not self.sequence_last_access:
            return
        victim_sequence = min(self.sequence_last_access, key=self.sequence_last_access.get)
        for node_id in self.sequence_paths.pop(victim_sequence, []):
            node = self.nodes.get(node_id)
            if node is not None:
                node.owners.discard(victim_sequence)
        del self.sequence_last_access[victim_sequence]
        self.evictions += 1
        self._prune_unowned()

    def _make_room(self, request: Request, is_new_sequence: bool) -> None:
        if self.strategy == "trie_leaf_lru":
            while len(self.nodes) - 1 > self.capacity:
                self._evict_leaf()
        else:
            if is_new_sequence:
                while len(self.sequence_paths) > self.capacity:
                    self._evict_sequence()

    def _create_child(self, parent_id: int, chunk: tuple[str, ...], owner: str, index: int) -> int:
        edge = "\x1f".join(chunk)
        child = self.nodes[parent_id].children.get(edge)
        if child is None:
            node = TrieNode(self.next_id, parent_id, edge, self.nodes[parent_id].depth + len(chunk))
            self.next_id += 1
            self.nodes[node.node_id] = node
            self.nodes[parent_id].children[edge] = node.node_id
            child = node.node_id
        node = self.nodes[child]
        node.owners.add(owner)
        node.last_access = index
        return child

    def _run_background_build(self, index: int) -> int:
        """Build queued suffix chunks during an idle window, at most N per request."""
        if self.strategy != "trie_sequence_strict" or self.background_chunks <= 0:
            return 0
        built = 0
        next_queue: list[tuple[str, int, list[tuple[str, ...]], int]] = []
        for owner, parent_id, remaining, start in self.pending_builds:
            if built >= self.background_chunks or not remaining or parent_id not in self.nodes:
                if remaining and parent_id in self.nodes:
                    next_queue.append((owner, parent_id, remaining, start))
                continue
            child = self._create_child(parent_id, remaining[0], owner, index)
            built += 1
            rest = remaining[1:]
            if rest:
                next_queue.append((owner, child, rest, start + 1))
            self.sequence_paths.setdefault(owner, []).append(child)
        self.pending_builds = next_queue
        self._prune_unowned()
        return built

    def access(self, request: Request, index: int) -> TraceRow:
        is_new_sequence = request.sequence_id not in self.sequence_paths
        chunks = chunk_prefix(request.prefix, self.chunk_size)
        current = 0
        matched_chunks = 0
        walk_steps = 0
        path: list[int] = []
        for chunk in chunks:
            walk_steps += 1
            child = self.nodes[current].children.get("\x1f".join(chunk))
            if child is None:
                break
            current = child
            matched_chunks += 1
            path.append(current)
        reused = min(len(request.prefix), matched_chunks * self.chunk_size)
        complete = matched_chunks == len(chunks)
        created = 0
        current = self.nodes[path[-1]].node_id if path else 0
        missing_at = matched_chunks
        if self.strategy == "trie_sequence_strict" and missing_at < len(chunks):
            current = self._create_child(current, chunks[missing_at], request.sequence_id, index)
            path.append(current)
            created = 1
            remaining = chunks[missing_at + 1 :]
            if remaining:
                self.pending_builds.append((request.sequence_id, current, remaining, missing_at + 1))
        else:
            for chunk in chunks[missing_at:]:
                current = self._create_child(current, chunk, request.sequence_id, index)
                created += 1
                path.append(current)
        if self.strategy in {"trie_sequence_lru", "trie_sequence_strict"}:
            old_path = self.sequence_paths.get(request.sequence_id, [])
            for old_id in old_path:
                old_node = self.nodes.get(old_id)
                if old_node is not None and old_id not in path:
                    old_node.owners.discard(request.sequence_id)
            self.sequence_paths[request.sequence_id] = path
            self.sequence_last_access[request.sequence_id] = index
            self._prune_unowned()
        self._make_room(request, is_new_sequence)
        created += self._run_background_build(index)
        return TraceRow(request.workload, self.strategy, request.request_id, request.sequence_id,
                        request.lora_id, len(request.prefix), reused, int(reused > 0), int(complete),
                        created, self.evictions, len(self.nodes) - 1,
                        len(self.sequence_paths) if self.strategy in {"trie_sequence_lru", "trie_sequence_strict"} else len({owner for node in self.nodes.values() for owner in node.owners}),
                        walk_steps)


def simulate_trie(requests: list[Request], strategy: str, capacity: int, chunk_size: int, background_chunks: int = 0) -> tuple[list[TraceRow], Summary]:
    trie = IncrementalTrie(strategy, capacity, chunk_size, background_chunks)
    rows = [trie.access(request, index) for index, request in enumerate(requests)]
    return rows, summarize(rows)


def summarize(rows: list[TraceRow]) -> Summary:
    if not rows:
        raise ValueError("empty workload")
    requests = len(rows)
    total_prefix = sum(row.prefix_tokens for row in rows)
    total_reused = sum(row.reused_tokens for row in rows)
    return Summary(
        rows[0].workload,
        rows[0].strategy,
        requests,
        sum(row.hit for row in rows) / requests,
        sum(row.complete_hit for row in rows) / requests,
        total_reused / total_prefix if total_prefix else 0.0,
        total_reused / requests,
        max(row.cache_nodes for row in rows),
        max(row.active_sequences for row in rows),
        rows[-1].evictions,
        sum(row.walk_steps for row in rows) / requests,
        sum(row.created_nodes for row in rows),
    )


def run_all(workloads: dict[str, list[Request]], capacity: int, chunk_size: int) -> tuple[list[Summary], list[TraceRow]]:
    summaries: list[Summary] = []
    traces: list[TraceRow] = []
    for requests in workloads.values():
        for strategy in ("flat_leaf_lru", "trie_leaf_lru", "trie_sequence_lru", "trie_sequence_strict"):
            if strategy == "flat_leaf_lru":
                rows, summary = simulate_flat(requests, capacity)
            elif strategy == "trie_leaf_lru":
                rows, summary = simulate_trie(requests, strategy, capacity * chunk_size, chunk_size)
            elif strategy == "trie_sequence_lru":
                rows, summary = simulate_trie(requests, strategy, capacity, chunk_size)
            else:
                rows, summary = simulate_trie(requests, strategy, capacity, chunk_size, background_chunks=1)
            summaries.append(summary)
            traces.extend(rows)
    return summaries, traces


def write_csv(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in writer.fieldnames})


def sentence_fixed_chunks(text: str, min_tokens: int = 12, max_tokens: int = 32) -> list[tuple[str, ...]]:
    """Sentence-first splitter with a fixed-length fallback for long sentences."""
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("invalid min/max token bounds")
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?\.])\s*|[\r\n]+", text) if part.strip()]
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    for sentence in sentences:
        tokens = list(proxy_tokens(sentence))
        if not tokens:
            continue
        if len(tokens) > max_tokens:
            if current:
                chunks.append(tuple(current))
                current = []
            chunks.extend(tuple(tokens[start : start + max_tokens]) for start in range(0, len(tokens), max_tokens))
            continue
        if current and len(current) + len(tokens) > max_tokens and len(current) >= min_tokens:
            chunks.append(tuple(current))
            current = []
        current.extend(tokens)
        if len(current) >= max_tokens:
            chunks.append(tuple(current))
            current = []
    if current:
        chunks.append(tuple(current))
    return chunks


def run_chunking_experiment() -> list[ChunkingRow]:
    """Proxy timing experiment: serial TTFT is split+inference; parallel is max()."""
    texts = {
        "workbuddy_sentence": (
            "用户先说明出发地和目的地。助手记录周五去程并询问返程日期。"
            "几分钟后用户要求把返程改到周二并选择靠窗座位。"
        ),
        "long_document_sentence": "".join(
            f"第{i}段讨论能源转型、政策证据和实施风险。" for i in range(18)
        ),
        "long_sentence_fallback": "这是一个" + "非常长的连续技术描述 " * 45 + "句末。",
    }
    rows: list[ChunkingRow] = []
    for name, text in texts.items():
        token_count = len(proxy_tokens(text))
        chunks = sentence_fixed_chunks(text)
        # Deterministic local proxy, not a claim about GPU/model latency.
        split_ms = 0.025 * token_count + 0.45 * len(chunks)
        inference_ms = 0.18 * token_count + 6.0
        serial = split_ms + inference_ms
        parallel = max(split_ms, inference_ms)
        rows.append(ChunkingRow(name, token_count, len(chunks), split_ms, inference_ms,
                                serial, parallel, serial - parallel,
                                int(all(len(chunk) <= 32 for chunk in chunks))))
    return rows


def write_summary_json(path: Path, summaries: list[Summary], config: dict[str, object]) -> None:
    payload = {"config": config, "results": [{name: getattr(row, name) for name in row.__dataclass_fields__} for row in summaries]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_workloads(args: argparse.Namespace) -> dict[str, list[Request]]:
    workloads = synthetic_workloads()
    if args.real_root:
        root = Path(args.real_root)
        candidates = {
            "msc_real": root / "msc_continuous" / "grouped" / "grouped_requests.jsonl",
            "opus100_real": root / "opus100_parallel" / "grouped" / "grouped_requests.jsonl",
        }
        for name, path in candidates.items():
            if path.is_file():
                rows = load_real_jsonl(path, name, args.real_limit)
                if rows:
                    workloads[name] = rows
    return workloads


def generate_plots(output_dir: Path, summaries: list[Summary], traces: list[TraceRow]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["flat leaf", "trie leaf", "trie sequence", "strict + background"]
    strategies = ["flat_leaf_lru", "trie_leaf_lru", "trie_sequence_lru", "trie_sequence_strict"]
    workloads = list(dict.fromkeys(row.workload for row in summaries))
    colors = ["#6c757d", "#2a9d8f", "#e76f51", "#457b9d"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("token_coverage", "Reused prefix tokens", "coverage"),
        ("request_hit_rate", "Requests with any reuse", "rate"),
        ("peak_nodes", "Peak stored nodes", "count"),
        ("evictions", "Evictions", "count"),
    ]
    x = np.arange(len(workloads))
    width = 0.18
    offsets = (-(1.5 * width), -(0.5 * width), 0.5 * width, 1.5 * width)
    for axis, (field, title, _) in zip(axes.flat, metrics):
        for offset, strategy, label, color in zip(offsets, strategies, labels, colors):
            values = [getattr(next(row for row in summaries if row.workload == workload and row.strategy == strategy), field) for workload in workloads]
            axis.bar(x + offset, values, width, label=label, color=color)
        axis.set_title(title)
        axis.set_xticks(x, [name.replace("_", "\n") for name in workloads], fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Incremental prefix tree: storage and matching comparison", fontsize=14)
    fig.savefig(output_dir / "tree_policy_comparison.png", dpi=180)
    plt.close(fig)

    wb = [row for row in traces if row.workload == "workbuddy_return"]
    fig, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for strategy, label, color in zip(strategies, labels, colors):
        selected = [row for row in wb if row.strategy == strategy]
        axis.plot([row.request_id for row in selected], [row.reused_tokens for row in selected], marker="o", linewidth=2, label=label, color=color)
    axis.set_title("WorkBuddy-style return trace: matched prefix length")
    axis.set_xlabel("Request index")
    axis.set_ylabel("Reused prefix tokens")
    axis.set_xticks(range(max(row.request_id for row in wb) + 1))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output_dir / "workbuddy_match_trace.png", dpi=180)
    plt.close(fig)

    # A compact structural drawing: whole-prefix leaves versus shared branches.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, title, mode in zip(axes, ("Flat leaf storage", "Root-to-leaf incremental trie"), ("flat", "trie")):
        axis.axis("off")
        axis.set_title(title)
        positions = {"root": (0.5, 0.9)}
        if mode == "flat":
            nodes = [("r1", "S+C+T1", 0.2, 0.55), ("r2", "S+C+T2", 0.5, 0.35), ("r3", "S+C+T3", 0.8, 0.55)]
            for name, label, x_pos, y_pos in nodes:
                axis.plot([0.5, x_pos], [0.9, y_pos + 0.08], color="#adb5bd", linewidth=1)
                axis.scatter([x_pos], [y_pos], s=1300, color="#6c757d", alpha=0.9)
                axis.text(x_pos, y_pos, label, ha="center", va="center", color="white", fontsize=9)
        else:
            points = {"s": (0.5, 0.72), "c": (0.5, 0.5), "t1": (0.25, 0.25), "t2": (0.5, 0.25), "t3": (0.75, 0.25)}
            edges = [("root", "s"), ("s", "c"), ("c", "t1"), ("c", "t2"), ("c", "t3")]
            labels_map = {"root": "root", "s": "S", "c": "C", "t1": "T1", "t2": "T2", "t3": "T3"}
            for src, dst in edges:
                src_point = positions[src] if src in positions else points[src]
                x1, y1 = src_point; x2, y2 = points[dst]
                axis.plot([x1, x2], [y1 - 0.04, y2 + 0.05], color="#adb5bd", linewidth=1.5)
            for name, (x_pos, y_pos) in {**positions, **points}.items():
                axis.scatter([x_pos], [y_pos], s=900, color="#2a9d8f" if name in {"s", "c"} else "#e76f51", alpha=0.95)
                axis.text(x_pos, y_pos, labels_map.get(name, name), ha="center", va="center", color="white", fontsize=10)
        axis.set_xlim(0, 1); axis.set_ylim(0.05, 1.0)
    fig.suptitle("Storage unit changes the reusable structure", fontsize=14)
    fig.savefig(output_dir / "tree_structure_comparison.png", dpi=180)
    plt.close(fig)

    chunk_path = output_dir / "chunking_parallel_results.csv"
    if chunk_path.is_file():
        import csv
        with chunk_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
            names = [row["workload"].replace("_", "\n") for row in rows]
            x = np.arange(len(rows))
            width = 0.28
            axis.bar(x - width / 2, [float(row["serial_ttft_ms"]) for row in rows], width, label="serial TTFT", color="#6c757d")
            axis.bar(x + width / 2, [float(row["parallel_ttft_ms"]) for row in rows], width, label="parallel TTFT", color="#457b9d")
            axis.set_title("Sentence + fixed fallback: serial versus overlapped TTFT")
            axis.set_ylabel("Proxy milliseconds")
            axis.set_xticks(x, names)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.legend(frameon=False)
            fig.savefig(output_dir / "chunking_parallel_comparison.png", dpi=180)
            plt.close(fig)


def write_report(path: Path, summaries: list[Summary], config: dict[str, object]) -> None:
    lines = [
        "# test9 增量前缀树与端侧序列置换小实验结果",
        "",
        "本报告由 `run_tree_experiment.py` 自动生成。实验不加载大模型，只验证建树、前缀匹配和缓存置换三个与模型 logits 无关的系统机制。",
        "",
        "## 实验设置",
        "",
        f"- chunk 大小：`{config['chunk_size']}` 个代理 token。",
        f"- flat/trie 叶子容量：`{config['capacity']}`。序列策略的容量表示同时保留的 `conversation × LoRA` 序列数。",
        "- 匹配均从 root 开始，逐级比较 chunk；遇到不存在的边立即停止匹配并从该位置创建新分支。",
        "- 代理 token 是稳定的词/标点切分，不等同于 Qwen tokenizer；因此结果用于比较结构，不用于宣称真实 TTFT。",
        "",
        "## 四种策略",
        "- Capacity note: flat records use 4 complete-prefix entries; trie leaf uses a 16-node budget (capacity × chunk size); sequence policies use 4 active conversation × LoRA paths. These are separate budget units and should be equalized in a future memory-byte ablation.",
        "",
        "1. `flat_leaf_lru`：每个 LoRA 保存一条完整 prefix，只有完整 prefix 再次出现时才命中；淘汰单位是整条叶记录。",
        "2. `trie_leaf_lru`：按 chunk 从 root 增量建树，匹配到公共祖先即可复用；容量压力下逐个删除最低访问叶节点。",
        "3. `trie_sequence_lru`：建树和匹配与 2 相同，但把 `conversation_id × lora_id` 的整条路径作为端侧原子置换单位；淘汰时移除该序列拥有的所有节点，仍被其它序列使用的公共祖先保留。",
        "",
        "老师提出的 BFS/DFS 方向在这里落实为 root-to-leaf 增量遍历。对同一请求序列，BFS/DFS 只是节点访问顺序；真正改变系统性质的是“逐级匹配 + 分支创建”和“序列级置换”这两个组合。",
        "",
        "## 结果",
        "",
        "| workload | strategy | any-hit | complete-hit | token coverage | peak nodes | evictions | mean walk |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row.workload} | {row.strategy} | {row.request_hit_rate:.1%} | {row.complete_hit_rate:.1%} | {row.token_coverage:.1%} | {row.peak_nodes} | {row.evictions} | {row.mean_walk_steps:.2f} |")
    lines += [
        "",
        "## 如何解释",
        "",
        "- `flat_leaf_lru` 的 complete-hit 低，说明整段 prefix 只要有一个日期、座位或新对话轮次变化，就无法利用较浅的公共上下文。",
        "- 两种 trie 都可以在首个新 chunk 处停止并复用祖先，因此对“几分钟后回到同一会话、只改返程日期”的 WorkBuddy 场景更合适。",
        "- `trie_sequence_lru` 的创新不在于产生更多叶节点，而在于把端侧淘汰从“随机删一个叶”改成“删掉一个 conversation × LoRA 序列”，避免留下难以恢复的半条会话路径。",
        "- 服务器场景仍可使用细粒度 trie，因为多个用户共享 system/template 祖先；端侧则可以采用较粗 chunk 和序列级原子淘汰。",
        "- 这个小实验没有测量 BGE-M3/KeyBERT 推理延迟，也没有把关键词模型并行接入 llama.cpp；它只隔离出建树和存储机制本身的收益，后续可在同一 trace 上叠加真实 tokenizer、GPU TTFT 和语义模型成本。",
        "",
        "## 图片",
        "",
        "- `tree_policy_comparison.png`：各 workload 的命中、token 覆盖、节点峰值和淘汰次数。",
        "- `workbuddy_match_trace.png`：WorkBuddy 返回场景中每次请求实际复用的 prefix token。",
        "- `tree_structure_comparison.png`：整段叶存储与 root-to-leaf 增量树的结构差异。",
        "",
        "## 与 test8 的关系和下一步",
        "",
        "`test8` 已经具备 chunked prefix tree、family 评分和三级 KV 驻留，但当前容量管理仍以节点/family 评分为主，尚未把端侧 `conversation × LoRA` 作为独立原子序列。test9 的 Python 小实验验证了这个机制值得接入，下一步可以在 C++ 中增加 `sequence_family_id`、序列级 LRU/收益评分和路径引用计数，再用真实 workload 做 GPU KV 回放。",
        "",
        "本实验是结构性验证，不代表任何一种策略已经在真实模型上获得固定 TTFT 提升。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_extended_report(path: Path, chunking_rows: list[ChunkingRow]) -> None:
    """Append the strict-build and overlap experiment to the generated report."""
    lines = [
        "",
        "## Strict incremental build and overlap extension",
        "",
        "`trie_sequence_strict` is the strict variant added for the follow-up experiment. The foreground walk starts at root, stops at the first missing chunk, creates only that edge, and queues the remaining suffix. One queued chunk is built in each later idle window. This separates request-critical matching from background materialization.",
        "",
        "The strategy table above now contains four strategies: flat complete-prefix LRU, trie leaf LRU, trie sequence LRU, and strict sequence LRU with background build. The strict strategy is intentionally conservative: lower immediate coverage can be exchanged for less foreground tree work.",
        "",
        "## Sentence + fixed fallback parallel timing",
        "",
        "The splitter first uses sentence/newline boundaries and falls back to fixed chunks when a sentence exceeds 32 proxy tokens. The timing model is deterministic: serial TTFT = split + inference; parallel TTFT = max(split, inference). It demonstrates the overlap upper bound, not hardware performance.",
        "",
        "| workload | tokens | chunks | split ms | inference ms | serial TTFT | parallel TTFT | overlap saved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row.workload} | {row.text_tokens} | {row.chunks} | {row.split_ms:.2f} | {row.inference_ms:.2f} | {row.serial_ttft_ms:.2f} | {row.parallel_ttft_ms:.2f} | {row.overlap_saved_ms:.2f} |"
        for row in chunking_rows
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--real-root", type=Path)
    parser.add_argument("--real-limit", type=int, default=40)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.capacity <= 0 or args.chunk_size <= 0 or args.real_limit <= 0:
        raise SystemExit("capacity, chunk-size and real-limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workloads = load_workloads(args)
    summaries, traces = run_all(workloads, args.capacity, args.chunk_size)
    chunking_rows = run_chunking_experiment()
    config = {"capacity": args.capacity, "chunk_size": args.chunk_size, "workloads": list(workloads), "real_limit": args.real_limit}
    write_csv(args.output_dir / "tree_experiment_results.csv", summaries)
    write_csv(args.output_dir / "tree_request_trace.csv", traces)
    write_csv(args.output_dir / "chunking_parallel_results.csv", chunking_rows)
    write_summary_json(args.output_dir / "tree_experiment_results.json", summaries, config)
    write_report(args.output_dir / "TREE_EXPERIMENT_RESULTS.md", summaries, config)
    append_extended_report(args.output_dir / "TREE_EXPERIMENT_RESULTS.md", chunking_rows)
    if not args.no_plots:
        generate_plots(args.output_dir, summaries, traces)
    print(json.dumps({"output_dir": str(args.output_dir), "workloads": list(workloads), "rows": len(summaries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
