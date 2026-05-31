from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RelationTables:
    global_seq: pd.DataFrame
    global_co: pd.DataFrame
    user_seq: dict
    user_co: dict


def build_global_relation_tables(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build global sequential and co-occurrence relation tables from train splits."""
    train_parts = []
    for _, group in df.sort_values(["user_id", "timestamp"]).groupby("user_id"):
        split_idx = int((1.0 - test_ratio) * len(group))
        train_parts.append(group.iloc[:split_idx])
    train_df = pd.concat(train_parts).reset_index(drop=True)

    seq_df = train_df.copy()
    seq_df["next_app"] = seq_df.groupby(["user_id", "session_id"])["app_name"].shift(-1)
    seq_data = seq_df.dropna(subset=["next_app"])
    seq_counts = seq_data.groupby(["app_name", "next_app"]).size().reset_index(name="count")
    total_src = seq_data.groupby("app_name").size().reset_index(name="total")
    global_seq = seq_counts.merge(total_src, on="app_name")
    global_seq["weight"] = global_seq["count"] / global_seq["total"]
    global_seq = global_seq[["app_name", "next_app", "weight"]]
    global_seq.columns = ["src_lora", "dst_lora", "weight"]
    global_seq = global_seq.sort_values("weight", ascending=False).reset_index(drop=True)

    app_session_counts = train_df.groupby("app_name")["session_id"].nunique().to_dict()
    pairs = []
    for _, sess_group in train_df.groupby(["user_id", "session_id"]):
        apps = sorted(sess_group["app_name"].unique())
        for i in range(len(apps)):
            for j in range(i + 1, len(apps)):
                pairs.append((apps[i], apps[j]))

    if pairs:
        co_df = pd.DataFrame(pairs, columns=["lora_a", "lora_b"])
        global_co = co_df.groupby(["lora_a", "lora_b"]).size().reset_index(name="intersect_count")
        global_co["weight"] = global_co.apply(
            lambda r: _jaccard_weight(r, app_session_counts), axis=1
        )
        global_co = global_co.sort_values("weight", ascending=False).reset_index(drop=True)
    else:
        global_co = pd.DataFrame(columns=["lora_a", "lora_b", "intersect_count", "weight"])

    return global_seq, global_co


def build_user_relation_tables(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[dict, dict]:
    """Build per-user sequential and co-occurrence relation tables from train splits."""
    user_seq = {}
    user_co = {}
    for uid, group in df.sort_values(["user_id", "timestamp"]).groupby("user_id"):
        if len(group) < 5:
            continue
        split_idx = int((1.0 - test_ratio) * len(group))
        train_df = group.iloc[:split_idx].copy()

        seq_df = train_df.copy()
        seq_df["next_app"] = seq_df.groupby("session_id")["app_name"].shift(-1)
        seq_data = seq_df.dropna(subset=["next_app"])
        if not seq_data.empty:
            seq_counts = seq_data.groupby(["app_name", "next_app"]).size().reset_index(name="count")
            total_src = seq_data.groupby("app_name").size().reset_index(name="total")
            table = seq_counts.merge(total_src, on="app_name")
            table["weight"] = table["count"] / table["total"]
            table = table[["app_name", "next_app", "weight"]]
            table.columns = ["src_lora", "dst_lora", "weight"]
            user_seq[uid] = table.sort_values("weight", ascending=False).reset_index(drop=True)

        app_session_counts = train_df.groupby("app_name")["session_id"].nunique().to_dict()
        pairs = []
        for _, sess_group in train_df.groupby("session_id"):
            apps = sorted(sess_group["app_name"].unique())
            for i in range(len(apps)):
                for j in range(i + 1, len(apps)):
                    pairs.append((apps[i], apps[j]))
        if pairs:
            co_df = pd.DataFrame(pairs, columns=["lora_a", "lora_b"])
            table = co_df.groupby(["lora_a", "lora_b"]).size().reset_index(name="intersect_count")
            table["weight"] = table.apply(lambda r: _jaccard_weight(r, app_session_counts), axis=1)
            user_co[uid] = table.sort_values("weight", ascending=False).reset_index(drop=True)

    return user_seq, user_co


def relation_scores(
    current_name: str | None,
    table: pd.DataFrame,
    relation_type: str,
    name_to_id: dict,
    num_classes: int,
    temperature: float = 0.05,
) -> np.ndarray:
    """Return masked-softmax normalized relation probability vector."""
    scores = np.zeros(num_classes, dtype=np.float64)
    if current_name is None or table is None or table.empty:
        return scores

    if relation_type == "seq":
        rows = table[table["src_lora"] == current_name]
        for _, row in rows.iterrows():
            dst = row["dst_lora"]
            if dst in name_to_id:
                scores[name_to_id[dst]] = float(row["weight"])
    elif relation_type == "co":
        rows_a = table[table["lora_a"] == current_name].rename(columns={"lora_b": "neighbor"})
        rows_b = table[table["lora_b"] == current_name].rename(columns={"lora_a": "neighbor"})
        rows = pd.concat([rows_a, rows_b], ignore_index=True)
        for _, row in rows.iterrows():
            neighbor = row["neighbor"]
            if neighbor in name_to_id:
                scores[name_to_id[neighbor]] = float(row["weight"])
    else:
        raise ValueError(f"Unknown relation_type: {relation_type}")

    mask = scores > 0
    if not mask.any():
        return scores
    exp_scores = np.zeros_like(scores)
    exp_scores[mask] = np.exp(scores[mask] / max(temperature, 1e-8))
    return exp_scores / (exp_scores.sum() + 1e-12)


def _jaccard_weight(row: pd.Series, app_session_counts: dict) -> float:
    a = row["lora_a"]
    b = row["lora_b"]
    inter = float(row["intersect_count"])
    union = app_session_counts.get(a, 0) + app_session_counts.get(b, 0) - inter
    return inter / union if union > 0 else 0.0
