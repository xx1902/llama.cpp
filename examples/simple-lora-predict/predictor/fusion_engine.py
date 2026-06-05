from __future__ import annotations

from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from predictor.relation_table import relation_scores


def topk_hit(probs: np.ndarray, target: int, k: int) -> bool:
    return int(target) in np.argsort(probs)[-k:]


def reciprocal_rank(probs: np.ndarray, target: int) -> float:
    ranking = np.argsort(probs)[::-1]
    matches = np.where(ranking == int(target))[0]
    return 1.0 / (matches[0] + 1) if len(matches) else 0.0


def evaluate_lru(samples_by_user: dict, capacity: int = 3) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    total = 0
    rr_sum = 0.0
    for _, data in tqdm(samples_by_user.items(), desc="LRU baseline", unit="user", leave=False):
        x_test, _, y_test = data["test"]
        cache = deque(maxlen=max(5, capacity))
        for seq, target in zip(x_test, y_test):
            recent = []
            for item in reversed(seq.tolist()):
                if item not in recent:
                    recent.append(item)
                if len(recent) >= 5:
                    break
            for k in hits:
                hits[k] += int(int(target) in recent[:k])
            if int(target) in recent:
                rr_sum += 1.0 / (recent.index(int(target)) + 1)
            cache.append(int(target))
            total += 1
    return {
        "top1": hits[1] / total,
        "top3": hits[3] / total,
        "top5": hits[5] / total,
        "mrr": rr_sum / total,
    }

def evaluate_lfu(samples_by_user: dict, capacity: int = 5) -> dict:
    """Evaluate an online LFU cache baseline.

    LFU 含义：
    - 对每个用户维护一个访问频率表。
    - 预测时优先选择历史访问次数最高的 adapter。
    - 每次真实 target 出现后，再更新频率。
    - Top-k 命中表示真实 target 是否在当前频率最高的 k 个 adapter 中。

    注意：
    - 这里的 LFU 是缓存策略基线，不调用 GRU。
    - 它和 LRU 使用同一份 samples_by_user 测试窗口。
    """
    from collections import Counter

    hits = {1: 0, 3: 0, 5: 0}
    total = 0
    rr_sum = 0.0

    for _, data in tqdm(samples_by_user.items(), desc="LFU baseline", unit="user", leave=False):
        x_train, _, y_train = data["train"]
        x_test, _, y_test = data["test"]

        freq = Counter()
        last_seen = {}
        time_step = 0

        # 用训练集初始化用户历史频率。
        for seq, target in zip(x_train, y_train):
            for item in seq.tolist():
                item = int(item)
                freq[item] += 1
                last_seen[item] = time_step
                time_step += 1

            target = int(target)
            freq[target] += 1
            last_seen[target] = time_step
            time_step += 1

        for seq, target in zip(x_test, y_test):
            target = int(target)

            # 当前窗口也作为预测前可见历史。
            for item in seq.tolist():
                item = int(item)

                if item not in freq:
                    freq[item] = 0

                last_seen[item] = time_step
                time_step += 1

            ranked = sorted(
                freq.keys(),
                key=lambda item: (freq[item], last_seen.get(item, -1)),
                reverse=True,
            )

            candidates = ranked[:max(5, capacity)]

            for k in hits:
                hits[k] += int(target in candidates[:k])

            if target in candidates:
                rr_sum += 1.0 / (candidates.index(target) + 1)

            # 预测完成后再更新真实访问。
            freq[target] += 1
            last_seen[target] = time_step
            time_step += 1

            total += 1

    return {
        "top1": hits[1] / total,
        "top3": hits[3] / total,
        "top5": hits[5] / total,
        "mrr": rr_sum / total,
    }


def evaluate_model_and_fusion(
    model: torch.nn.Module,
    samples_by_user: dict,
    id_to_name: dict,
    name_to_id: dict,
    global_seq_table,
    global_co_table,
    user_seq_tables: dict | None = None,
    user_co_tables: dict | None = None,
    alpha: float = 0.8,
    temperature: float = 0.05,
    device: str | torch.device = "cpu",
    desc: str = "Fusion eval",
) -> dict:
    """Evaluate GRU-only and relation-fusion variants on user test windows."""
    model.to(device).eval()
    num_classes = len(id_to_name)
    methods = [
        "GRU",
        "GRU+GlobalSeq",
        "GRU+GlobalComb",
        "GRU+UserSeq",
        "GRU+UserComb",
        "GlobalRelOnly",
        "UserRelOnly",
    ]
    stats = {m: {"top1": 0, "top3": 0, "top5": 0, "mrr": 0.0} for m in methods}
    total = 0

    with torch.no_grad():
        for uid, data in tqdm(samples_by_user.items(), desc=desc, unit="user", leave=False):
            x_test, t_test, y_test = data["test"]
            if len(y_test) == 0:
                continue
            logits = model(x_test.to(device), t_test.to(device))
            gru_probs = torch.softmax(logits, dim=1).cpu().numpy()
            for i, target_tensor in enumerate(y_test):
                target = int(target_tensor)
                last_id = int(x_test[i][-1])
                current_name = id_to_name.get(last_id)

                g_seq = relation_scores(current_name, global_seq_table, "seq", name_to_id, num_classes, temperature)
                g_co = relation_scores(current_name, global_co_table, "co", name_to_id, num_classes, temperature)
                g_comb = _combine_relations(g_seq, g_co)

                u_seq_table = (user_seq_tables or {}).get(uid)
                u_co_table = (user_co_tables or {}).get(uid)
                u_seq = relation_scores(current_name, u_seq_table, "seq", name_to_id, num_classes, temperature)
                u_co = relation_scores(current_name, u_co_table, "co", name_to_id, num_classes, temperature)
                u_comb = _combine_relations(u_seq, u_co)

                pred_map = {
                    "GRU": gru_probs[i],
                    "GRU+GlobalSeq": alpha * gru_probs[i] + (1 - alpha) * g_seq,
                    "GRU+GlobalComb": alpha * gru_probs[i] + (1 - alpha) * g_comb,
                    "GRU+UserSeq": alpha * gru_probs[i] + (1 - alpha) * u_seq,
                    "GRU+UserComb": alpha * gru_probs[i] + (1 - alpha) * u_comb,
                    "GlobalRelOnly": g_comb,
                    "UserRelOnly": u_comb,
                }
                for method, probs in pred_map.items():
                    _update_stats(stats[method], probs, target)
                total += 1

    return {
        method: {metric: value / total for metric, value in values.items()}
        for method, values in stats.items()
    }


def _combine_relations(seq_scores: np.ndarray, co_scores: np.ndarray) -> np.ndarray:
    scores = 0.7 * seq_scores + 0.3 * co_scores
    return scores / (scores.sum() + 1e-12) if scores.sum() > 0 else scores


def _update_stats(row: dict, probs: np.ndarray, target: int) -> None:
    row["top1"] += int(topk_hit(probs, target, 1))
    row["top3"] += int(topk_hit(probs, target, 3))
    row["top5"] += int(topk_hit(probs, target, 5))
    row["mrr"] += reciprocal_rank(probs, target)
