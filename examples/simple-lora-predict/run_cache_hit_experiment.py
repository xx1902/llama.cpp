from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from predictor.gru_model import TinyGRUPredictor
from predictor.relation_table import (
    build_global_relation_tables,
    build_user_relation_tables,
    relation_scores,
)

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Source Han Sans SC",
]

POLICY_ORDER = ["LRU", "LFU", "Predictive"]

POLICY_NAME = {
    "LRU": "LRU",
    "LFU": "LFU",
    "Predictive": "Predictive",
}

POLICY_COLOR = {
    "LRU": "#4C72B0",
    "LFU": "#DD8452",
    "Predictive": "#55A868",
}


def setup_chinese_font() -> None:
    selected = None

    for name in CN_FONT_CANDIDATES:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            selected = name
            break
        except Exception:
            continue

    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        print("warning: 未找到常见中文字体，中文可能无法正常显示。")

    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lsapp(path: str | Path) -> tuple[pd.DataFrame, LabelEncoder]:
    df = pd.read_csv(path, sep="\t" if str(path).endswith((".tsv", ".tsv.gz")) else ",")

    if len(df.columns) >= 5 and not {"user_id", "timestamp", "app_name"}.issubset(df.columns):
        df = df.iloc[:, :5]
        df.columns = ["user_id", "session_id_raw", "timestamp", "app_name", "event_type"]

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "event_type" in df.columns:
        df = df[df["event_type"].eq("Opened")].copy()

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    user_changed = df["user_id"].ne(df["user_id"].shift(1))
    app_changed = df["app_name"].ne(df["app_name"].shift(1))
    gap_1m = df["timestamp"].sub(df["timestamp"].shift(1)) > pd.Timedelta(minutes=1)
    gap_5m = df["timestamp"].sub(df["timestamp"].shift(1)) > pd.Timedelta(minutes=5)

    df["interaction_id"] = (user_changed | app_changed | gap_1m).cumsum()
    df["session_id"] = (user_changed | gap_5m).cumsum()

    df_start = df.drop_duplicates("interaction_id", keep="first").copy()
    df_start["open_time"] = df_start["timestamp"]
    df_start["hour"] = df_start["open_time"].dt.hour
    df_start["sin_hour"] = np.sin(2 * np.pi * df_start["hour"] / 24)
    df_start["cos_hour"] = np.cos(2 * np.pi * df_start["hour"] / 24)
    df_start["delta_sec"] = (
        df_start.groupby("user_id")["open_time"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )
    df_start["delta_norm"] = np.log1p(df_start["delta_sec"].clip(lower=0))

    encoder = LabelEncoder()
    df_start["lora_id"] = encoder.fit_transform(df_start["app_name"])

    return df_start.reset_index(drop=True), encoder


def to_tensors(ids, times, labels):
    return (
        torch.tensor(np.asarray(ids), dtype=torch.long),
        torch.tensor(np.asarray(times), dtype=torch.float32),
        torch.tensor(np.asarray(labels), dtype=torch.long),
    )


def build_user_windows(
    df: pd.DataFrame,
    window_size: int,
    test_ratio: float,
) -> dict:
    samples = {}

    groups = list(df.sort_values(["user_id", "timestamp"]).groupby("user_id"))

    for uid, group in tqdm(groups, desc=f"构造窗口 L={window_size}", unit="user"):
        if len(group) <= window_size + 1:
            continue

        ids = group["lora_id"].to_numpy()
        times = group[["sin_hour", "cos_hour", "delta_norm"]].to_numpy(dtype=np.float32)

        split_idx = int((1.0 - test_ratio) * len(group))

        train_ids, train_t, train_y = [], [], []
        test_ids, test_t, test_y = [], [], []

        for i in range(len(group) - window_size):
            x_id = ids[i : i + window_size]
            x_t = times[i : i + window_size]
            y = ids[i + window_size]

            if i + window_size < split_idx:
                train_ids.append(x_id)
                train_t.append(x_t)
                train_y.append(y)
            else:
                test_ids.append(x_id)
                test_t.append(x_t)
                test_y.append(y)

        if train_y and test_y:
            samples[uid] = {
                "train": to_tensors(train_ids, train_t, train_y),
                "test": to_tensors(test_ids, test_t, test_y),
            }

    return samples


def merge_train_samples(samples_by_user: dict):
    xs, ts, ys = [], [], []

    for data in samples_by_user.values():
        x, t, y = data["train"]
        xs.append(x)
        ts.append(t)
        ys.append(y)

    return torch.cat(xs), torch.cat(ts), torch.cat(ys)


def train_model(
    samples_by_user: dict,
    num_loras: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> TinyGRUPredictor:
    model = TinyGRUPredictor(
        num_loras=num_loras,
        hidden_dim=64,
        num_layers=1,
        time_dim=3,
    ).to(device)

    x_train, t_train, y_train = merge_train_samples(samples_by_user)
    loader = DataLoader(
        TensorDataset(x_train, t_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for x, t, y in tqdm(loader, desc=f"训练 GRU {epoch}/{epochs}", unit="batch"):
            x = x.to(device)
            t = t.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x, t), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        print(f"epoch={epoch}, loss={np.mean(losses):.4f}")

    return model


def combine_relations(seq_scores: np.ndarray, co_scores: np.ndarray) -> np.ndarray:
    scores = 0.7 * seq_scores + 0.3 * co_scores
    return scores / (scores.sum() + 1e-12) if scores.sum() > 0 else scores


def collect_prediction_trace(
    model: torch.nn.Module,
    samples_by_user: dict,
    id_to_name: dict,
    name_to_id: dict,
    global_seq,
    global_co,
    user_seq_tables: dict,
    user_co_tables: dict,
    alpha: float,
    temperature: float,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    num_classes = len(id_to_name)

    with torch.no_grad():
        for uid, data in tqdm(samples_by_user.items(), desc="生成预测序列", unit="user"):
            x_test, t_test, y_test = data["test"]

            logits = model(x_test.to(device), t_test.to(device))
            gru_probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i, target_tensor in enumerate(y_test):
                target = int(target_tensor)
                last_id = int(x_test[i][-1])
                current_name = id_to_name.get(last_id)

                g_seq = relation_scores(
                    current_name,
                    global_seq,
                    "seq",
                    name_to_id,
                    num_classes,
                    temperature,
                )
                g_co = relation_scores(
                    current_name,
                    global_co,
                    "co",
                    name_to_id,
                    num_classes,
                    temperature,
                )
                g_comb = combine_relations(g_seq, g_co)

                u_seq_table = user_seq_tables.get(uid)
                u_co_table = user_co_tables.get(uid)

                u_seq = relation_scores(
                    current_name,
                    u_seq_table,
                    "seq",
                    name_to_id,
                    num_classes,
                    temperature,
                )
                u_co = relation_scores(
                    current_name,
                    u_co_table,
                    "co",
                    name_to_id,
                    num_classes,
                    temperature,
                )
                u_comb = combine_relations(u_seq, u_co)

                # Predictive 策略使用 GRU + 用户关系表融合概率。
                # 这里没有偷看未来请求，target 只用于评估命中率。
                probs = alpha * gru_probs[i] + (1.0 - alpha) * u_comb

                ranking = np.argsort(probs)[::-1]

                rows.append({
                    "user_id": uid,
                    "target_lora": target,
                    "ranking": " ".join(str(int(x)) for x in ranking),
                })

    return pd.DataFrame(rows)


def evaluate_cache_hit_rate(
    trace: pd.DataFrame,
    num_loras: int,
    capacity_ratio: float,
    policy: str,
) -> float:
    capacity = max(1, int(round(num_loras * capacity_ratio)))

    cache = []
    freq = Counter()
    hits = 0
    total = 0

    for _, row in trace.iterrows():
        target = int(row["target_lora"])

        if target in cache:
            hits += 1

        total += 1
        freq[target] += 1

        if policy == "LRU":
            if target in cache:
                cache.remove(target)
            cache.insert(0, target)

        elif policy == "LFU":
            if target not in cache:
                cache.append(target)

            cache = sorted(
                cache,
                key=lambda x: (-freq[x], x),
            )

        elif policy == "Predictive":
            ranking = [int(x) for x in str(row["ranking"]).split()]
            cache = ranking[:capacity]

        else:
            raise ValueError(f"unknown policy: {policy}")

        if len(cache) > capacity:
            cache = cache[:capacity]

    return hits / total if total > 0 else 0.0


def run_cache_hit_experiment(
    trace: pd.DataFrame,
    num_loras: int,
    capacity_ratios: list[float],
) -> pd.DataFrame:
    rows = []

    for ratio in capacity_ratios:
        for policy in POLICY_ORDER:
            hit_rate = evaluate_cache_hit_rate(
                trace=trace,
                num_loras=num_loras,
                capacity_ratio=ratio,
                policy=policy,
            )

            rows.append({
                "capacity_ratio": ratio,
                "capacity_label": f"{int(ratio * 100)}%",
                "policy": policy,
                "hit_rate": hit_rate,
            })

    return pd.DataFrame(rows)


def plot_cache_hit_rate(result_df: pd.DataFrame, out_path: Path) -> None:
    setup_chinese_font()

    labels = ["10%", "20%", "30%", "50%"]
    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(8, 5))

    for idx, policy in enumerate(POLICY_ORDER):
        values = []

        for label in labels:
            sub = result_df[
                (result_df["capacity_label"] == label)
                & (result_df["policy"] == policy)
            ]

            if sub.empty:
                values.append(0.0)
            else:
                values.append(float(sub.iloc[0]["hit_rate"]))

        offset = (idx - 1) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=POLICY_NAME[policy],
            color=POLICY_COLOR[policy],
            alpha=0.92,
        )

    plt.xlabel("GPU热缓存容量")
    plt.ylabel("适配器缓存命中率")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.02)
    plt.grid(True, axis="y", linestyle="-", alpha=0.35)
    plt.legend(title="策略", loc="upper left")
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-predict/datasets/lsapp.tsv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-predict/output",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"加载数据集: {args.input}")
    df, encoder = load_lsapp(args.input)

    id_to_name = {i: name for i, name in enumerate(encoder.classes_)}
    name_to_id = {name: i for i, name in id_to_name.items()}
    num_loras = len(id_to_name)

    print(f"记录数={len(df)}, 用户数={df['user_id'].nunique()}, LoRA数量={num_loras}, device={device}")

    print("构建关系表...")
    global_seq, global_co = build_global_relation_tables(df, test_ratio=args.test_ratio)
    user_seq, user_co = build_user_relation_tables(df, test_ratio=args.test_ratio)

    print(f"构建预测窗口 L={args.window}...")
    samples = build_user_windows(df, args.window, args.test_ratio)

    print("训练 GRU 预测模型...")
    model = train_model(
        samples_by_user=samples,
        num_loras=num_loras,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    print("生成预测 trace...")
    trace = collect_prediction_trace(
        model=model,
        samples_by_user=samples,
        id_to_name=id_to_name,
        name_to_id=name_to_id,
        global_seq=global_seq,
        global_co=global_co,
        user_seq_tables=user_seq,
        user_co_tables=user_co,
        alpha=args.alpha,
        temperature=args.temperature,
        device=device,
    )

    trace_path = out_dir / "cache_prediction_trace.csv"
    trace.to_csv(trace_path, index=False)
    print(f"预测 trace 已保存: {trace_path}")

    print("计算缓存命中率...")
    result_df = run_cache_hit_experiment(
        trace=trace,
        num_loras=num_loras,
        capacity_ratios=[0.10, 0.20, 0.30, 0.50],
    )

    result_path = out_dir / "cache_hit_rate_by_capacity.csv"
    result_df.to_csv(result_path, index=False)
    print(f"缓存命中率结果已保存: {result_path}")

    plot_cache_hit_rate(
        result_df,
        out_dir / "cache_hit_rate_by_capacity.png",
    )

    print("\n缓存命中率结果:")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()