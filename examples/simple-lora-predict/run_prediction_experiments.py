from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictor.fusion_engine import evaluate_lru, evaluate_model_and_fusion
from predictor.gru_model import TinyGRUPredictor
from predictor.relation_table import build_global_relation_tables, build_user_relation_tables

from matplotlib import font_manager


CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]

BW_COLORS = ["#111111", "#444444", "#777777", "#999999", "#BBBBBB"]
BW_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BW_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
BW_HATCHES = ["", "//", "\\\\", "xx", "..", "++"]

def setup_cn_plot() -> None:
    """Windows-safe Chinese font setup. Falls back without raising errors."""
    selected = None
    for font_name in CN_FONT_CANDIDATES:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            selected = font_name
            break
        except Exception:
            continue

    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    else:
        print("[plot] 未找到常见中文字体，将使用 Matplotlib 默认字体，若中文乱码请安装 Microsoft YaHei 或 SimHei。")

    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font_scale=1.0)

def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lsapp(path: str | Path) -> pd.DataFrame:
    print(f"[1/9] Loading dataset: {path}", flush=True)
    df = pd.read_csv(path, sep="\t" if str(path).endswith((".tsv", ".tsv.gz")) else ",")
    if len(df.columns) >= 5 and not {"user_id", "timestamp", "app_name"}.issubset(df.columns):
        df = df.iloc[:, :5]
        df.columns = ["user_id", "session_id_raw", "timestamp", "app_name", "event_type"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "event_type" in df.columns:
        df = df[df["event_type"].eq("Opened")].copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Interaction/session reconstruction follows the original notebook, but keeps
    # the output columns compact for reproducible experiments.
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
    df_start["delta_sec"] = df_start.groupby("user_id")["open_time"].diff().dt.total_seconds().fillna(0)
    df_start["delta_norm"] = np.log1p(df_start["delta_sec"].clip(lower=0))

    enc = LabelEncoder()
    df_start["lora_id"] = enc.fit_transform(df_start["app_name"])
    return df_start.reset_index(drop=True), enc


def build_user_windows(df: pd.DataFrame, window_size: int, test_ratio: float = 0.2) -> dict:
    samples = {}
    groups = list(df.sort_values(["user_id", "timestamp"]).groupby("user_id"))
    for uid, group in tqdm(groups, desc=f"Build windows L={window_size}", unit="user", leave=False):
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
                "train": _to_tensors(train_ids, train_t, train_y),
                "test": _to_tensors(test_ids, test_t, test_y),
            }
    return samples


def merge_train_samples(samples_by_user: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    hidden_dim: int = 64,
    num_layers: int = 1,
) -> tuple[TinyGRUPredictor, pd.DataFrame]:
    model = TinyGRUPredictor(num_loras=num_loras, hidden_dim=hidden_dim, num_layers=num_layers, time_dim=3).to(device)
    x_train, t_train, y_train = merge_train_samples(samples_by_user)
    loader = DataLoader(TensorDataset(x_train, t_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    history = []

    for epoch in tqdm(range(1, epochs + 1), desc="Train GRU epochs", unit="epoch"):
        model.train()
        losses = []
        start = time.perf_counter()
        for x, t, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False):
            x, t, y = x.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, t), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val = evaluate_model_only(model, samples_by_user, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_top1": val["top1"],
            "val_top3": val["top3"],
            "val_top5": val["top5"],
            "epoch_seconds": time.perf_counter() - start,
        })
        print(f"epoch={epoch:02d} loss={history[-1]['train_loss']:.4f} top3={val['top3']:.4f}")
    return model, pd.DataFrame(history)


def evaluate_model_only(model: torch.nn.Module, samples_by_user: dict, device: torch.device) -> dict:
    model.eval()
    hits = {1: 0, 3: 0, 5: 0}
    total = 0
    with torch.no_grad():
        for data in tqdm(samples_by_user.values(), desc="Evaluate GRU", unit="user", leave=False):
            x, t, y = data["test"]
            logits = model(x.to(device), t.to(device))
            pred = torch.argsort(logits, dim=1, descending=True).cpu()
            for k in hits:
                hits[k] += (pred[:, :k] == y.unsqueeze(1)).any(dim=1).sum().item()
            total += len(y)
    return {f"top{k}": hits[k] / total for k in hits}


def measure_prediction_latency(model: torch.nn.Module, samples_by_user: dict, device: torch.device, repeats: int = 200) -> float:
    model.eval()
    first = next(iter(samples_by_user.values()))["test"]
    x, t, _ = first
    x = x[:1].to(device)
    t = t[:1].to(device)
    with torch.no_grad():
        for _ in range(20):
            _ = model(x, t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x, t)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def plot_training_curve(history: pd.DataFrame, out: Path) -> None:
    setup_cn_plot()
    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        linestyle="-",
        color="#111111",
        linewidth=1.8,
        markersize=5,
        label="训练损失",
    )
    ax1.set_xlabel("训练轮次")
    ax1.set_ylabel("损失值")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        history["epoch"],
        history["val_top3"],
        marker="s",
        linestyle="--",
        color="#555555",
        linewidth=1.8,
        markersize=5,
        label="Top-3命中率",
    )
    ax2.set_ylabel("Top-3命中率")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_window_sweep(df: pd.DataFrame, out: Path) -> None:
    setup_cn_plot()
    plt.figure(figsize=(6, 4))
    plt.plot(
        df["window"],
        df["top3"],
        marker="o",
        linestyle="-",
        color="#111111",
        linewidth=1.8,
        markersize=6,
        label="GRU",
    )
    plt.xlabel("窗口长度")
    plt.ylabel("Top-3命中率")
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig(out)


def plot_method_bars(metrics: pd.DataFrame, out: Path) -> None:
    setup_cn_plot()
    method_map = {
        "LRU": "LRU基线",
        "GRU": "GRU",
        "GRU+GlobalComb": "GRU+全局关系表",
        "GRU+UserComb": "GRU+用户关系表",
    }

    sub = metrics[metrics["method"].isin(method_map.keys())].copy()
    sub["method_cn"] = sub["method"].map(method_map)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sub))
    bars = ax.bar(
        x,
        sub["top3"],
        color=["#EEEEEE", "#CCCCCC", "#AAAAAA", "#888888"],
        edgecolor="#111111",
        linewidth=1.0,
    )

    for bar, hatch in zip(bars, BW_HATCHES):
        bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(sub["method_cn"], rotation=20, ha="right")
    ax.set_xlabel("预测方法")
    ax.set_ylabel("Top-3命中率")
    ax.grid(axis="y", alpha=0.3)

    save_fig(out)

def plot_relation_heatmap(df: pd.DataFrame, out: Path, top_n: int = 8) -> None:
    setup_cn_plot()
    top_apps = df["app_name"].value_counts().head(top_n).index.tolist()
    sub = df[df["app_name"].isin(top_apps)].sort_values(["user_id", "timestamp"]).copy()
    sub["next_app"] = sub.groupby(["user_id", "session_id"])["app_name"].shift(-1)
    sub = sub[sub["next_app"].isin(top_apps)]

    mat = (
        pd.crosstab(sub["app_name"], sub["next_app"], normalize="index")
        .reindex(index=top_apps, columns=top_apps)
        .fillna(0)
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat,
        cmap="Greys",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "转移概率"},
        linewidths=0.3,
        linecolor="#DDDDDD",
    )
    plt.xlabel("下一LoRA")
    plt.ylabel("当前LoRA")
    save_fig(out)


def plot_relation_topology(global_seq: pd.DataFrame, out: Path, threshold: float = 0.3) -> None:
    g = nx.DiGraph()
    for _, row in global_seq.iterrows():
        if row["weight"] >= threshold:
            g.add_edge(row["src_lora"], row["dst_lora"], weight=row["weight"])
    plt.figure(figsize=(10, 7))
    if len(g) == 0:
        plt.text(0.5, 0.5, "No edges above threshold", ha="center", va="center")
    else:
        pos = nx.spring_layout(g, seed=42)
        weights = [g[u][v]["weight"] * 3 for u, v in g.edges()]
        nx.draw_networkx_nodes(g, pos, node_size=700, node_color="#9bd3ec")
        nx.draw_networkx_edges(g, pos, width=weights, edge_color="#666666", arrows=True, alpha=0.75)
        nx.draw_networkx_labels(g, pos, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_sweep(df: pd.DataFrame, x: str, y: str, out: Path, xlabel: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Top-3 Hit Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def _to_tensors(ids, times, labels):
    return (
        torch.tensor(np.asarray(ids), dtype=torch.long),
        torch.tensor(np.asarray(times), dtype=torch.float32),
        torch.tensor(np.asarray(labels), dtype=torch.long),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="D:/ecnu_experiment/LoRA/datasets/lsapp.tsv.gz")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "prediction"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-window", type=int, default=15)
    parser.add_argument("--window-sweep", default="5,10,15,20,50")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, encoder = load_lsapp(args.input)
    id_to_name = {i: name for i, name in enumerate(encoder.classes_)}
    name_to_id = {name: i for i, name in id_to_name.items()}
    num_loras = len(id_to_name)
    print(f"[2/9] Loaded records={len(df)} users={df['user_id'].nunique()} loras={num_loras} device={device}", flush=True)

    print("[3/9] Building global relation tables...", flush=True)
    global_seq, global_co = build_global_relation_tables(df)
    print("[4/9] Building user relation tables...", flush=True)
    user_seq, user_co = build_user_relation_tables(df)
    print("[5/9] Plotting relation figures...", flush=True)
    plot_relation_heatmap(df, out_dir / "fig3_3_relation_heatmap.png")
    plot_relation_topology(global_seq, out_dir / "fig3_4_relation_topology.png")

    print(f"[6/9] Building default windows L={args.default_window}...", flush=True)
    samples = build_user_windows(df, args.default_window)
    print(f"[6/9] Training default GRU on {len(samples)} users...", flush=True)
    model, history = train_model(samples, num_loras, args.epochs, args.batch_size, args.lr, device)
    history.to_csv(out_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), out_dir / "gru_global.pt")
    plot_training_curve(history, out_dir / "fig5_1_training_curve.png")

    print("[7/9] Evaluating LRU and fusion methods...", flush=True)
    lru = evaluate_lru(samples)
    fusion = evaluate_model_and_fusion(
        model, samples, id_to_name, name_to_id, global_seq, global_co, user_seq, user_co,
        alpha=0.8, temperature=0.05, device=device, desc="Fusion eval alpha=0.8 T=0.05"
    )
    latency = measure_prediction_latency(model, samples, device)
    rows = [{"method": "LRU", **lru}]
    rows.extend({"method": k, **v} for k, v in fusion.items())
    metrics = pd.DataFrame(rows)
    metrics["prediction_latency_ms"] = np.nan
    metrics.loc[metrics["method"].eq("GRU"), "prediction_latency_ms"] = latency
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    plot_method_bars(metrics, out_dir / "fig5_3_method_topk.png")

    window_rows = []
    windows = [int(x) for x in args.window_sweep.split(",") if x.strip()]
    print(f"[8/9] Running window sweep: {windows}", flush=True)
    for w in tqdm(windows, desc="Window sweep", unit="window"):
        print(f"window sweep: {w}", flush=True)
        s = build_user_windows(df, w)
        m, h = train_model(s, num_loras, max(3, args.epochs // 2), args.batch_size, args.lr, device)
        val = evaluate_model_only(m, s, device)
        window_rows.append({"window": w, **val})
    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(out_dir / "window_sweep.csv", index=False)
    plot_window_sweep(window_df, out_dir / "fig5_2_window_top3.png")

    alpha_rows = []
    print("[9/9] Running alpha and temperature sweeps...", flush=True)
    for alpha in tqdm([0.5, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95], desc="Alpha sweep", unit="alpha"):
        res = evaluate_model_and_fusion(
            model, samples, id_to_name, name_to_id, global_seq, global_co, user_seq, user_co,
            alpha=alpha, temperature=0.05, device=device, desc=f"Fusion alpha={alpha}"
        )
        alpha_rows.append({"alpha": alpha, "top3": res["GRU+UserComb"]["top3"]})
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(out_dir / "alpha_sweep.csv", index=False)
    plot_sweep(alpha_df, "alpha", "top3", out_dir / "fig5_4_alpha_sweep.png", "Fusion Weight alpha")

    temp_rows = []
    for temp in tqdm([0.03, 0.05, 0.1, 0.2, 0.5, 1.0], desc="Temperature sweep", unit="T"):
        res = evaluate_model_and_fusion(
            model, samples, id_to_name, name_to_id, global_seq, global_co, user_seq, user_co,
            alpha=0.8, temperature=temp, device=device, desc=f"Fusion T={temp}"
        )
        temp_rows.append({"temperature": temp, "top3": res["GRU+UserComb"]["top3"]})
    temp_df = pd.DataFrame(temp_rows)
    temp_df.to_csv(out_dir / "temperature_sweep.csv", index=False)
    plot_sweep(temp_df, "temperature", "top3", out_dir / "fig5_5_temperature_sweep.png", "Temperature")

    config = vars(args)
    config.update({"num_loras": num_loras, "num_users": int(df["user_id"].nunique()), "device": str(device)})
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"done. results written to {out_dir}")


if __name__ == "__main__":
    main()
