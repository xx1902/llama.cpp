#!/usr/bin/env python3
"""Summarize and visualize the three 87-LoRA experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(r"D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test")
OUTPUT = Path(__file__).resolve().parent / "experiment_87_combined"

COLORS = {
    "baseline": "#6B7280",
    "online": "#2878B5",
    "ingroup": "#2A9D8F",
    "crossgroup": "#E76F51",
    "key": "#3A86FF",
    "value": "#F4A261",
    "probe": "#E9C46A",
    "build": "#E15759",
}


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "axes.edgecolor": "#CBD5E1",
            "grid.color": "#DCE3EA",
            "grid.alpha": 0.55,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png")
    fig.savefig(OUTPUT / f"{name}.pdf")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars, fmt: str = ".1f") -> None:
    ax.bar_label(bars, labels=[format(bar.get_height(), fmt) for bar in bars], padding=4)


def summarize_probe(probe: pd.DataFrame, label: str) -> dict[str, float | str]:
    worst = probe.loc[probe["kv_cos"].idxmin()]
    return {
        "experiment": label,
        "pairs": len(probe),
        "ok_rate": (probe["status"] == "ok").mean(),
        "can_encode_rate": (probe["can_encode_delta"] == 1).mean(),
        "kv_cos_mean": probe["kv_cos"].mean(),
        "kv_cos_median": probe["kv_cos"].median(),
        "kv_cos_p05": probe["kv_cos"].quantile(0.05),
        "kv_l2_mean": probe["kv_l2"].mean(),
        "kv_l2_median": probe["kv_l2"].median(),
        "kv_l2_p95": probe["kv_l2"].quantile(0.95),
        "anchor_eval_ms": probe["anchor_eval_ms"].mean(),
        "child_eval_ms": probe["child_eval_ms"].mean(),
        "probe_ms": probe["probe_ms"].mean(),
        "delta_build_ms": probe["delta_build_ms"].mean(),
        "logical_saved_rate": probe["logical_saved_rate"].mean(),
        "worst_pair": worst["pair_name"],
        "worst_kv_cos": worst["kv_cos"],
        "worst_kv_l2": worst["kv_l2"],
    }


def plot_online(online: pd.DataFrame) -> None:
    order = ["baseline", "online"]
    grouped = online.groupby("benchmark")
    mean_ttft = [grouped.get_group(name)["ttft_ms"].mean() for name in order]
    median_ttft = [grouped.get_group(name)["ttft_ms"].median() for name in order]
    p95_ttft = [grouped.get_group(name)["ttft_ms"].quantile(0.95) for name in order]

    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    baseline_bars = ax.bar(
        x - width / 2,
        [mean_ttft[0], median_ttft[0], p95_ttft[0]],
        width,
        label="Baseline",
        color=COLORS["baseline"],
    )
    online_bars = ax.bar(
        x + width / 2,
        [mean_ttft[1], median_ttft[1], p95_ttft[1]],
        width,
        label="Online prefix tree",
        color=COLORS["online"],
    )
    add_bar_labels(ax, baseline_bars, ".0f")
    add_bar_labels(ax, online_bars, ".0f")
    ax.set_xticks(x, ["Mean", "Median", "P95"])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Online exact-prefix reuse reduces TTFT")
    ax.legend()
    save(fig, "01_online_ttft")


def plot_delta_distributions(in_group: pd.DataFrame, cross_group: pd.DataFrame) -> None:
    datasets = [in_group, cross_group]
    labels = ["In-group", "Cross-group"]
    colors = [COLORS["ingroup"], COLORS["crossgroup"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, column, title, ylabel in [
        (axes[0], "kv_cos", "KV cosine distribution", "Cosine similarity"),
        (axes[1], "kv_l2", "KV L2 distribution", "Normalized L2 difference"),
    ]:
        violin = ax.violinplot(
            [data[column].to_numpy() for data in datasets],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
            widths=0.75,
        )
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.72)
        for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            violin[key].set_color("#334155")
            violin[key].set_linewidth(1.1)
        ax.set_xticks([1, 2], labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        means = [data[column].mean() for data in datasets]
        for pos, mean in enumerate(means, 1):
            ax.text(pos, mean, f"  mean={mean:.4f}", va="center", fontsize=9)

    fig.suptitle("Grouping improves delta similarity, but distributions still overlap", fontweight="bold")
    save(fig, "02_ingroup_vs_crossgroup_distribution")


def plot_delta_timing(in_group: pd.DataFrame, cross_group: pd.DataFrame) -> None:
    stages = ["Anchor eval", "Child eval", "Probe", "Delta build"]
    columns = ["anchor_eval_ms", "child_eval_ms", "probe_ms", "delta_build_ms"]
    in_values = [in_group[column].mean() for column in columns]
    cross_values = [cross_group[column].mean() for column in columns]
    x = np.arange(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    bars_in = ax.bar(x - width / 2, in_values, width, label="In-group", color=COLORS["ingroup"])
    bars_cross = ax.bar(x + width / 2, cross_values, width, label="Cross-group", color=COLORS["crossgroup"])
    add_bar_labels(ax, bars_in, ".0f")
    add_bar_labels(ax, bars_cross, ".0f")
    ax.set_xticks(x, stages)
    ax.set_ylabel("Mean time per pair (ms)")
    ax.set_title("Synchronous delta construction is too expensive for the request path")
    ax.legend()
    save(fig, "03_delta_timing")


def plot_layerwise(in_layers: pd.DataFrame, cross_layers: pd.DataFrame) -> None:
    in_mean = in_layers.groupby("layer_id")[["k_l2", "v_l2"]].mean()
    cross_mean = cross_layers.groupby("layer_id")[["k_l2", "v_l2"]].mean()
    x = in_mean.index.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    axes[0].plot(x, in_mean["k_l2"], color=COLORS["ingroup"], linewidth=2.2, label="In-group")
    axes[0].plot(x, cross_mean["k_l2"], color=COLORS["crossgroup"], linewidth=2.2, linestyle="--", label="Cross-group")
    axes[0].set_title("Key-cache L2 by layer")
    axes[0].set_ylabel("Normalized L2 difference")
    axes[0].set_xlabel("Transformer layer")
    axes[0].legend()

    axes[1].plot(x, in_mean["v_l2"], color=COLORS["ingroup"], linewidth=2.2, label="In-group")
    axes[1].plot(x, cross_mean["v_l2"], color=COLORS["crossgroup"], linewidth=2.2, linestyle="--", label="Cross-group")
    axes[1].fill_between(x, in_mean["v_l2"], cross_mean["v_l2"], color=COLORS["crossgroup"], alpha=0.10)
    axes[1].set_title("Value-cache L2 by layer")
    axes[1].set_xlabel("Transformer layer")
    axes[1].legend()
    fig.suptitle("Cross-group Value cache diverges most in later layers", fontweight="bold")
    save(fig, "04_layerwise_ingroup_vs_crossgroup")


def write_report(
    online: pd.DataFrame,
    in_group: pd.DataFrame,
    cross_group: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    baseline = online[online["benchmark"] == "baseline"]
    optimized = online[online["benchmark"] == "online"]
    exact = optimized[optimized["mode"] == "reuse_exact_prefix_same_lora"]
    baseline_by_request = baseline[["request_id", "ttft_ms"]].rename(columns={"ttft_ms": "baseline_ttft_ms"})
    paired = exact.merge(baseline_by_request, on="request_id")

    online_summary = pd.DataFrame(
        [
            {
                "benchmark": "baseline",
                "requests": len(baseline),
                "mean_ttft_ms": baseline["ttft_ms"].mean(),
                "median_ttft_ms": baseline["ttft_ms"].median(),
                "p95_ttft_ms": baseline["ttft_ms"].quantile(0.95),
                "mean_total_ms": baseline["total_ms"].mean(),
            },
            {
                "benchmark": "online_off",
                "requests": len(optimized),
                "mean_ttft_ms": optimized["ttft_ms"].mean(),
                "median_ttft_ms": optimized["ttft_ms"].median(),
                "p95_ttft_ms": optimized["ttft_ms"].quantile(0.95),
                "mean_total_ms": optimized["total_ms"].mean(),
            },
        ]
    )
    online_summary.to_csv(OUTPUT / "online_summary.csv", index=False)
    summaries.to_csv(OUTPUT / "delta_summary.csv", index=False)

    l2_reduction = 1.0 - in_group["kv_l2"].mean() / cross_group["kv_l2"].mean()
    report = f"""# 87-LoRA 三组实验结果整理

## 1. 在线前缀树基线（output_87_off）

- 共 300 个请求，每个请求分别执行 baseline 和 online，共 600 条结果。
- Mean TTFT：{baseline['ttft_ms'].mean():.2f} ms -> {optimized['ttft_ms'].mean():.2f} ms，降低 {(1 - optimized['ttft_ms'].mean() / baseline['ttft_ms'].mean()) * 100:.2f}%。
- Median TTFT：{baseline['ttft_ms'].median():.2f} ms -> {optimized['ttft_ms'].median():.2f} ms，降低 {(1 - optimized['ttft_ms'].median() / baseline['ttft_ms'].median()) * 100:.2f}%。
- P95 TTFT：{baseline['ttft_ms'].quantile(0.95):.2f} ms -> {optimized['ttft_ms'].quantile(0.95):.2f} ms，降低 {(1 - optimized['ttft_ms'].quantile(0.95) / baseline['ttft_ms'].quantile(0.95)) * 100:.2f}%。
- 95 次同 LoRA 精确前缀命中中，配对 Mean TTFT 从 {paired['baseline_ttft_ms'].mean():.2f} ms 降至 {paired['ttft_ms'].mean():.2f} ms，降低 {(1 - paired['ttft_ms'].mean() / paired['baseline_ttft_ms'].mean()) * 100:.2f}%，约 {paired['baseline_ttft_ms'].mean() / paired['ttft_ms'].mean():.2f}x 加速。

## 2. 组内 Delta 探测（output_87_delta_ingroup）

- 385 对均探测成功，平均 KV cosine 为 {in_group['kv_cos'].mean():.6f}，平均 KV L2 为 {in_group['kv_l2'].mean():.6f}。
- 平均 probe 为 {in_group['probe_ms'].mean():.2f} ms，平均 delta build 为 {in_group['delta_build_ms'].mean():.2f} ms。
- 当前逻辑存储节省率固定为 {in_group['logical_saved_rate'].mean() * 100:.4f}%。

## 3. 跨组 Delta 对照（output_87_delta_crossgroup）

- 100 对均探测成功，平均 KV cosine 为 {cross_group['kv_cos'].mean():.6f}，平均 KV L2 为 {cross_group['kv_l2'].mean():.6f}。
- 平均 probe 为 {cross_group['probe_ms'].mean():.2f} ms，平均 delta build 为 {cross_group['delta_build_ms'].mean():.2f} ms。
- 最差 pair 为 `{cross_group.loc[cross_group['kv_cos'].idxmin(), 'pair_name']}`，cosine={cross_group['kv_cos'].min():.6f}，L2={cross_group.loc[cross_group['kv_cos'].idxmin(), 'kv_l2']:.6f}。

## 4. 合并结论

- 组内平均 cosine 比跨组高 {in_group['kv_cos'].mean() - cross_group['kv_cos'].mean():.6f}。
- 组内平均 L2 比跨组低 {cross_group['kv_l2'].mean() - in_group['kv_l2'].mean():.6f}，相对降低 {l2_reduction * 100:.2f}%。这说明分组具有统计意义。
- 两组分布仍有明显重叠，因此 group 适合缩小候选范围，不能直接作为是否构建 delta 的最终条件。
- 当前所有 pair 的 `can_encode_delta` 都为 1，且节省率恒为 49.2188%，说明当前判断和存储模型还不能根据实际相似度区分优劣。
- 同步 probe + delta build 的平均开销约为组内 {(in_group['probe_ms'].mean() + in_group['delta_build_ms'].mean()) / 1000:.2f} s、跨组 {(cross_group['probe_ms'].mean() + cross_group['delta_build_ms'].mean()) / 1000:.2f} s，不应放在在线请求关键路径，应改为离线、异步或预测驱动的预计算。
"""
    (OUTPUT / "三组实验结果整理.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    online = pd.read_csv(BASE / "output_87_off" / "online_request_results.csv")
    in_group = pd.read_csv(BASE / "output_87_delta_ingroup" / "delta_prefix_probe.csv")
    cross_group = pd.read_csv(BASE / "output_87_delta_crossgroup" / "delta_prefix_probe.csv")
    in_layers = pd.read_csv(BASE / "output_87_delta_ingroup" / "delta_prefix_layers.csv")
    cross_layers = pd.read_csv(BASE / "output_87_delta_crossgroup" / "delta_prefix_layers.csv")

    summaries = pd.DataFrame(
        [summarize_probe(in_group, "in_group"), summarize_probe(cross_group, "cross_group")]
    )
    setup_style()
    plot_online(online)
    plot_delta_distributions(in_group, cross_group)
    plot_delta_timing(in_group, cross_group)
    plot_layerwise(in_layers, cross_layers)
    write_report(online, in_group, cross_group, summaries)
    print(f"Combined report saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
