from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output")
INPUT_CSV = OUTPUT_DIR / "group_node_kv_lora_tree_summary.csv"

OUTPUT_FIG = OUTPUT_DIR / "online_prefix_reuse_compare.png"
OUTPUT_BENEFIT_CSV = OUTPUT_DIR / "online_prefix_reuse_benefit.csv"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
]

LORA_ORDER = [
    "monkey",
    "tiger",
    "wolf",
    "marketing",
]

LORA_NAME = {
    "monkey": "LoRA-monkey",
    "tiger": "LoRA-tiger",
    "wolf": "LoRA-wolf",
    "marketing": "LoRA-marketing",
}

MODE_NAME = {
    "baseline_no_group_kv_reuse": "不复用KV",
    "online_prefix_build": "在线建节点",
    "online_prefix_reuse": "在线复用Prefix KV",
}

MODE_COLOR = {
    "baseline_no_group_kv_reuse": "#8F969E",
    "online_prefix_build": "#D99058",
    "online_prefix_reuse": "#3A947C",
}


def setup_font() -> None:
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
        print("warning: 未找到常见中文字体，中文可能无法正常显示。")

    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到实验结果文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = {
        "mode",
        "group_name",
        "lora_name",
        "n_prompt_tokens",
        "n_prefix_tokens",
        "n_suffix_tokens",
        "prefix_reuse_rate",
        "suffix_delta_rate",
        "ttft_ms",
        "total_ms",
        "tps",
        "gpu_peak_delta_mb",
        "estimated_saved_kv_mb",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    df["lora_label"] = df["lora_name"].map(LORA_NAME).fillna(df["lora_name"])
    df["mode_label"] = df["mode"].map(MODE_NAME).fillna(df["mode"])
    df["prefix_reuse_percent"] = df["prefix_reuse_rate"] * 100.0
    df["suffix_delta_percent"] = df["suffix_delta_rate"] * 100.0
    df["gpu_peak_delta_gb"] = df["gpu_peak_delta_mb"] / 1024.0

    df["lora_order"] = df["lora_name"].apply(
        lambda x: LORA_ORDER.index(x) if x in LORA_ORDER else len(LORA_ORDER)
    )

    return df.sort_values(["lora_order", "mode"]).reset_index(drop=True)


def build_benefit(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["mode"] == "baseline_no_group_kv_reuse"].copy()
    online = df[df["mode"].isin(["online_prefix_build", "online_prefix_reuse"])].copy()

    rows = []

    for _, base in baseline.iterrows():
        matched = online[online["lora_name"] == base["lora_name"]]

        if matched.empty:
            continue

        cur = matched.iloc[0]

        rows.append({
            "group_name": cur["group_name"],
            "lora_name": cur["lora_name"],
            "lora_label": cur["lora_label"],
            "online_mode": cur["mode"],
            "online_mode_label": cur["mode_label"],
            "exact_prefix_hit": int(cur.get("exact_prefix_hit", 0)),
            "online_node_id": int(cur.get("online_node_id", -1)),
            "prompt_tokens": int(cur["n_prompt_tokens"]),
            "prefix_tokens": int(cur["n_prefix_tokens"]),
            "suffix_tokens": int(cur["n_suffix_tokens"]),
            "prefix_reuse_percent": float(cur["prefix_reuse_percent"]),
            "suffix_delta_percent": float(cur["suffix_delta_percent"]),
            "baseline_ttft_ms": float(base["ttft_ms"]),
            "online_ttft_ms": float(cur["ttft_ms"]),
            "ttft_drop_percent": (float(base["ttft_ms"]) - float(cur["ttft_ms"])) / max(float(base["ttft_ms"]), 1e-6) * 100.0,
            "baseline_total_ms": float(base["total_ms"]),
            "online_total_ms": float(cur["total_ms"]),
            "total_drop_percent": (float(base["total_ms"]) - float(cur["total_ms"])) / max(float(base["total_ms"]), 1e-6) * 100.0,
            "baseline_tps": float(base["tps"]),
            "online_tps": float(cur["tps"]),
            "tps_change_percent": (float(cur["tps"]) - float(base["tps"])) / max(float(base["tps"]), 1e-6) * 100.0,
            "estimated_saved_kv_mb": float(cur["estimated_saved_kv_mb"]),
        })

    benefit = pd.DataFrame(rows)
    benefit.to_csv(OUTPUT_BENEFIT_CSV, index=False, encoding="utf-8-sig")
    return benefit


def print_summary(df: pd.DataFrame, benefit: pd.DataFrame) -> None:
    show = df[
        [
            "group_name",
            "lora_label",
            "mode_label",
            "online_node_id",
            "exact_prefix_hit",
            "n_prompt_tokens",
            "n_prefix_tokens",
            "n_suffix_tokens",
            "prefix_reuse_percent",
            "ttft_ms",
            "total_ms",
            "tps",
            "estimated_saved_kv_mb",
        ]
    ].rename(columns={
        "group_name": "组",
        "lora_label": "LoRA",
        "mode_label": "模式",
        "online_node_id": "节点ID",
        "exact_prefix_hit": "是否命中",
        "n_prompt_tokens": "Prompt tokens",
        "n_prefix_tokens": "复用Prefix tokens",
        "n_suffix_tokens": "剩余Suffix tokens",
        "prefix_reuse_percent": "Prefix复用率(%)",
        "ttft_ms": "TTFT(ms)",
        "total_ms": "总耗时(ms)",
        "tps": "TPS",
        "estimated_saved_kv_mb": "估算节省KV(MB)",
    })

    benefit_show = benefit[
        [
            "group_name",
            "lora_label",
            "online_mode_label",
            "ttft_drop_percent",
            "total_drop_percent",
            "tps_change_percent",
            "prefix_tokens",
            "suffix_tokens",
            "estimated_saved_kv_mb",
        ]
    ].rename(columns={
        "group_name": "组",
        "lora_label": "LoRA",
        "online_mode_label": "在线模式",
        "ttft_drop_percent": "TTFT下降(%)",
        "total_drop_percent": "总耗时下降(%)",
        "tps_change_percent": "TPS变化(%)",
        "prefix_tokens": "复用Prefix tokens",
        "suffix_tokens": "剩余Suffix tokens",
        "estimated_saved_kv_mb": "估算节省KV(MB)",
    })

    print("\n在线 Prefix KV 复用实验结果:")
    print(show.to_markdown(index=False, floatfmt=".2f"))

    print("\n相对不复用 KV 的收益:")
    print(benefit_show.to_markdown(index=False, floatfmt=".2f"))


def get_metric_values(df: pd.DataFrame, metric: str) -> tuple[list[str], list[float], list[float]]:
    labels = []
    baseline_values = []
    online_values = []

    baseline = df[df["mode"] == "baseline_no_group_kv_reuse"]
    online = df[df["mode"].isin(["online_prefix_build", "online_prefix_reuse"])]

    for lora_name in LORA_ORDER:
        base = baseline[baseline["lora_name"] == lora_name]
        cur = online[online["lora_name"] == lora_name]

        if base.empty or cur.empty:
            continue

        labels.append(LORA_NAME.get(lora_name, lora_name))
        baseline_values.append(float(base.iloc[0][metric]))
        online_values.append(float(cur.iloc[0][metric]))

    return labels, baseline_values, online_values


def draw_compare_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    fmt: str,
) -> None:
    labels, baseline_values, online_values = get_metric_values(df, metric)

    x = np.arange(len(labels))
    width = 0.36

    bars1 = ax.bar(
        x - width / 2,
        baseline_values,
        width=width,
        label="不复用KV",
        color=MODE_COLOR["baseline_no_group_kv_reuse"],
        alpha=0.94,
    )

    bars2 = ax.bar(
        x + width / 2,
        online_values,
        width=width,
        label="在线Prefix KV",
        color="#3A947C",
        alpha=0.94,
    )

    max_value = max(baseline_values + online_values + [1.0])
    ax.set_ylim(0, max_value * 1.22)

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.018,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, axis="y", alpha=0.28)


def draw_benefit_bar(
    ax: plt.Axes,
    benefit: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    color: str,
    fmt: str,
) -> None:
    labels = benefit["lora_label"].tolist()
    values = benefit[metric].astype(float).tolist()

    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=color,
        alpha=0.94,
    )

    min_value = min(values + [0.0])
    max_value = max(values + [0.0])
    span = max(max_value - min_value, 1.0)

    ax.set_ylim(min(0.0, min_value) - span * 0.2, max(0.0, max_value) + span * 0.24)
    ax.axhline(0, color="#333333", linewidth=0.8)

    for bar, value in zip(bars, values):
        if value >= 0:
            y = value + span * 0.035
            va = "bottom"
        else:
            y = value - span * 0.06
            va = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, axis="y", alpha=0.28)


def draw_prefix_stack(ax: plt.Axes, df: pd.DataFrame) -> None:
    online = df[df["mode"].isin(["online_prefix_build", "online_prefix_reuse"])].copy()
    online = online.sort_values("lora_order")

    labels = online["lora_label"].tolist()
    prefix_values = online["prefix_reuse_percent"].astype(float).tolist()
    suffix_values = (100.0 - online["prefix_reuse_percent"].astype(float)).tolist()

    x = np.arange(len(labels))

    ax.bar(
        x,
        prefix_values,
        label="复用Prefix",
        color="#3A947C",
        width=0.62,
        alpha=0.94,
    )

    ax.bar(
        x,
        suffix_values,
        bottom=prefix_values,
        label="单独计算Suffix",
        color="#D99058",
        width=0.62,
        alpha=0.94,
    )

    for idx, value in enumerate(prefix_values):
        if value > 5:
            ax.text(
                idx,
                value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    for idx, value in enumerate(suffix_values):
        if value > 5:
            ax.text(
                idx,
                prefix_values[idx] + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    ax.set_title("Prefix 复用比例", fontsize=12)
    ax.set_ylabel("Token 占比 / %")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8)


def plot_all(df: pd.DataFrame, benefit: pd.DataFrame) -> None:
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), dpi=180)

    draw_compare_bar(
        axes[0, 0],
        df,
        "ttft_ms",
        "TTFT 对比",
        "TTFT / ms",
        "{:.0f}",
    )

    draw_compare_bar(
        axes[0, 1],
        df,
        "total_ms",
        "总耗时对比",
        "总耗时 / ms",
        "{:.0f}",
    )

    draw_compare_bar(
        axes[0, 2],
        df,
        "tps",
        "TPS 对比",
        "tokens/s",
        "{:.2f}",
    )

    draw_benefit_bar(
        axes[1, 0],
        benefit,
        "ttft_drop_percent",
        "TTFT 下降比例",
        "下降比例 / %",
        "#3A947C",
        "{:.1f}%",
    )

    draw_benefit_bar(
        axes[1, 1],
        benefit,
        "estimated_saved_kv_mb",
        "估算节省 KV 内存",
        "MB",
        "#4C72B0",
        "{:.2f}",
    )

    draw_prefix_stack(axes[1, 2], df)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.965),
    )

    fig.suptitle("在线 Prefix KV 复用实验结果", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(OUTPUT_FIG, bbox_inches="tight")
    print(f"图像已保存到: {OUTPUT_FIG}")


def main() -> None:
    df = load_data()
    benefit = build_benefit(df)

    print_summary(df, benefit)
    plot_all(df, benefit)

    plt.show()

    print(f"收益数据已保存到: {OUTPUT_BENEFIT_CSV}")


if __name__ == "__main__":
    main()