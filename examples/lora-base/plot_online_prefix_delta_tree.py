from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output")
INPUT_CSV = OUTPUT_DIR / "online_prefix_delta_tree_summary.csv"

OUTPUT_FIG = OUTPUT_DIR / "online_prefix_delta_tree.png"
OUTPUT_READABLE = OUTPUT_DIR / "online_prefix_delta_tree_readable.csv"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]


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

    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"找不到输入文件: {INPUT_CSV}\n"
            "请先确认 C++ 里已经调用 save_online_prefix_delta_tree(online_nodes)。"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "node_id",
        "group_id",
        "group_name",
        "parent_node_id",
        "delta_parent_node_id",
        "prefix_len",
        "hit_count",
        "prefix_can_reuse",
        "suffix_can_delta",
        "should_open_new_branch",
        "suffix_kv_cos",
        "suffix_kv_l2",
        "reuse_decision",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    df["node_label"] = (
        "node "
        + df["node_id"].astype(str)
        + "\n"
        + df["group_name"].astype(str)
    )

    df["decision_label"] = df["reuse_decision"].replace(
        {
            "prefix_reuse_suffix_delta": "Prefix+Delta候选",
            "prefix_reuse_suffix_recompute": "Prefix复用+Suffix单算",
            "new_branch": "单独分支",
            "create_prefix_node": "创建Prefix节点",
            "exact_prefix_reuse_suffix_recompute": "Prefix复用+Suffix单算",
        }
    )

    df.to_csv(OUTPUT_READABLE, index=False, encoding="utf-8-sig")

    return df


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}") -> None:
    heights = [bar.get_height() for bar in bars]
    ymax = max(heights + [1.0])

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.02,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_prefix_len(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = df["node_label"].tolist()
    x = np.arange(len(labels))

    bars = ax.bar(x, df["prefix_len"], color="#2E8B72", width=0.62)
    add_bar_labels(ax, bars, fmt="{:.0f}")

    ax.set_title("Online Prefix 节点长度")
    ax.set_ylabel("Prefix tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot_hit_count(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = df["node_label"].tolist()
    x = np.arange(len(labels))

    bars = ax.bar(x, df["hit_count"], color="#356D9A", width=0.62)
    add_bar_labels(ax, bars, fmt="{:.0f}")

    ax.set_title("Online Prefix 节点命中次数")
    ax.set_ylabel("Hit count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot_suffix_probe(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = df["node_label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df["suffix_kv_cos"],
        width=width,
        color="#D88A48",
        label="Suffix KV Cos",
    )

    bars2 = ax.bar(
        x + width / 2,
        df["suffix_kv_l2"],
        width=width,
        color="#8F969E",
        label="Suffix KV L2",
    )

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)

    ax.axhline(0.80, color="#B84040", linestyle="--", linewidth=1.2, label="Cos阈值0.80")
    ax.set_title("节点上的 Suffix Delta Probe 指标")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot_parent_relation(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = df["node_label"].tolist()
    x = np.arange(len(labels))

    values = np.where(df["delta_parent_node_id"].astype(int) >= 0, 1, 0)
    colors = np.where(values == 1, "#2E8B72", "#D88A48")

    bars = ax.bar(x, values, color=colors, width=0.62)

    for bar, has_parent, decision in zip(bars, values, df["decision_label"].tolist()):
        text = "挂到delta父节点" if has_parent else decision
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.5 if has_parent else 0.08,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if has_parent else "#333333",
        )

    ax.set_title("Delta Parent 映射结果")
    ax.set_ylabel("是否存在 delta_parent")
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["无", "有"])
    ax.grid(True, axis="y", alpha=0.25)


def plot_all(df: pd.DataFrame) -> None:
    setup_font()

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.8), dpi=180)

    plot_prefix_len(axes[0, 0], df)
    plot_hit_count(axes[0, 1], df)
    plot_suffix_probe(axes[1, 0], df)
    plot_parent_relation(axes[1, 1], df)

    fig.suptitle(
        "Online Prefix Tree 与 KV Delta Probe 决策映射",
        fontsize=15,
        y=0.995,
    )

    fig.text(
        0.5,
        0.01,
        "说明：delta_parent_node_id 表示当前节点可以挂到哪个 anchor prefix 节点下；这仍是probe决策，不代表已真实写回delta KV。",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(OUTPUT_FIG, bbox_inches="tight")

    print(f"Online tree 图已保存到: {OUTPUT_FIG}")


def main() -> None:
    df = load_data()

    print("\nOnline Prefix Delta Tree:")
    show = df[
        [
            "node_id",
            "group_name",
            "parent_node_id",
            "delta_parent_node_id",
            "prefix_len",
            "hit_count",
            "prefix_can_reuse",
            "suffix_can_delta",
            "suffix_kv_cos",
            "suffix_kv_l2",
            "decision_label",
        ]
    ]

    try:
        print(show.to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        print(show.to_string(index=False))

    print(f"\n可读表格已保存到: {OUTPUT_READABLE}")

    plot_all(df)
    plt.show()


if __name__ == "__main__":
    main()