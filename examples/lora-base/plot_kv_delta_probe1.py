from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output")
INPUT_CSV = OUTPUT_DIR / "kv_delta_probe_summary.csv"

OUTPUT_FIG = OUTPUT_DIR / "kv_delta_probe_compare.png"
OUTPUT_READABLE = OUTPUT_DIR / "kv_delta_probe_readable.csv"
OUTPUT_VALID = OUTPUT_DIR / "kv_delta_probe_valid_only.csv"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]

PAIR_NAME = {
    "monkey_vs_tiger": "monkey vs tiger",
    "monkey_vs_wolf": "monkey vs wolf",
    "tiger_cn_vs_tiger_en": "tiger prompt A/B",
    "same_prompt_monkey": "same prompt sanity",
    "same_prefix_monkey_tiger_no_lora": "same prefix animal",
    "same_article_summary_vs_rewrite": "article summarize/rewrite",
    "same_product_summary_vs_optimize": "product summarize/optimize",
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

    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {
        "pair_name",
        "group_name",
        "memory_kind",
        "probe_status",
        "tokens_a",
        "tokens_b",
        "common_prefix_tokens",
        "compared_tokens",
        "probed_layers",
        "full_kv_l2_avg",
        "full_kv_cos_avg",
        "prefix_kv_l2_avg",
        "prefix_kv_cos_avg",
        "suffix_kv_l2_avg",
        "suffix_kv_cos_avg",
        "prefix_can_reuse",
        "suffix_can_delta",
        "should_open_new_branch",
        "reuse_decision",
        "tree_action",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    df["pair_label"] = df["pair_name"].map(PAIR_NAME).fillna(df["pair_name"])

    compared = df["compared_tokens"].replace(0, np.nan)
    df["prefix_token_rate"] = (df["common_prefix_tokens"] / compared).fillna(0.0) * 100.0
    df["suffix_token_rate"] = 100.0 - df["prefix_token_rate"]

    df["is_valid"] = df["probe_status"].eq("ok")
    df["is_sanity"] = df["group_name"].eq("sanity")

    # df["decision"] = np.select(
    #     [
    #         ~df["is_valid"],
    #         df["should_open_new_branch"].astype(int).eq(1),
    #         df["suffix_can_delta"].astype(int).eq(1),
    #         df["prefix_can_reuse"].astype(int).eq(1),
    #     ],
    #     [
    #         "异常/跳过",
    #         "单独开分支",
    #         "Prefix复用+Suffix delta",
    #         "只复用Prefix",
    #     ],
    #     default="待判断",
    # )

    df["decision"] = df["reuse_decision"].replace(
        {
            "prefix_reuse_suffix_delta": "Prefix复用+Suffix delta",
            "prefix_reuse_suffix_recompute": "Prefix复用+Suffix单独计算",
            "new_branch": "单独开分支",
            "suffix_delta_only": "仅Suffix delta",
        }
    )

    readable = df[
        [
            "pair_label",
            "group_name",
            "memory_kind",
            "probe_status",
            "tokens_a",
            "tokens_b",
            "common_prefix_tokens",
            "compared_tokens",
            "prefix_token_rate",
            "suffix_token_rate",
            "probed_layers",
            "full_kv_cos_avg",
            "prefix_kv_cos_avg",
            "suffix_kv_cos_avg",
            "full_kv_l2_avg",
            "prefix_kv_l2_avg",
            "suffix_kv_l2_avg",
            "prefix_can_reuse",
            "suffix_can_delta",
            "should_open_new_branch",
            "decision",
        ]
    ].copy()

    readable.to_csv(OUTPUT_READABLE, index=False, encoding="utf-8-sig")

    valid_df = df[df["is_valid"]].copy()
    valid_df.to_csv(OUTPUT_VALID, index=False, encoding="utf-8-sig")

    return df


def print_summary(df: pd.DataFrame) -> None:
    show = df[
        [
            "pair_label",
            "group_name",
            "probe_status",
            "common_prefix_tokens",
            "compared_tokens",
            "prefix_token_rate",
            "probed_layers",
            "full_kv_cos_avg",
            "prefix_kv_cos_avg",
            "suffix_kv_cos_avg",
            "full_kv_l2_avg",
            "prefix_kv_l2_avg",
            "suffix_kv_l2_avg",
            "decision",
        ]
    ].rename(
        columns={
            "pair_label": "Pair",
            "group_name": "Group",
            "probe_status": "状态",
            "common_prefix_tokens": "公共Prefix tokens",
            "compared_tokens": "比较tokens",
            "prefix_token_rate": "Prefix比例(%)",
            "probed_layers": "Probe层数",
            "full_kv_cos_avg": "Full Cos",
            "prefix_kv_cos_avg": "Prefix Cos",
            "suffix_kv_cos_avg": "Suffix Cos",
            "full_kv_l2_avg": "Full L2",
            "prefix_kv_l2_avg": "Prefix L2",
            "suffix_kv_l2_avg": "Suffix L2",
            "decision": "决策",
        }
    )

    print("\nKV Delta Probe 总结果:")
    try:
        print(show.to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        print(show.to_string(index=False))

    bad_df = df[~df["is_valid"]]
    if not bad_df.empty:
        print("\n异常样本，不建议放入正式图:")
        print(bad_df[["pair_name", "probe_status", "group_name"]].to_string(index=False))

    sanity_df = df[df["is_sanity"]]
    if not sanity_df.empty:
        bad_sanity = sanity_df[
            (sanity_df["full_kv_cos_avg"] < 0.99) |
            (sanity_df["full_kv_l2_avg"] > 1e-2)
        ]
        if not bad_sanity.empty:
            print("\n警告：sanity check 未通过，当前 KV delta 结果只能作为调试参考，不能作为最终实验结论。")
        print("\nSanity Check:")
        print(
            sanity_df[
                [
                    "pair_name",
                    "probe_status",
                    "full_kv_cos_avg",
                    "prefix_kv_cos_avg",
                    "full_kv_l2_avg",
                    "prefix_kv_l2_avg",
                ]
            ].to_string(index=False)
        )

    print(f"\n可读表格已保存到: {OUTPUT_READABLE}")
    print(f"有效样本已保存到: {OUTPUT_VALID}")


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}") -> None:
    heights = [bar.get_height() for bar in bars]
    max_value = max(heights + [1.0])

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max_value * 0.02,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )


def short_labels(labels: list[str]) -> list[str]:
    result = []
    for label in labels:
        label = label.replace("same prefix animal", "prefix animal")
        label = label.replace("article summarize/rewrite", "article task")
        label = label.replace("product summarize/optimize", "product task")
        label = label.replace("same prompt sanity", "sanity")
        result.append(label)
    return result


def plot_cosine(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df["pair_label"].tolist())
    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        df["full_kv_cos_avg"],
        width=width,
        label="Full KV",
        color="#8F969E",
        alpha=0.95,
    )
    bars2 = ax.bar(
        x,
        df["prefix_kv_cos_avg"],
        width=width,
        label="Prefix KV",
        color="#2E8B72",
        alpha=0.95,
    )
    bars3 = ax.bar(
        x + width,
        df["suffix_kv_cos_avg"],
        width=width,
        label="Suffix KV",
        color="#D88A48",
        alpha=0.95,
    )

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    add_bar_labels(ax, bars3)

    ax.axhline(0.80, color="#B84040", linestyle="--", linewidth=1.2, label="delta阈值 0.80")
    ax.set_title("KV Cosine 相似度")
    ax.set_ylabel("Cosine")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def plot_l2(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df["pair_label"].tolist())
    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        df["full_kv_l2_avg"],
        width=width,
        label="Full KV",
        color="#8F969E",
        alpha=0.95,
    )
    bars2 = ax.bar(
        x,
        df["prefix_kv_l2_avg"],
        width=width,
        label="Prefix KV",
        color="#2E8B72",
        alpha=0.95,
    )
    bars3 = ax.bar(
        x + width,
        df["suffix_kv_l2_avg"],
        width=width,
        label="Suffix KV",
        color="#D88A48",
        alpha=0.95,
    )

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    add_bar_labels(ax, bars3)

    ymax = max(
        df["full_kv_l2_avg"].max(),
        df["prefix_kv_l2_avg"].max(),
        df["suffix_kv_l2_avg"].max(),
        0.1,
    )

    ax.axhline(0.05, color="#B84040", linestyle="--", linewidth=1.2, label="L2阈值 0.05")
    ax.set_title("KV L2 数值差异")
    ax.set_ylabel("L2")
    ax.set_ylim(0.0, ymax * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def plot_token_rate(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df["pair_label"].tolist())
    x = np.arange(len(labels))

    prefix = df["prefix_token_rate"].to_numpy()
    suffix = df["suffix_token_rate"].to_numpy()

    ax.bar(
        x,
        prefix,
        label="公共Prefix tokens",
        color="#2E8B72",
        width=0.62,
        alpha=0.95,
    )

    ax.bar(
        x,
        suffix,
        bottom=prefix,
        label="差异Suffix tokens",
        color="#D88A48",
        width=0.62,
        alpha=0.95,
    )

    for i, value in enumerate(prefix):
        if value > 5:
            ax.text(
                i,
                value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    for i, value in enumerate(suffix):
        if value > 5:
            ax.text(
                i,
                prefix[i] + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    ax.set_title("Token Prefix / Suffix 比例")
    ax.set_ylabel("Token占比 / %")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def plot_decision(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df["pair_label"].tolist())
    x = np.arange(len(labels))

    values = df["should_open_new_branch"].astype(int).to_numpy()

    colors = [
        "#B84040" if v == 1 else "#2E8B72"
        for v in values
    ]

    bars = ax.bar(
        x,
        values,
        color=colors,
        width=0.62,
        alpha=0.95,
    )

    for bar, value, decision in zip(bars, values, df["decision"].tolist()):
        text = "单独分支" if value == 1 else "可复用"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.55 if value == 1 else 0.08,
            text,
            ha="center",
            va="center",
            color="white" if value == 1 else "#333333",
            fontsize=8,
        )

    ax.set_title("Delta 分支决策")
    ax.set_ylabel("是否新开分支")
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["复用/挂delta", "单独开分支"])
    ax.grid(True, axis="y", alpha=0.25)


def plot_all(df: pd.DataFrame) -> None:
    setup_font()

    plot_df = df[
        (df["probe_status"] == "ok") &
        (df["group_name"] != "sanity")
    ].copy()

    if plot_df.empty:
        raise ValueError("没有可用于画图的有效样本。请检查 probe_status 是否为 ok。")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.8), dpi=180)

    plot_cosine(axes[0, 0], plot_df)
    plot_l2(axes[0, 1], plot_df)
    plot_token_rate(axes[1, 0], plot_df)
    plot_decision(axes[1, 1], plot_df)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        fontsize=9,
    )

    fig.suptitle(
        "KV Delta Probe 分段结果（仅展示 probe_status=ok，sanity 单独检查）",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_FIG, bbox_inches="tight")

    print(f"图像已保存到: {OUTPUT_FIG}")


def main() -> None:
    df = load_data()
    print_summary(df)
    plot_all(df)
    plt.show()


if __name__ == "__main__":
    main()