from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output")
INPUT_CSV = OUTPUT_DIR / "group_node_kv_lora_tree_summary.csv"

OUTPUT_FIG = OUTPUT_DIR / "runtime_kv_reuse_cost.png"
OUTPUT_READABLE = OUTPUT_DIR / "runtime_kv_reuse_readable.csv"

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
        raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {
        "mode",
        "group_name",
        "lora_name",
        "reuse_decision",
        "online_node_id",
        "exact_prefix_hit",
        "n_prompt_tokens",
        "n_prefix_tokens",
        "n_suffix_tokens",
        "prefix_reuse_rate",
        "prefix_ms",
        "suffix_ms",
        "ttft_ms",
        "decode_ms",
        "total_ms",
        "estimated_saved_kv_mb",
        "gpu_peak_delta_mb",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    df["request_label"] = (
        df["mode"].astype(str)
        + "\n"
        + df["group_name"].astype(str)
        + "/"
        + df["lora_name"].astype(str)
    )

    df["is_online"] = df["mode"].str.contains("online_prefix", na=False)
    df["is_baseline"] = df["mode"].eq("baseline_no_group_kv_reuse")

    readable = df[
        [
            "mode",
            "group_name",
            "lora_name",
            "reuse_decision",
            "online_node_id",
            "exact_prefix_hit",
            "n_prompt_tokens",
            "n_prefix_tokens",
            "n_suffix_tokens",
            "prefix_reuse_rate",
            "prefix_ms",
            "suffix_ms",
            "ttft_ms",
            "decode_ms",
            "total_ms",
            "estimated_saved_kv_mb",
            "gpu_peak_delta_mb",
        ]
    ].copy()

    readable.to_csv(OUTPUT_READABLE, index=False, encoding="utf-8-sig")

    return df


def short_labels(df: pd.DataFrame) -> list[str]:
    labels = []

    for _, row in df.iterrows():
        mode = str(row["mode"])
        lora = str(row["lora_name"])

        if mode == "baseline_no_group_kv_reuse":
            prefix = "baseline"
        elif mode == "online_prefix_build":
            prefix = "build"
        elif mode == "online_prefix_reuse":
            prefix = "reuse"
        else:
            prefix = mode

        labels.append(f"{prefix}\n{lora}")

    return labels


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
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


def plot_ttft(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df)
    x = np.arange(len(labels))

    colors = np.where(df["is_online"], "#2E8B72", "#8F969E")

    bars = ax.bar(x, df["ttft_ms"], color=colors, width=0.62)
    add_bar_labels(ax, bars)

    ax.set_title("真实运行 TTFT 对比")
    ax.set_ylabel("TTFT / ms")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def plot_prefill_breakdown(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df)
    x = np.arange(len(labels))

    prefix_ms = df["prefix_ms"].to_numpy()
    suffix_ms = df["suffix_ms"].to_numpy()

    ax.bar(x, prefix_ms, label="prefix copy/build", color="#2E8B72", width=0.62)
    ax.bar(x, suffix_ms, bottom=prefix_ms, label="suffix recompute", color="#D88A48", width=0.62)

    for i, v in enumerate(prefix_ms):
        if v > 1:
            ax.text(i, v / 2, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white")

    for i, v in enumerate(suffix_ms):
        if v > 1:
            ax.text(i, prefix_ms[i] + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white")

    ax.set_title("真实运行 Prefix / Suffix 时间拆分")
    ax.set_ylabel("Time / ms")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot_saved_kv(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df)
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        df["estimated_saved_kv_mb"],
        color="#356D9A",
        width=0.62,
    )

    add_bar_labels(ax, bars, fmt="{:.3f}")

    ax.set_title("Prefix KV 复用带来的估算 KV 节省")
    ax.set_ylabel("Estimated saved KV / MB")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def plot_token_reuse(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = short_labels(df)
    x = np.arange(len(labels))

    prefix_rate = df["prefix_reuse_rate"].to_numpy() * 100.0
    suffix_rate = 100.0 - prefix_rate

    ax.bar(x, prefix_rate, label="reused prefix", color="#2E8B72", width=0.62)
    ax.bar(x, suffix_rate, bottom=prefix_rate, label="computed suffix", color="#D88A48", width=0.62)

    for i, v in enumerate(prefix_rate):
        if v > 5:
            ax.text(i, v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=7, color="white")

    ax.set_title("真实请求中的 Prefix 复用比例")
    ax.set_ylabel("Token ratio / %")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot_all(df: pd.DataFrame) -> None:
    setup_font()

    # 只画 baseline 和 online，避免其它旧实验模式混进来。
    plot_df = df[
        df["mode"].isin(
            [
                "baseline_no_group_kv_reuse",
                "online_prefix_build",
                "online_prefix_reuse",
            ]
        )
    ].copy()

    if plot_df.empty:
        raise ValueError("没有可画的 runtime 数据，请先运行 llama-lora-base 生成 CSV。")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.8), dpi=180)

    plot_ttft(axes[0, 0], plot_df)
    plot_prefill_breakdown(axes[0, 1], plot_df)
    plot_saved_kv(axes[1, 0], plot_df)
    plot_token_reuse(axes[1, 1], plot_df)

    fig.suptitle(
        "多LoRA Prefix KV 复用真实运行开销图",
        fontsize=15,
        y=0.995,
    )

    fig.text(
        0.5,
        0.01,
        "说明：TTFT/prefix_ms/suffix_ms 来自真实运行日志；estimated_saved_kv_mb 是按复用 token 数和模型 KV 大小估算的节省。",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(OUTPUT_FIG, bbox_inches="tight")

    print(f"真实运行开销图已保存到: {OUTPUT_FIG}")


def main() -> None:
    df = load_data()

    print("\nRuntime KV Reuse 结果:")
    show = df[
        [
            "mode",
            "group_name",
            "lora_name",
            "reuse_decision",
            "n_prompt_tokens",
            "n_prefix_tokens",
            "n_suffix_tokens",
            "prefix_ms",
            "suffix_ms",
            "ttft_ms",
            "total_ms",
            "estimated_saved_kv_mb",
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