from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
]

OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output")

INPUT_CSV = OUTPUT_DIR / "group_node_kv_lora_tree_summary.csv"
BENEFIT_CSV = OUTPUT_DIR / "group_node_kv_lora_tree_benefit.csv"
OUTPUT_FIG = OUTPUT_DIR / "group_node_kv_lora_tree_six_compare.png"

MODE_NAME = {
    "baseline_no_group_kv_reuse": "不复用KV",
    "group_node_kv_reuse_lora_tree": "Group KV复用+LoRA树",
}

MODE_ORDER = [
    "baseline_no_group_kv_reuse",
    "group_node_kv_reuse_lora_tree",
]

MODE_COLOR = {
    "baseline_no_group_kv_reuse": "#8A9099",
    "group_node_kv_reuse_lora_tree": "#2E8B73",
}

LORA_NAME = {
    "monkey": "LoRA-monkey",
    "tiger": "LoRA-tiger",
    "wolf": "LoRA-wolf",
    "marketing": "LoRA-marketing",
}

LORA_ORDER = [
    "monkey",
    "tiger",
    "wolf",
    "marketing",
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
    else:
        print("warning: 未找到常见中文字体，中文可能无法正常显示。")

    plt.rcParams["axes.unicode_minus"] = False


def load_results() -> pd.DataFrame:
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
        "ttft_ms",
        "total_ms",
        "tps",
        "gpu_peak_delta_mb",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    if "prefix_reuse_rate" not in df.columns:
        df["prefix_reuse_rate"] = df["n_prefix_tokens"] / df["n_prompt_tokens"].replace(0, np.nan)

    if "suffix_delta_rate" not in df.columns:
        df["suffix_delta_rate"] = df["n_suffix_tokens"] / df["n_prompt_tokens"].replace(0, np.nan)

    df["prefix_reuse_rate"] = df["prefix_reuse_rate"].fillna(0.0)
    df["suffix_delta_rate"] = df["suffix_delta_rate"].fillna(0.0)

    df["prefix_reuse_percent"] = df["prefix_reuse_rate"] * 100.0
    df["suffix_delta_percent"] = df["suffix_delta_rate"] * 100.0
    df["gpu_peak_delta_gb"] = df["gpu_peak_delta_mb"] / 1024.0

    df["mode_label"] = df["mode"].map(MODE_NAME).fillna(df["mode"])
    df["lora_label"] = df["lora_name"].map(LORA_NAME).fillna(df["lora_name"])

    df["lora_order"] = df["lora_name"].apply(
        lambda x: LORA_ORDER.index(x) if x in LORA_ORDER else len(LORA_ORDER)
    )
    df["mode_order"] = df["mode"].apply(
        lambda x: MODE_ORDER.index(x) if x in MODE_ORDER else len(MODE_ORDER)
    )

    df = df.sort_values(["lora_order", "mode_order"]).reset_index(drop=True)
    return df


def compute_benefit(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["mode"] == "baseline_no_group_kv_reuse"].copy()
    reuse = df[df["mode"] == "group_node_kv_reuse_lora_tree"].copy()

    rows = []

    for _, base_row in baseline.iterrows():
        matched = reuse[
            (reuse["group_name"] == base_row["group_name"])
            & (reuse["lora_name"] == base_row["lora_name"])
        ]

        if matched.empty:
            continue

        reuse_row = matched.iloc[0]

        ttft_drop = (base_row["ttft_ms"] - reuse_row["ttft_ms"]) / base_row["ttft_ms"] * 100.0
        total_drop = (base_row["total_ms"] - reuse_row["total_ms"]) / base_row["total_ms"] * 100.0
        tps_change = (reuse_row["tps"] - base_row["tps"]) / base_row["tps"] * 100.0

        rows.append(
            {
                "group_name": base_row["group_name"],
                "lora_name": base_row["lora_name"],
                "lora_label": base_row["lora_label"],
                "prompt_tokens": int(reuse_row["n_prompt_tokens"]),
                "prefix_tokens": int(reuse_row["n_prefix_tokens"]),
                "suffix_tokens": int(reuse_row["n_suffix_tokens"]),
                "prefix_reuse_percent": reuse_row["prefix_reuse_percent"],
                "suffix_delta_percent": reuse_row["suffix_delta_percent"],
                "baseline_ttft_ms": base_row["ttft_ms"],
                "reuse_ttft_ms": reuse_row["ttft_ms"],
                "ttft_drop_percent": ttft_drop,
                "baseline_total_ms": base_row["total_ms"],
                "reuse_total_ms": reuse_row["total_ms"],
                "total_drop_percent": total_drop,
                "baseline_tps": base_row["tps"],
                "reuse_tps": reuse_row["tps"],
                "tps_change_percent": tps_change,
                "baseline_gpu_delta_mb": base_row["gpu_peak_delta_mb"],
                "reuse_gpu_delta_mb": reuse_row["gpu_peak_delta_mb"],
            }
        )

    benefit = pd.DataFrame(rows)
    benefit.to_csv(BENEFIT_CSV, index=False, encoding="utf-8-sig")
    return benefit


def print_tables(df: pd.DataFrame, benefit: pd.DataFrame) -> None:
    show_cols = [
        "group_name",
        "lora_label",
        "mode_label",
        "n_prompt_tokens",
        "n_prefix_tokens",
        "n_suffix_tokens",
        "prefix_reuse_percent",
        "suffix_delta_percent",
        "ttft_ms",
        "total_ms",
        "tps",
        "gpu_peak_delta_gb",
    ]

    display_df = df[show_cols].rename(
        columns={
            "group_name": "组",
            "lora_label": "LoRA",
            "mode_label": "模式",
            "n_prompt_tokens": "Prompt tokens",
            "n_prefix_tokens": "复用前缀 tokens",
            "n_suffix_tokens": "剩余后缀 tokens",
            "prefix_reuse_percent": "前缀复用率(%)",
            "suffix_delta_percent": "差异后缀率(%)",
            "ttft_ms": "TTFT(ms)",
            "total_ms": "总耗时(ms)",
            "tps": "TPS",
            "gpu_peak_delta_gb": "GPU增量(GB)",
        }
    )

    benefit_show = benefit[
        [
            "group_name",
            "lora_label",
            "ttft_drop_percent",
            "total_drop_percent",
            "tps_change_percent",
            "prefix_tokens",
            "suffix_tokens",
            "prefix_reuse_percent",
            "suffix_delta_percent",
        ]
    ].rename(
        columns={
            "group_name": "组",
            "lora_label": "LoRA",
            "ttft_drop_percent": "TTFT下降(%)",
            "total_drop_percent": "总耗时下降(%)",
            "tps_change_percent": "TPS变化(%)",
            "prefix_tokens": "复用前缀tokens",
            "suffix_tokens": "剩余后缀tokens",
            "prefix_reuse_percent": "前缀复用率(%)",
            "suffix_delta_percent": "差异后缀率(%)",
        }
    )

    print("\nGroup Node KV复用 + LoRA树实验结果:")
    try:
        print(display_df.to_markdown(index=False, floatfmt=".2f"))
    except Exception:
        print(display_df.to_string(index=False))

    print("\n对比收益:")
    try:
        print(benefit_show.to_markdown(index=False, floatfmt=".2f"))
    except Exception:
        print(benefit_show.to_string(index=False))


def get_lora_labels(df: pd.DataFrame) -> list[str]:
    labels = []

    for lora_name in LORA_ORDER:
        sub = df[df["lora_name"] == lora_name]
        if not sub.empty:
            labels.append(sub.iloc[0]["lora_label"])

    return labels


def get_mode_values(df: pd.DataFrame, value_col: str) -> tuple[list[str], list[float], list[float]]:
    labels = []
    baseline_values = []
    reuse_values = []

    for lora_name in LORA_ORDER:
        sub = df[df["lora_name"] == lora_name]

        if sub.empty:
            continue

        labels.append(sub.iloc[0]["lora_label"])

        baseline = sub[sub["mode"] == "baseline_no_group_kv_reuse"]
        reuse = sub[sub["mode"] == "group_node_kv_reuse_lora_tree"]

        baseline_values.append(float(baseline.iloc[0][value_col]) if not baseline.empty else 0.0)
        reuse_values.append(float(reuse.iloc[0][value_col]) if not reuse.empty else 0.0)

    return labels, baseline_values, reuse_values


def draw_compare_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    value_fmt: str,
) -> None:
    labels, baseline_values, reuse_values = get_mode_values(df, value_col)

    x = np.arange(len(labels))
    width = 0.36

    bars1 = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="不复用KV",
        color=MODE_COLOR["baseline_no_group_kv_reuse"],
        alpha=0.92,
    )
    bars2 = ax.bar(
        x + width / 2,
        reuse_values,
        width,
        label="Group KV复用+LoRA树",
        color=MODE_COLOR["group_node_kv_reuse_lora_tree"],
        alpha=0.92,
    )

    max_value = max(baseline_values + reuse_values + [1.0])
    ax.set_ylim(0, max_value * 1.18)

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.015,
                value_fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, axis="y", linestyle="-", alpha=0.28)


def draw_benefit_bar(
    ax: plt.Axes,
    benefit: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    color: str,
    value_fmt: str,
) -> None:
    labels = benefit["lora_label"].tolist()
    values = benefit[value_col].astype(float).tolist()

    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        color=color,
        width=0.62,
        alpha=0.94,
    )

    min_value = min(values + [0.0])
    max_value = max(values + [0.0])
    span = max(max_value - min_value, 1.0)

    ax.set_ylim(min(0, min_value) - span * 0.18, max(0, max_value) + span * 0.22)

    for bar, value in zip(bars, values):
        y = value + span * 0.025 if value >= 0 else value - span * 0.08
        va = "bottom" if value >= 0 else "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            value_fmt.format(value),
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, axis="y", linestyle="-", alpha=0.28)


def draw_reuse_stack(ax: plt.Axes, df: pd.DataFrame) -> None:
    reuse_df = df[df["mode"] == "group_node_kv_reuse_lora_tree"].copy()
    reuse_df = reuse_df.sort_values("lora_order")

    labels = reuse_df["lora_label"].tolist()
    prefix_values = reuse_df["prefix_reuse_percent"].astype(float).tolist()
    suffix_values = reuse_df["suffix_delta_percent"].astype(float).tolist()

    x = np.arange(len(labels))

    ax.bar(
        x,
        prefix_values,
        label="复用前缀",
        color="#2E8B73",
        width=0.62,
        alpha=0.94,
    )
    ax.bar(
        x,
        suffix_values,
        bottom=prefix_values,
        label="差异后缀",
        color="#D9895B",
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
                fontsize=8,
                color="white",
            )

    for idx, value in enumerate(suffix_values):
        if value > 5:
            ax.text(
                idx,
                prefix_values[idx] + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    ax.set_title("Prompt Token 复用比例", fontsize=12, pad=8)
    ax.set_ylabel("占比 / %", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, axis="y", linestyle="-", alpha=0.28)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8)


def plot_six_in_one(df: pd.DataFrame, benefit: pd.DataFrame) -> None:
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
        "#3A8D7A",
        "{:.1f}%",
    )

    draw_benefit_bar(
        axes[1, 1],
        benefit,
        "total_drop_percent",
        "总耗时下降比例",
        "下降比例 / %",
        "#4C72B0",
        "{:.1f}%",
    )

    draw_reuse_stack(axes[1, 2], df)

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

    fig.suptitle("Group Node KV复用 + LoRA树实验结果", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(OUTPUT_FIG, bbox_inches="tight")
    print(f"六图合并结果已保存到: {OUTPUT_FIG}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    benefit = compute_benefit(df)

    print_tables(df, benefit)

    plot_six_in_one(df, benefit)

    plt.show()

    print(f"\n收益数据已保存到: {BENEFIT_CSV}")


if __name__ == "__main__":
    main()