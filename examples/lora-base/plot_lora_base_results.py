from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


BASE_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base")
OUTPUT_DIR = BASE_DIR / "output"

CSV_PATH = OUTPUT_DIR / "multi_level_kv_tree_lora_tree_summary.csv"
FIG_PATH = OUTPUT_DIR / "multi_level_kv_tree_lora_tree_compare.png"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Source Han Sans SC",
]

LORA_NAME_MAP = {
    "subliminal_monkey": "LoRA-monkey",
    "delta_monkey": "LoRA-monkey",
    "monkey": "LoRA-monkey",

    "subliminal_tiger": "LoRA-tiger",
    "delta_tiger": "LoRA-tiger",
    "tiger": "LoRA-tiger",

    "subliminal_wolf": "LoRA-wolf",
    "delta_wolf": "LoRA-wolf",
    "wolf": "LoRA-wolf",

    "marketing_adapter": "LoRA-marketing",
    "anchor_marketing": "LoRA-marketing",
    "marketing": "LoRA-marketing",
}

LORA_ORDER = [
    "LoRA-monkey",
    "LoRA-tiger",
    "LoRA-wolf",
    "LoRA-marketing",
]

MODE_NAME_MAP = {
    "baseline_no_kv_tree": "不复用KV树",
    "multi_level_kv_tree_lora_tree": "多层KV树+LoRA树",
    "baseline_no_kv_prefix_reuse": "不复用KV前缀",
    "fused_kv_prefix_lora_tree": "KV前缀复用+LoRA树",
    "auto_kv_prefix_tree_lora_tree": "自动KV前缀树+LoRA树",
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
        print("warning: 未找到常见中文字体，中文可能显示异常。")

    plt.rcParams["axes.unicode_minus"] = False


def normalize_lora_name(name: str) -> str:
    if name in LORA_NAME_MAP:
        return LORA_NAME_MAP[name]

    lower = str(name).lower()

    if "monkey" in lower:
        return "LoRA-monkey"

    if "tiger" in lower:
        return "LoRA-tiger"

    if "wolf" in lower:
        return "LoRA-wolf"

    if "marketing" in lower:
        return "LoRA-marketing"

    return str(name)


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到文件: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    df["lora_label"] = df["lora_name"].apply(normalize_lora_name)
    df["mode_label"] = df["mode"].map(MODE_NAME_MAP).fillna(df["mode"])
    df["gpu_peak_delta_gb"] = df["gpu_peak_delta_mb"] / 1024.0

    return df


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = df.pivot_table(
        index="lora_label",
        columns="mode_label",
        values=metric,
        aggfunc="mean",
    )

    order = [x for x in LORA_ORDER if x in table.index]
    extra = [x for x in table.index if x not in order]

    table = table.reindex(order + extra)

    return table


def add_labels(ax, bars, values, fmt="{:.1f}") -> None:
    valid_values = [float(v) for v in values if pd.notna(v)]
    ymax = max(valid_values) if valid_values else 1.0

    if ymax <= 0:
        ymax = 1.0

    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.025,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_grouped_metric(
        ax,
        df: pd.DataFrame,
        metric: str,
        title: str,
        ylabel: str,
        fmt: str) -> None:
    table = pivot_metric(df, metric)

    modes = list(table.columns)
    x = np.arange(len(table.index))
    width = 0.34 if len(modes) <= 2 else 0.24

    colors = {
        "不复用KV树": "#4C72B0",
        "多层KV树+LoRA树": "#DD8452",
        "不复用KV前缀": "#4C72B0",
        "KV前缀复用+LoRA树": "#DD8452",
        "自动KV前缀树+LoRA树": "#55A868",
    }

    for i, mode in enumerate(modes):
        values = table[mode].fillna(0.0).to_numpy()
        offset = (i - (len(modes) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=mode,
            color=colors.get(mode, "#999999"),
            alpha=0.92,
        )

        add_labels(ax, bars, values, fmt)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ymax = df[metric].max()

    if ymax > 0:
        ax.set_ylim(0, ymax * 1.25)
    else:
        ax.set_ylim(0, 1.0)


def print_summary(df: pd.DataFrame) -> None:
    show = pd.DataFrame({
        "LoRA": df["lora_label"],
        "模式": df["mode_label"],
        "前缀token数": df["n_prefix_tokens"],
        "后缀token数": df["n_suffix_tokens"],
        "prefix成本(ms)": df["prefix_ms"].map(lambda x: f"{x:.3f}"),
        "LoRA绑定(ms)": df["lora_bind_ms"].map(lambda x: f"{x:.3f}"),
        "TTFT(ms)": df["ttft_ms"].map(lambda x: f"{x:.2f}"),
        "总耗时(ms)": df["total_ms"].map(lambda x: f"{x:.2f}"),
        "TPS": df["tps"].map(lambda x: f"{x:.2f}"),
        "GPU峰值增量(GB)": df["gpu_peak_delta_gb"].map(lambda x: f"{x:.4f}"),
    })

    print("\n多层KV树 + LoRA树实验结果:")
    print(show.to_markdown(index=False))

    appeared_loras = set(df["lora_label"])
    missing_loras = [x for x in LORA_ORDER if x not in appeared_loras]

    if missing_loras:
        print("\n提示:")
        print(f"这些 LoRA 没有出现在图里，是因为 CSV 中没有对应请求结果: {', '.join(missing_loras)}")

    abnormal = df[(df["ttft_ms"] > 10000) | (df["total_ms"] > 10000)]

    if not abnormal.empty:
        print("\n检测到异常耗时点，建议复跑确认:")
        print(abnormal[["mode", "lora_name", "ttft_ms", "total_ms"]].to_string(index=False))


def main() -> None:
    setup_font()

    df = load_data()
    print_summary(df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), dpi=180)
    axes = axes.ravel()

    plot_grouped_metric(
        axes[0],
        df,
        "prefix_ms",
        "Prefix复制/复用成本",
        "ms",
        "{:.3f}",
    )

    plot_grouped_metric(
        axes[1],
        df,
        "lora_bind_ms",
        "LoRA绑定耗时",
        "ms",
        "{:.3f}",
    )

    plot_grouped_metric(
        axes[2],
        df,
        "ttft_ms",
        "首Token延迟",
        "ms",
        "{:.1f}",
    )

    plot_grouped_metric(
        axes[3],
        df,
        "total_ms",
        "总耗时",
        "ms",
        "{:.1f}",
    )

    plot_grouped_metric(
        axes[4],
        df,
        "tps",
        "生成速度",
        "tokens/s",
        "{:.2f}",
    )

    plot_grouped_metric(
        axes[5],
        df,
        "gpu_peak_delta_gb",
        "GPU峰值显存增量",
        "GB",
        "{:.4f}",
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(
        "多层KV前缀树与LoRA树绑定实验",
        fontsize=17,
        y=0.985,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"\n图像已保存到: {FIG_PATH}")


if __name__ == "__main__":
    main()