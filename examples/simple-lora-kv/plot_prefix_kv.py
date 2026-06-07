from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-kv/output")
CSV_PATH = OUTPUT_DIR / "prefix_kv_reuse_nopage_summary.csv"

FIG_PATH = OUTPUT_DIR / "prefix_kv_reuse_nopage_compare.png"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Source Han Sans SC",
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


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到 CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    df["label"] = df["mode"].map({
        "baseline_no_prefix_cache": "不复用prefix",
        "prefix_cache_reuse": "复用prefix",
    }).fillna(df["mode"])

    df["gpu_peak_delta_gb"] = df["gpu_peak_delta_mb"] / 1024.0
    df["process_delta_gb"] = df["process_delta_mb"] / 1024.0

    return df


def add_labels(ax, bars, fmt="{:.2f}") -> None:
    ymax = max([bar.get_height() for bar in bars])
    if ymax <= 0:
        ymax = 1.0

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.03,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot(df: pd.DataFrame) -> None:
    setup_font()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=180)
    axes = axes.ravel()

    items = [
        ("throughput_tps", "吞吐量 TPS", "tokens/s", "{:.2f}", "#4C72B0"),
        ("avg_prefix_ms", "平均prefix成本", "ms", "{:.2f}", "#DD8452"),
        ("avg_ttft_ms", "平均TTFT", "ms", "{:.2f}", "#55A868"),
        ("avg_total_ms", "平均总耗时", "ms", "{:.2f}", "#C44E52"),
        ("gpu_peak_delta_gb", "GPU峰值显存增量", "GB", "{:.2f}", "#8172B2"),
        ("process_delta_gb", "进程内存增量", "GB", "{:.2f}", "#64B5CD"),
    ]

    for ax, (col, title, ylabel, fmt, color) in zip(axes, items):
        bars = ax.bar(
            df["label"],
            df[col],
            color=color,
            alpha=0.92,
            width=0.58,
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

        ymax = df[col].max()
        ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)

        add_labels(ax, bars, fmt)

    fig.suptitle("连续KV下公共prefix KV复用效果", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"图像已保存到: {FIG_PATH}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n实验结果:")
    show = pd.DataFrame({
        "配置": df["label"],
        "TPS": df["throughput_tps"].map(lambda x: f"{x:.2f}"),
        "平均prefix(ms)": df["avg_prefix_ms"].map(lambda x: f"{x:.2f}"),
        "平均TTFT(ms)": df["avg_ttft_ms"].map(lambda x: f"{x:.2f}"),
        "平均总耗时(ms)": df["avg_total_ms"].map(lambda x: f"{x:.2f}"),
        "GPU峰值增量(GB)": df["gpu_peak_delta_gb"].map(lambda x: f"{x:.2f}"),
        "进程内存增量(GB)": df["process_delta_gb"].map(lambda x: f"{x:.2f}"),
    })

    print(show.to_markdown(index=False))

    base = df[df["mode"] == "baseline_no_prefix_cache"].iloc[0]
    reuse = df[df["mode"] == "prefix_cache_reuse"].iloc[0]

    tps_gain = (reuse["throughput_tps"] - base["throughput_tps"]) / base["throughput_tps"] * 100.0
    prefix_save = (base["avg_prefix_ms"] - reuse["avg_prefix_ms"]) / base["avg_prefix_ms"] * 100.0
    total_save = (base["avg_total_ms"] - reuse["avg_total_ms"]) / base["avg_total_ms"] * 100.0
    gpu_save = (base["gpu_peak_delta_mb"] - reuse["gpu_peak_delta_mb"]) / base["gpu_peak_delta_mb"] * 100.0

    print("\n对比结论:")
    print(f"TPS变化: {tps_gain:+.2f}%")
    print(f"prefix成本降低: {prefix_save:+.2f}%")
    print(f"平均总耗时降低: {total_save:+.2f}%")
    print(f"GPU峰值显存降低: {gpu_save:+.2f}%")

    print("\n说明:")
    print("当前是不分页的连续KV实验。连续KV会在context创建时预留完整KV buffer，")
    print("所以显存降低可能不明显；这个实验主要看公共prefix复用对时延的影响。")


def main() -> None:
    df = load_data()
    print_summary(df)
    plot(df)


if __name__ == "__main__":
    main()