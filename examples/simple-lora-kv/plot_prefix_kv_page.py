from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-kv/output")
CSV_PATH = OUTPUT_DIR / "prefix_kv_reuse_summary.csv"

FIG_LATENCY = OUTPUT_DIR / "prefix_kv_reuse_latency.png"
FIG_MEMORY = OUTPUT_DIR / "prefix_kv_reuse_memory.png"
FIG_SPEEDUP = OUTPUT_DIR / "prefix_kv_reuse_speedup.png"

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

    required = {
        "mode",
        "use_prefix_reuse",
        "use_paged_kv",
        "elapsed_ms",
        "throughput_tps",
        "avg_prefix_ms",
        "avg_ttft_ms",
        "avg_total_ms",
        "gpu_peak_delta_mb",
        "process_delta_mb",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    df["kv_label"] = df["use_paged_kv"].map({
        0: "连续KV",
        1: "分页KV",
    })

    df["reuse_label"] = df["use_prefix_reuse"].map({
        0: "不复用prefix",
        1: "复用prefix",
    })

    df["label"] = df["kv_label"] + "\n" + df["reuse_label"]
    df["elapsed_s"] = df["elapsed_ms"] / 1000.0
    df["gpu_peak_delta_gb"] = df["gpu_peak_delta_mb"] / 1024.0
    df["process_delta_gb"] = df["process_delta_mb"] / 1024.0

    return df


def plot_latency(df: pd.DataFrame) -> None:
    setup_font()

    x = np.arange(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=180)

    bars1 = ax.bar(
        x - width,
        df["avg_prefix_ms"],
        width=width,
        label="平均prefix成本",
        color="#4C72B0",
        alpha=0.92,
    )

    bars2 = ax.bar(
        x,
        df["avg_ttft_ms"],
        width=width,
        label="平均TTFT",
        color="#DD8452",
        alpha=0.92,
    )

    bars3 = ax.bar(
        x + width,
        df["avg_total_ms"],
        width=width,
        label="平均总耗时",
        color="#55A868",
        alpha=0.92,
    )

    ax.set_title("Prefix KV复用对请求时延的影响", fontsize=16)
    ax.set_xlabel("实验配置")
    ax.set_ylabel("时延 / ms")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    ymax = max(df["avg_prefix_ms"].max(), df["avg_ttft_ms"].max(), df["avg_total_ms"].max())
    ax.set_ylim(0, ymax * 1.22)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(FIG_LATENCY, bbox_inches="tight")
    plt.close(fig)

    print(f"时延图已保存: {FIG_LATENCY}")


def plot_memory(df: pd.DataFrame) -> None:
    setup_font()

    x = np.arange(len(df))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=180)

    bars1 = ax.bar(
        x - width / 2,
        df["gpu_peak_delta_gb"],
        width=width,
        label="GPU峰值显存增量",
        color="#4C72B0",
        alpha=0.92,
    )

    bars2 = ax.bar(
        x + width / 2,
        df["process_delta_gb"],
        width=width,
        label="进程内存增量",
        color="#C44E52",
        alpha=0.92,
    )

    ax.set_title("Prefix KV复用对内存开销的影响", fontsize=16)
    ax.set_xlabel("实验配置")
    ax.set_ylabel("内存增量 / GB")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    ymax = max(df["gpu_peak_delta_gb"].max(), df["process_delta_gb"].max())
    ax.set_ylim(0, ymax * 1.22)

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(FIG_MEMORY, bbox_inches="tight")
    plt.close(fig)

    print(f"内存图已保存: {FIG_MEMORY}")


def plot_speedup(df: pd.DataFrame) -> None:
    setup_font()

    base_cont = df[(df["use_paged_kv"] == 0) & (df["use_prefix_reuse"] == 0)].iloc[0]
    base_page = df[(df["use_paged_kv"] == 1) & (df["use_prefix_reuse"] == 0)].iloc[0]

    rows = []

    for _, row in df.iterrows():
        if row["use_paged_kv"] == 0:
            base = base_cont
        else:
            base = base_page

        rows.append({
            "label": row["label"],
            "speedup": row["throughput_tps"] / base["throughput_tps"],
            "latency_save": (base["avg_total_ms"] - row["avg_total_ms"]) / base["avg_total_ms"] * 100.0,
            "gpu_save": (base["gpu_peak_delta_mb"] - row["gpu_peak_delta_mb"]) / base["gpu_peak_delta_mb"] * 100.0,
        })

    out = pd.DataFrame(rows)

    x = np.arange(len(out))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=180)

    bars1 = ax.bar(
        x - width,
        out["speedup"],
        width=width,
        label="TPS相对提升",
        color="#8172B2",
        alpha=0.92,
    )

    bars2 = ax.bar(
        x,
        out["latency_save"],
        width=width,
        label="总时延降低比例",
        color="#55A868",
        alpha=0.92,
    )

    bars3 = ax.bar(
        x + width,
        out["gpu_save"],
        width=width,
        label="GPU峰值显存降低比例",
        color="#DD8452",
        alpha=0.92,
    )

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(1, color="#777777", linewidth=1.0, linestyle=":", alpha=0.5)

    ax.set_title("Prefix KV复用收益对比", fontsize=16)
    ax.set_xlabel("实验配置")
    ax.set_ylabel("比例")
    ax.set_xticks(x)
    ax.set_xticklabels(out["label"], fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_SPEEDUP, bbox_inches="tight")
    plt.close(fig)

    print(f"收益图已保存: {FIG_SPEEDUP}")


def print_summary(df: pd.DataFrame) -> None:
    show = pd.DataFrame({
        "配置": df["label"].str.replace("\n", " / "),
        "TPS": df["throughput_tps"].map(lambda x: f"{x:.2f}"),
        "平均prefix(ms)": df["avg_prefix_ms"].map(lambda x: f"{x:.2f}"),
        "平均TTFT(ms)": df["avg_ttft_ms"].map(lambda x: f"{x:.2f}"),
        "平均总耗时(ms)": df["avg_total_ms"].map(lambda x: f"{x:.2f}"),
        "GPU峰值增量(GB)": df["gpu_peak_delta_gb"].map(lambda x: f"{x:.2f}"),
        "进程内存增量(GB)": df["process_delta_gb"].map(lambda x: f"{x:.2f}"),
    })

    print("\nPrefix KV复用实验结果:")
    print(show.to_markdown(index=False))

    for use_paged, name in [(0, "连续KV"), (1, "分页KV")]:
        base = df[(df["use_paged_kv"] == use_paged) & (df["use_prefix_reuse"] == 0)].iloc[0]
        reuse = df[(df["use_paged_kv"] == use_paged) & (df["use_prefix_reuse"] == 1)].iloc[0]

        tps_gain = (reuse["throughput_tps"] - base["throughput_tps"]) / base["throughput_tps"] * 100.0
        latency_save = (base["avg_total_ms"] - reuse["avg_total_ms"]) / base["avg_total_ms"] * 100.0
        gpu_save = (base["gpu_peak_delta_mb"] - reuse["gpu_peak_delta_mb"]) / base["gpu_peak_delta_mb"] * 100.0

        print(f"\n{name} 下 prefix KV复用对比:")
        print(f"TPS变化: {tps_gain:+.2f}%")
        print(f"平均总时延变化: {-latency_save:+.2f}%")
        print(f"GPU峰值显存降低: {gpu_save:+.2f}%")


def main() -> None:
    df = load_data()

    print_summary(df)

    plot_latency(df)
    plot_memory(df)
    plot_speedup(df)


if __name__ == "__main__":
    main()