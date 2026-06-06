from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Source Han Sans SC",
]

BASE_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory")
OUTPUT_DIR = BASE_DIR / "output"

SUMMARY_CSV = OUTPUT_DIR / "kv_cache_speed_summary.csv"
SAMPLES_CSV = OUTPUT_DIR / "kv_cache_speed_samples.csv"

FIG_TPS = OUTPUT_DIR / "kv_cache_speed_tps.png"
FIG_LATENCY = OUTPUT_DIR / "kv_cache_speed_latency.png"
FIG_BREAKDOWN = OUTPUT_DIR / "kv_cache_speed_breakdown.png"


MODE_NAME = {
    "continuous_kv": "连续KV",
    "paged_kv": "分页KV",
}

COLOR = {
    "continuous_kv": "#4C72B0",
    "paged_kv": "#DD8452",
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {SUMMARY_CSV}")

    if not SAMPLES_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {SAMPLES_CSV}")

    summary = pd.read_csv(SUMMARY_CSV)
    samples = pd.read_csv(SAMPLES_CSV)

    summary["mode_zh"] = summary["mode"].map(MODE_NAME)
    samples["mode_zh"] = samples["mode"].map(MODE_NAME)

    return summary, samples


def plot_tps(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=180)

    bars = ax.bar(
        summary["mode_zh"],
        summary["avg_tps"],
        color=[COLOR[m] for m in summary["mode"]],
        width=0.55,
        alpha=0.95,
    )

    for bar, value in zip(bars, summary["avg_tps"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(summary["avg_tps"]) * 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("连续KV与分页KV的推理吞吐量对比", fontsize=15, pad=12)
    ax.set_xlabel("KV cache 方式", fontsize=12)
    ax.set_ylabel("Decode 吞吐量（tokens/s）", fontsize=12)
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIG_TPS, bbox_inches="tight")
    plt.close(fig)

    print(f"TPS 图已保存: {FIG_TPS}")


def plot_latency(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)

    x = range(len(summary))
    width = 0.22

    metrics = [
        ("avg_prefill_ms", "Prefill"),
        ("avg_ttft_ms", "TTFT"),
        ("avg_decode_ms", "Decode"),
    ]

    offsets = [-width, 0, width]

    for (metric, label), offset in zip(metrics, offsets):
        values = summary[metric].tolist()
        ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            label=label,
            alpha=0.9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["mode_zh"])
    ax.set_title("连续KV与分页KV的主要延迟指标对比", fontsize=15, pad=12)
    ax.set_xlabel("KV cache 方式", fontsize=12)
    ax.set_ylabel("耗时（ms）", fontsize=12)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_LATENCY, bbox_inches="tight")
    plt.close(fig)

    print(f"延迟图已保存: {FIG_LATENCY}")


def plot_breakdown(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)

    labels = summary["mode_zh"].tolist()

    ctx = summary["avg_context_create_ms"].tolist()
    bind = summary["avg_lora_bind_ms"].tolist()
    prefill = summary["avg_prefill_ms"].tolist()
    decode = summary["avg_decode_ms"].tolist()

    bottom = [0.0 for _ in labels]

    for values, name, color in [
        (ctx, "Context创建", "#4C72B0"),
        (bind, "LoRA绑定", "#55A868"),
        (prefill, "Prefill", "#DD8452"),
        (decode, "Decode", "#C44E52"),
    ]:
        ax.bar(
            labels,
            values,
            bottom=bottom,
            label=name,
            color=color,
            alpha=0.9,
            width=0.55,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_title("连续KV与分页KV的推理耗时组成", fontsize=15, pad=12)
    ax.set_xlabel("KV cache 方式", fontsize=12)
    ax.set_ylabel("平均耗时（ms）", fontsize=12)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_BREAKDOWN, bbox_inches="tight")
    plt.close(fig)

    print(f"耗时组成图已保存: {FIG_BREAKDOWN}")


def print_analysis(summary: pd.DataFrame) -> None:
    print("\nKV cache 速度汇总:")
    print(summary.to_string(index=False))

    cont = summary[summary["mode"] == "continuous_kv"].iloc[0]
    paged = summary[summary["mode"] == "paged_kv"].iloc[0]

    tps_change = (paged["avg_tps"] - cont["avg_tps"]) / cont["avg_tps"] * 100.0
    total_change = (paged["avg_total_ms"] - cont["avg_total_ms"]) / cont["avg_total_ms"] * 100.0
    decode_change = (paged["avg_decode_ms"] - cont["avg_decode_ms"]) / cont["avg_decode_ms"] * 100.0

    print("\n对比结论:")
    print(f"分页KV TPS 变化: {tps_change:+.2f}%")
    print(f"分页KV 总耗时变化: {total_change:+.2f}%")
    print(f"分页KV Decode 耗时变化: {decode_change:+.2f}%")


def main() -> None:
    setup_font()

    summary, samples = load_data()

    print_analysis(summary)
    plot_tps(summary)
    plot_latency(summary)
    plot_breakdown(summary)


if __name__ == "__main__":
    main()