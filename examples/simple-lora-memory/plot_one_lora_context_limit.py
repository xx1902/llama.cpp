from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


BASE_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory")
OUTPUT_DIR = BASE_DIR / "output"

CONTINUOUS_CSV = OUTPUT_DIR / "one_lora_context_limit_continuous.csv"
PAGED_CSV = OUTPUT_DIR / "one_lora_context_limit_paged.csv"
SUMMARY_CSV = OUTPUT_DIR / "one_lora_context_limit_summary.csv"

FIG_REAL_GPU = OUTPUT_DIR / "one_lora_context_limit_real_gpu.png"
FIG_KV_GROWTH = OUTPUT_DIR / "one_lora_context_limit_kv_growth.png"
FIG_PAGE_RATE = OUTPUT_DIR / "one_lora_context_limit_page_rate.png"
FIG_SUMMARY = OUTPUT_DIR / "one_lora_context_limit_summary.png"

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


def load_trace(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {path}")

    df = pd.read_csv(path)

    required = {
        "mode",
        "step",
        "event",
        "position",
        "generated_tokens",
        "n_ctx",
        "kv_page_size",
        "theoretical_used_pages",
        "context_used_rate",
        "page_used_rate",
        "gpu_used_mb",
        "gpu_delta_mb",
        "gpu_ready_delta_mb",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少字段: {missing}")

    df["label"] = label
    df["gpu_delta_gb"] = df["gpu_delta_mb"] / 1024.0
    df["gpu_ready_delta_gb"] = df["gpu_ready_delta_mb"] / 1024.0
    df["context_used_percent"] = df["context_used_rate"] * 100.0
    df["page_used_percent"] = df["page_used_rate"] * 100.0

    return df


def plot_real_gpu(continuous: pd.DataFrame, paged: pd.DataFrame) -> None:
    setup_font()

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)

    ax.plot(
        continuous["position"],
        continuous["gpu_delta_gb"],
        color="#4C72B0",
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        markevery=max(1, len(continuous) // 14),
        label="连续KV：真实GPU显存增量",
    )

    ax.plot(
        paged["position"],
        paged["gpu_delta_gb"],
        color="#DD8452",
        linewidth=2.4,
        marker="s",
        markersize=4.5,
        markevery=max(1, len(paged) // 14),
        label="分页KV：真实GPU显存增量",
    )

    ax.set_title("单LoRA长上下文下真实GPU显存占用对比", fontsize=17, pad=12)
    ax.set_xlabel("已写入上下文位置 / token")
    ax.set_ylabel("GPU显存增量 / GB")
    ax.grid(True, linestyle="-", alpha=0.32)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_REAL_GPU, bbox_inches="tight")
    plt.close(fig)

    print(f"真实GPU显存图已保存: {FIG_REAL_GPU}")


def plot_kv_growth(continuous: pd.DataFrame, paged: pd.DataFrame) -> None:
    setup_font()

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)

    ax.plot(
        continuous["position"],
        continuous["gpu_ready_delta_gb"],
        color="#4C72B0",
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        markevery=max(1, len(continuous) // 14),
        label="连续KV：ready后新增显存",
    )

    ax.plot(
        paged["position"],
        paged["gpu_ready_delta_gb"],
        color="#DD8452",
        linewidth=2.4,
        marker="s",
        markersize=4.5,
        markevery=max(1, len(paged) // 14),
        label="分页KV：ready后新增显存",
    )

    ax.axhline(0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_title("去除模型和LoRA加载后的KV增长趋势", fontsize=17, pad=12)
    ax.set_xlabel("已写入上下文位置 / token")
    ax.set_ylabel("ready后GPU显存增量 / GB")
    ax.grid(True, linestyle="-", alpha=0.32)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_KV_GROWTH, bbox_inches="tight")
    plt.close(fig)

    print(f"KV增长趋势图已保存: {FIG_KV_GROWTH}")


def plot_page_rate(paged: pd.DataFrame) -> None:
    setup_font()

    fig, ax1 = plt.subplots(figsize=(11, 5.8), dpi=180)

    ax1.plot(
        paged["position"],
        paged["page_used_percent"],
        color="#55A868",
        linewidth=2.4,
        marker="s",
        markersize=4.5,
        markevery=max(1, len(paged) // 14),
        label="分页KV页使用率",
    )

    ax1.set_title("分页KV页使用率随上下文增长的变化", fontsize=17, pad=12)
    ax1.set_xlabel("已写入上下文位置 / token")
    ax1.set_ylabel("页使用率 / %")
    ax1.grid(True, linestyle="-", alpha=0.32)

    ax2 = ax1.twinx()
    ax2.step(
        paged["position"],
        paged["theoretical_used_pages"],
        where="post",
        color="#C44E52",
        linewidth=2.0,
        alpha=0.85,
        label="已使用页数",
    )
    ax2.set_ylabel("已使用页数")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_PAGE_RATE, bbox_inches="tight")
    plt.close(fig)

    print(f"分页页使用率图已保存: {FIG_PAGE_RATE}")


def plot_summary() -> None:
    if not SUMMARY_CSV.exists():
        print(f"warning: 找不到 summary 文件: {SUMMARY_CSV}")
        return

    setup_font()

    summary = pd.read_csv(SUMMARY_CSV)

    if summary.empty:
        print("warning: summary 为空")
        return

    label_map = {
        "continuous_kv": "连续KV",
        "paged_kv": "分页KV",
    }

    summary["label"] = summary["mode"].map(label_map).fillna(summary["mode"])
    summary["gpu_peak_delta_gb"] = summary["gpu_peak_delta_mb"] / 1024.0
    summary["gpu_peak_ready_delta_gb"] = summary["gpu_peak_ready_delta_mb"] / 1024.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=180)

    colors = ["#4C72B0", "#DD8452"]

    bars0 = axes[0].bar(
        summary["label"],
        summary["gpu_peak_delta_gb"],
        color=colors,
        alpha=0.92,
        width=0.58,
    )
    axes[0].set_title("GPU峰值显存增量")
    axes[0].set_ylabel("GB")
    axes[0].grid(True, axis="y", alpha=0.3)

    bars1 = axes[1].bar(
        summary["label"],
        summary["gpu_peak_ready_delta_gb"],
        color=colors,
        alpha=0.92,
        width=0.58,
    )
    axes[1].set_title("ready后峰值新增显存")
    axes[1].set_ylabel("GB")
    axes[1].grid(True, axis="y", alpha=0.3)

    bars2 = axes[2].bar(
        summary["label"],
        summary["tps"],
        color=colors,
        alpha=0.92,
        width=0.58,
    )
    axes[2].set_title("推理速度")
    axes[2].set_ylabel("tokens/s")
    axes[2].grid(True, axis="y", alpha=0.3)

    for ax, bars in zip(axes, [bars0, bars1, bars2]):
        y_max = max([bar.get_height() for bar in bars])
        ax.set_ylim(0, y_max * 1.25 if y_max > 0 else 1.0)

        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + y_max * 0.04,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("单LoRA长上下文实验汇总", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_SUMMARY, bbox_inches="tight")
    plt.close(fig)

    print(f"汇总图已保存: {FIG_SUMMARY}")

    print("\n实验汇总:")
    print(summary[[
        "mode",
        "n_ctx",
        "kv_page_size",
        "n_requests",
        "final_position",
        "generated_tokens",
        "theoretical_used_pages",
        "elapsed_ms",
        "tps",
        "gpu_peak_delta_mb",
        "gpu_peak_ready_delta_mb",
        "process_delta_mb",
    ]].to_string(index=False))


def print_conclusion(continuous: pd.DataFrame, paged: pd.DataFrame) -> None:
    c_peak = continuous["gpu_delta_mb"].max()
    p_peak = paged["gpu_delta_mb"].max()

    c_ready_peak = continuous["gpu_ready_delta_mb"].max()
    p_ready_peak = paged["gpu_ready_delta_mb"].max()

    final_pos = int(max(continuous["position"].max(), paged["position"].max()))
    n_ctx = int(max(continuous["n_ctx"].max(), paged["n_ctx"].max()))

    print("\n关键对比:")
    print(f"上下文最终写入位置: {final_pos}/{n_ctx}")
    print(f"连续KV GPU峰值显存增量: {c_peak:.2f} MB")
    print(f"分页KV GPU峰值显存增量: {p_peak:.2f} MB")

    if c_peak > 0:
        print(f"分页KV相对连续KV峰值降低: {(c_peak - p_peak) / c_peak * 100.0:.2f}%")

    print(f"连续KV ready后峰值新增显存: {c_ready_peak:.2f} MB")
    print(f"分页KV ready后峰值新增显存: {p_ready_peak:.2f} MB")

    print("\n解释:")
    print("如果最终位置接近 n_ctx，上下文基本被填满，分页KV需要把大部分页都申请出来。")
    print("因此分页KV的显存优势会比短上下文、多context碎片场景更小。")
    print("这说明分页KV主要节省的是未使用上下文容量，而不是压缩已经实际写入的KV内容。")


def main() -> None:
    continuous = load_trace(CONTINUOUS_CSV, "连续KV")
    paged = load_trace(PAGED_CSV, "分页KV")

    plot_real_gpu(continuous, paged)
    plot_kv_growth(continuous, paged)
    plot_page_rate(paged)
    plot_summary()
    print_conclusion(continuous, paged)


if __name__ == "__main__":
    main()