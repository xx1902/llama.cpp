# plot_real_gpu_memory.py
#
# 画真实 LoRA/KV 生命周期下 GPU 显存占用变化。
#
# 输入：
# output/real_lora_kv_gpu_memory.csv
#
# 输出：
# output/real_lora_kv_gpu_memory.png

import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = r"D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CSV_PATH = os.path.join(OUTPUT_DIR, "real_lora_kv_gpu_memory.csv")
FIG_PATH = os.path.join(OUTPUT_DIR, "real_lora_kv_gpu_memory.png")


def setup_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "Source Han Sans SC",
    ]

    plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False


def main():
    setup_font()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 文件: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_cols = ["step", "gpu_delta_mb"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少列: {col}")

    df["gpu_delta_gb"] = df["gpu_delta_mb"] / 1024.0

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)

    ax.plot(
        df["step"],
        df["gpu_delta_gb"],
        color="#4C72B0",
        linewidth=2.6,
        marker="o",
        markersize=6,
        markevery=max(1, len(df) // 12),
        label="GPU显存增量",
    )

    ax.set_title("真实 LoRA/KV 生命周期下的GPU显存变化", fontsize=18, pad=14)
    ax.set_xlabel("生命周期采样步数", fontsize=14)
    ax.set_ylabel("GPU显存增量（GB）", fontsize=14)

    ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.35)
    ax.legend(fontsize=12, loc="upper left", frameon=True)
    ax.tick_params(axis="both", labelsize=12)

    # 标注关键事件，避免图太乱，只标几个核心点。
    important_events = [
        "model_loaded",
        "context_created",
        "lora_loaded",
        "prompt_eval",
        "before_release",
        "released",
        "model_released",
    ]

    marked = set()

    for _, row in df.iterrows():
        event = str(row["event"])

        if event not in important_events:
            continue

        # 同类事件只标第一次，释放事件保留。
        if event in marked and event not in ["before_release", "released", "model_released"]:
            continue

        marked.add(event)

        ax.annotate(
            event,
            xy=(row["step"], row["gpu_delta_gb"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                linewidth=0.8,
                alpha=0.7,
            ),
        )

    fig.tight_layout()
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"GPU 显存图已保存到: {FIG_PATH}")


if __name__ == "__main__":
    main()