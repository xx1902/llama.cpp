import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = r"D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CONTINUOUS_CSV = os.path.join(OUTPUT_DIR, "real_lora_kv_continuous_memory.csv")
PAGED_CSV = os.path.join(OUTPUT_DIR, "real_lora_kv_page_memory.csv")

FIG_PATH = os.path.join(OUTPUT_DIR, "real_lora_kv_gpu_compare.png")


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


def load_data(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    df = pd.read_csv(path)

    if "gpu_delta_mb" not in df.columns:
        raise ValueError(f"{path} 缺少 gpu_delta_mb 列")

    df["label"] = label
    df["gpu_delta_gb"] = df["gpu_delta_mb"] / 1024.0

    return df


def main():
    setup_font()

    continuous = load_data(CONTINUOUS_CSV, "连续KV")
    paged = load_data(PAGED_CSV, "分页KV")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)

    ax.plot(
        continuous["step"],
        continuous["gpu_delta_gb"],
        color="#4C72B0",
        linewidth=2.4,
        marker="o",
        markersize=5,
        markevery=max(1, len(continuous) // 12),
        label="连续KV GPU显存增量",
    )

    ax.plot(
        paged["step"],
        paged["gpu_delta_gb"],
        color="#DD8452",
        linewidth=2.4,
        marker="s",
        markersize=5,
        markevery=max(1, len(paged) // 12),
        label="分页KV GPU显存增量",
    )

    ax.set_title("连续KV与分页KV真实生命周期下的GPU显存变化", fontsize=18, pad=14)
    ax.set_xlabel("生命周期采样步数", fontsize=14)
    ax.set_ylabel("GPU显存增量（GB）", fontsize=14)

    ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.35)
    ax.legend(fontsize=12, loc="upper left", frameon=True)
    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"图像已保存到: {FIG_PATH}")

    c_peak = continuous["gpu_delta_mb"].max()
    p_peak = paged["gpu_delta_mb"].max()

    print("\nGPU峰值显存增量:")
    print(f"连续KV: {c_peak:.2f} MB")
    print(f"分页KV: {p_peak:.2f} MB")

    if c_peak > 0:
        print(f"分页KV相对降低: {(c_peak - p_peak) / c_peak * 100.0:.2f}%")


if __name__ == "__main__":
    main()