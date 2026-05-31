import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT_DIR = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output"
CSV_PATH = os.path.join(OUTPUT_DIR, "real_memory_trace.csv")


def setup_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "Source Han Sans SC",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}

    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name

    plt.rcParams["axes.unicode_minus"] = False
    return None


def plot_memory_usage(df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["step"],
        df["gpu_delta_mb"] / 1024.0,
        label="GPU显存增量",
        color="#4C72B0",
        linewidth=1.8,
    )

    plt.plot(
        df["step"],
        df["process_private_mb"] / 1024.0,
        label="进程内存",
        color="#DD8452",
        linewidth=1.8,
    )

    plt.title("真实 LoRA/KV 生命周期下的内存变化")
    plt.xlabel("分配步数")
    plt.ylabel("内存占用（GB）")
    plt.grid(True, linestyle="-", alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "real_memory_usage_trace.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def plot_live_objects(df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["step"],
        df["live_contexts"],
        label="活跃 context 数",
        color="#55A868",
        linewidth=1.8,
    )

    plt.plot(
        df["step"],
        df["live_loras"],
        label="活跃 LoRA 数",
        color="#C44E52",
        linewidth=1.8,
    )

    plt.title("真实实验中的活跃对象数量")
    plt.xlabel("分配步数")
    plt.ylabel("对象数量")
    plt.grid(True, linestyle="-", alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "real_live_objects_trace.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def main():
    setup_chinese_font()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到结果文件: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    plot_memory_usage(df)
    plot_live_objects(df)

    print("\n真实内存采样预览:")
    print(df.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()