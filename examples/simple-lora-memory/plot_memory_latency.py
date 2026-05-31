import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT_DIR = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output"
CSV_PATH = os.path.join(OUTPUT_DIR, "lora_load_path_latency.csv")
OUT_PATH = os.path.join(OUTPUT_DIR, "lora_load_path_latency_box.png")

PATH_ORDER = [
    "gpu_hot",
    "cpu_warm",
    "file_load",
]

PATH_NAME = {
    "gpu_hot": "GPU 热区",
    "cpu_warm": "CPU 温区",
    "file_load": "SSD 冷区",
}

PATH_COLOR = {
    "gpu_hot": "#4C72B0",
    "cpu_warm": "#DD8452",
    "file_load": "#55A868",
}


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
    print("warning: 未找到常见中文字体，中文可能无法正常显示。")
    return None


def plot_box(df):
    data = []
    labels = []
    colors = []

    for path_type in PATH_ORDER:
        sub = df[df["path_type"] == path_type]["total_ms"].values
        sub = sub[np.isfinite(sub)]
        sub = sub[sub > 0]

        if len(sub) == 0:
            continue

        data.append(sub)
        labels.append(PATH_NAME[path_type])
        colors.append(PATH_COLOR[path_type])

    if not data:
        raise ValueError("没有可绘制的延迟数据。")

    plt.figure(figsize=(8, 5))

    flierprops = {
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgecolor": "#555555",
        "markersize": 5,
        "linestyle": "none",
    }

    medianprops = {
        "color": "#222222",
        "linewidth": 1.5,
    }

    whiskerprops = {
        "color": "#555555",
        "linewidth": 1.1,
    }

    capprops = {
        "color": "#555555",
        "linewidth": 1.1,
    }

    box = plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        flierprops=flierprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.1)

    plt.yscale("log")
    plt.title("不同 LoRA 加载路径的真实延迟分布")
    plt.xlabel("加载路径")
    plt.ylabel("适配器加载延迟（ms）")
    plt.grid(True, axis="y", linestyle="-", alpha=0.35)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    plt.show()

    print(f"图像已保存到: {OUT_PATH}")


def print_summary(df):
    summary = (
        df.groupby("path_type")["total_ms"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    summary["路径"] = summary["path_type"].map(PATH_NAME)

    print("\n不同 LoRA 加载路径延迟统计:")
    print(summary[["路径", "count", "mean", "median", "min", "max"]].to_string(index=False))

    cols = ["init_ms", "bind_ms", "decode_ms", "total_ms"]
    existing_cols = [c for c in cols if c in df.columns]

    detail = (
        df.groupby("path_type")[existing_cols]
        .mean()
        .reset_index()
    )

    detail["路径"] = detail["path_type"].map(PATH_NAME)

    print("\n各阶段平均耗时:")
    print(detail[["路径"] + existing_cols].to_string(index=False))


def main():
    setup_chinese_font()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到延迟数据文件: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    plot_box(df)
    print_summary(df)


if __name__ == "__main__":
    main()