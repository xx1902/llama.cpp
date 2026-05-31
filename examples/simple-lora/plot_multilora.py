import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

CSV_PATH = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output/multilora_results.csv"
OUT_PATH = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output/multilora_tps.png"


def setup_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",      # 加上
        "Noto Serif CJK JP",     # 加上
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


def main():
    font_name = setup_chinese_font()
    if font_name is None:
        print("warning: 未找到常见中文字体，中文可能无法正常显示。")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 文件不存在: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    name_map = {
        "sequential": "顺序",
        "batch_unfused": "批处理",
        "batch_grouped": "批处理融合",
    }

    df["方案中文"] = df["scheme"].map(name_map)

    plt.figure(figsize=(8, 5))

    styles = {
        "顺序": {
            "color": "#1f77b4",
            "marker": "o",
        },
        "批处理": {
            "color": "#ff7f0e",
            "marker": "s",
        },
        "批处理融合": {
            "color": "#2ca02c",
            "marker": "^",
        },
    }

    for scheme in ["顺序", "批处理", "批处理融合"]:
        sub = df[df["方案中文"] == scheme].sort_values("n_lora") # type: ignore

        if sub.empty:
            continue

        plt.plot(
            sub["n_lora"],
            sub["tps"],
            label=scheme,
            color=styles[scheme]["color"],
            marker=styles[scheme]["marker"],
            linewidth=1.8,
            markersize=5,
        )

    plt.title("不同多 LoRA 执行方案下的吞吐量对比")
    plt.xlabel("并行 LoRA 数量")
    plt.ylabel("吞吐量 (tokens/s)")

    x_values = sorted(df["n_lora"].unique())
    plt.xticks(x_values)

    plt.grid(True, linestyle="-", alpha=0.3)
    plt.legend(title="方案", loc="upper left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    plt.show()

    print(f"图像已保存到: {OUT_PATH}")


if __name__ == "__main__":
    main()