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

OUTPUT_DIR = Path(
    "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output"
)

OUTPUT_CSV = OUTPUT_DIR / "system_speedup_summary.csv"
OUTPUT_FIG = OUTPUT_DIR / "fig5_11_system_speedup.png"


SYSTEM_DATA = [
    {
        "system": "llama.cpp",
        "speedup": 1.00,
    },
    {
        "system": "+cache",
        "speedup": 1.13,
    },
    {
        "system": "+parallel",
        "speedup": 1.84,
    },
    {
        "system": "PreCoLoRA",
        "speedup": 2.42,
    },
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


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(SYSTEM_DATA)


def plot_speedup(df: pd.DataFrame) -> None:
    setup_font()

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=180)

    bars = ax.bar(
        df["system"],
        df["speedup"],
        color="#88AFC5",
        width=0.72,
        alpha=0.96,
    )

    for bar, value in zip(bars, df["speedup"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_title("不同系统配置的端到端吞吐量对比", fontsize=16, pad=14)
    ax.set_xlabel("系统配置", fontsize=12)
    ax.set_ylabel("端到端吞吐量加速比", fontsize=12)

    ax.set_ylim(0, 2.8)
    ax.grid(True, axis="y", linestyle="-", alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.show()

    print(f"图像已保存到: {OUTPUT_FIG}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataframe()
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n系统吞吐量加速比数据:")
    print(df.to_string(index=False))

    plot_speedup(df)


if __name__ == "__main__":
    main()