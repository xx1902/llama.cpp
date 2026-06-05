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

SUMMARY_CSV = OUTPUT_DIR / "system_ablation_summary.csv"
FIG_SPEEDUP = OUTPUT_DIR / "system_ablation_speedup.png"
FIG_TTFT = OUTPUT_DIR / "system_ablation_p95_ttft.png"

CONFIG_ORDER = [
    "native_llama.cpp",
    "w/o cache",
    "w/o batch",
    "w/o grouping",
    "PreCoLoRA",
]

COLORS = {
    "native_llama.cpp": "#88AFC5",
    "w/o cache": "#DD8452",
    "w/o batch": "#55A868",
    "w/o grouping": "#C44E52",
    "PreCoLoRA": "#4C72B0",
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

    plt.rcParams["axes.unicode_minus"] = False


def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"找不到文件: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    df["config"] = pd.Categorical(
        df["config"],
        categories=CONFIG_ORDER,
        ordered=True,
    )

    return df.sort_values("config")


def plot_speedup(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=180)

    bars = ax.bar(
        df["config"],
        df["speedup_vs_native"],
        color=[COLORS[str(x)] for x in df["config"]],
        width=0.68,
        alpha=0.95,
    )

    for bar, value in zip(bars, df["speedup_vs_native"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("PreCoLoRA 消融实验：端到端吞吐量对比", fontsize=15, pad=12)
    ax.set_xlabel("系统配置", fontsize=12)
    ax.set_ylabel("相对原生 llama.cpp 的吞吐量加速比", fontsize=12)

    ax.grid(True, axis="y", linestyle="-", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=15)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(FIG_SPEEDUP, bbox_inches="tight")
    plt.close(fig)

    print(f"吞吐量消融图已保存: {FIG_SPEEDUP}")


def plot_ttft(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=180)

    bars = ax.bar(
        df["config"],
        df["p95_ttft_ms"],
        color=[COLORS[str(x)] for x in df["config"]],
        width=0.68,
        alpha=0.95,
    )

    for bar, value in zip(bars, df["p95_ttft_ms"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(df["p95_ttft_ms"]) * 0.015,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("PreCoLoRA 消融实验：P95 TTFT 对比", fontsize=15, pad=12)
    ax.set_xlabel("系统配置", fontsize=12)
    ax.set_ylabel("P95 TTFT（ms）", fontsize=12)

    ax.grid(True, axis="y", linestyle="-", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=15)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(FIG_TTFT, bbox_inches="tight")
    plt.close(fig)

    print(f"P95 TTFT 消融图已保存: {FIG_TTFT}")


def main() -> None:
    setup_font()

    df = load_summary()

    print("\n消融实验结果:")
    print(df.to_string(index=False))

    plot_speedup(df)
    plot_ttft(df)


if __name__ == "__main__":
    main()