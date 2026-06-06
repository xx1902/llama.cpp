from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


OUTPUT_DIR = Path(
    "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-sys/output"
)

CSV_FILES = [
    OUTPUT_DIR / "nopage_nomultilora.csv",
    OUTPUT_DIR / "page_nomultilora.csv",
    OUTPUT_DIR / "nopage_multilora.csv",
    OUTPUT_DIR / "page_multilora.csv",
]

SUMMARY_CSV = OUTPUT_DIR / "system_compare_summary.csv"

FIG_OVERVIEW = OUTPUT_DIR / "system_compare_overview.png"
FIG_SPEEDUP = OUTPUT_DIR / "system_speedup_compare.png"
FIG_MEMORY = OUTPUT_DIR / "system_memory_compare.png"
FIG_PAGING_EFFECT = OUTPUT_DIR / "paging_effect_compare.png"
FIG_MULTILORA_EFFECT = OUTPUT_DIR / "multilora_effect_compare.png"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Source Han Sans SC",
]

CONFIG_LABELS = {
    "nopage_nomultilora": "连续KV\n无多LoRA",
    "page_nomultilora": "分页KV\n无多LoRA",
    "nopage_multilora": "连续KV\n多LoRA",
    "page_multilora": "分页KV\n多LoRA",
}

CONFIG_ORDER = [
    "nopage_nomultilora",
    "page_nomultilora",
    "nopage_multilora",
    "page_multilora",
]

CONFIG_COLORS = {
    "nopage_nomultilora": "#4C72B0",
    "page_nomultilora": "#55A868",
    "nopage_multilora": "#DD8452",
    "page_multilora": "#C44E52",
}


def setup_chinese_font() -> None:
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


def load_all_csv() -> pd.DataFrame:
    rows = []

    for csv_path in CSV_FILES:
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到文件: {csv_path}")

        df = pd.read_csv(csv_path)

        if len(df) != 1:
            print(f"warning: {csv_path.name} 有 {len(df)} 行，默认只使用第一行。")

        rows.append(df.iloc[0].to_dict())

    result = pd.DataFrame(rows)

    required_cols = {
        "config",
        "use_paged_kv",
        "use_multilora_batch",
        "n_requests",
        "n_lora",
        "n_ctx",
        "kv_page_size",
        "n_predict",
        "n_tokens",
        "elapsed_ms",
        "throughput_tps",
        "avg_ttft_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "avg_request_total_ms",
        "p95_request_total_ms",
        "gpu_start_mb",
        "gpu_peak_mb",
        "gpu_end_mb",
        "gpu_peak_delta_mb",
        "process_start_mb",
        "process_end_mb",
        "process_delta_mb",
    }

    missing = required_cols - set(result.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    result["config"] = result["config"].astype(str)
    result["config_label"] = result["config"].map(CONFIG_LABELS)

    result = result.set_index("config").loc[CONFIG_ORDER].reset_index()

    baseline_tps = float(
        result[result["config"] == "nopage_nomultilora"].iloc[0]["throughput_tps"]
    )

    result["elapsed_s"] = result["elapsed_ms"] / 1000.0
    result["speedup_vs_baseline"] = result["throughput_tps"] / baseline_tps
    result["gpu_peak_delta_gb"] = result["gpu_peak_delta_mb"] / 1024.0
    result["process_delta_gb"] = result["process_delta_mb"] / 1024.0

    return result


def add_bar_labels(ax, bars, fmt="{:.2f}", dy=0.02) -> None:
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + span * dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_overview(df: pd.DataFrame) -> None:
    setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    axes = axes.ravel()

    x = np.arange(len(df))
    labels = df["config_label"].tolist()
    colors = [CONFIG_COLORS[c] for c in df["config"]]

    items = [
        ("throughput_tps", "吞吐量 TPS", "tokens/s", "{:.2f}"),
        ("elapsed_s", "总耗时", "秒", "{:.1f}"),
        ("p95_ttft_ms", "P95 TTFT", "毫秒", "{:.0f}"),
        ("gpu_peak_delta_gb", "GPU 峰值显存增量", "GB", "{:.2f}"),
    ]

    for ax, (col, title, ylabel, fmt) in zip(axes, items):
        bars = ax.bar(x, df[col], color=colors, width=0.68, alpha=0.92)

        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(True, axis="y", linestyle="-", alpha=0.28)

        y_max = float(df[col].max())
        ax.set_ylim(0, y_max * 1.22 if y_max > 0 else 1.0)

        add_bar_labels(ax, bars, fmt=fmt)

    fig.suptitle("四种系统配置的核心指标对比", fontsize=16, y=0.995)
    fig.tight_layout()

    fig.savefig(FIG_OVERVIEW, bbox_inches="tight")
    plt.close(fig)

    print(f"总览图已保存: {FIG_OVERVIEW}")


def plot_speedup(df: pd.DataFrame) -> None:
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)

    x = np.arange(len(df))
    colors = [CONFIG_COLORS[c] for c in df["config"]]

    bars = ax.bar(
        x,
        df["speedup_vs_baseline"],
        color=colors,
        width=0.68,
        alpha=0.94,
    )

    ax.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--", alpha=0.7)

    ax.set_title("相对原生连续KV顺序执行的吞吐量加速比", fontsize=15)
    ax.set_xlabel("系统配置")
    ax.set_ylabel("加速比")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config_label"], fontsize=10)
    ax.grid(True, axis="y", linestyle="-", alpha=0.3)

    ax.set_ylim(0, max(df["speedup_vs_baseline"]) * 1.25)
    add_bar_labels(ax, bars, fmt="{:.2f}x")

    fig.tight_layout()
    fig.savefig(FIG_SPEEDUP, bbox_inches="tight")
    plt.close(fig)

    print(f"加速比图已保存: {FIG_SPEEDUP}")


def plot_memory(df: pd.DataFrame) -> None:
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)

    x = np.arange(len(df))
    width = 0.34

    gpu_bars = ax.bar(
        x - width / 2,
        df["gpu_peak_delta_gb"],
        width=width,
        color="#4C72B0",
        label="GPU峰值显存增量",
        alpha=0.92,
    )

    process_bars = ax.bar(
        x + width / 2,
        df["process_delta_gb"],
        width=width,
        color="#DD8452",
        label="进程内存增量",
        alpha=0.92,
    )

    ax.set_title("GPU显存与进程内存占用对比", fontsize=15)
    ax.set_xlabel("系统配置")
    ax.set_ylabel("内存增量 / GB")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config_label"], fontsize=10)
    ax.grid(True, axis="y", linestyle="-", alpha=0.3)
    ax.legend()

    ax.set_ylim(0, max(df["gpu_peak_delta_gb"].max(), df["process_delta_gb"].max()) * 1.2)

    add_bar_labels(ax, gpu_bars, fmt="{:.2f}")
    add_bar_labels(ax, process_bars, fmt="{:.2f}")

    fig.tight_layout()
    fig.savefig(FIG_MEMORY, bbox_inches="tight")
    plt.close(fig)

    print(f"内存对比图已保存: {FIG_MEMORY}")


def plot_paging_effect(df: pd.DataFrame) -> None:
    setup_chinese_font()

    no_multi_nopage = df[df["config"] == "nopage_nomultilora"].iloc[0]
    no_multi_page = df[df["config"] == "page_nomultilora"].iloc[0]
    multi_nopage = df[df["config"] == "nopage_multilora"].iloc[0]
    multi_page = df[df["config"] == "page_multilora"].iloc[0]

    groups = ["无多LoRA", "多LoRA"]
    nopage_tps = [
        no_multi_nopage["throughput_tps"],
        multi_nopage["throughput_tps"],
    ]
    page_tps = [
        no_multi_page["throughput_tps"],
        multi_page["throughput_tps"],
    ]

    x = np.arange(len(groups))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)

    bars1 = ax.bar(
        x - width / 2,
        nopage_tps,
        width=width,
        label="连续KV",
        color="#4C72B0",
        alpha=0.92,
    )

    bars2 = ax.bar(
        x + width / 2,
        page_tps,
        width=width,
        label="分页KV",
        color="#55A868",
        alpha=0.92,
    )

    ax.set_title("分页KV对吞吐量的影响", fontsize=15)
    ax.set_xlabel("执行方式")
    ax.set_ylabel("吞吐量 TPS")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(True, axis="y", linestyle="-", alpha=0.3)
    ax.legend()

    ax.set_ylim(0, max(max(nopage_tps), max(page_tps)) * 1.25)

    add_bar_labels(ax, bars1, fmt="{:.2f}")
    add_bar_labels(ax, bars2, fmt="{:.2f}")

    fig.tight_layout()
    fig.savefig(FIG_PAGING_EFFECT, bbox_inches="tight")
    plt.close(fig)

    print(f"分页影响图已保存: {FIG_PAGING_EFFECT}")


def plot_multilora_effect(df: pd.DataFrame) -> None:
    setup_chinese_font()

    nopage_nomulti = df[df["config"] == "nopage_nomultilora"].iloc[0]
    nopage_multi = df[df["config"] == "nopage_multilora"].iloc[0]
    page_nomulti = df[df["config"] == "page_nomultilora"].iloc[0]
    page_multi = df[df["config"] == "page_multilora"].iloc[0]

    groups = ["连续KV", "分页KV"]
    no_multi_tps = [
        nopage_nomulti["throughput_tps"],
        page_nomulti["throughput_tps"],
    ]
    multi_tps = [
        nopage_multi["throughput_tps"],
        page_multi["throughput_tps"],
    ]

    x = np.arange(len(groups))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)

    bars1 = ax.bar(
        x - width / 2,
        no_multi_tps,
        width=width,
        label="无多LoRA",
        color="#8172B2",
        alpha=0.92,
    )

    bars2 = ax.bar(
        x + width / 2,
        multi_tps,
        width=width,
        label="多LoRA批处理",
        color="#DD8452",
        alpha=0.92,
    )

    ax.set_title("多LoRA批处理对吞吐量的影响", fontsize=15)
    ax.set_xlabel("KV管理方式")
    ax.set_ylabel("吞吐量 TPS")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(True, axis="y", linestyle="-", alpha=0.3)
    ax.legend()

    ax.set_ylim(0, max(max(no_multi_tps), max(multi_tps)) * 1.25)

    add_bar_labels(ax, bars1, fmt="{:.2f}")
    add_bar_labels(ax, bars2, fmt="{:.2f}")

    fig.tight_layout()
    fig.savefig(FIG_MULTILORA_EFFECT, bbox_inches="tight")
    plt.close(fig)

    print(f"多LoRA影响图已保存: {FIG_MULTILORA_EFFECT}")


def print_markdown_table(df: pd.DataFrame) -> None:
    show_df = pd.DataFrame({
        "配置": df["config_label"].str.replace("\n", " / "),
        "TPS": df["throughput_tps"].map(lambda x: f"{x:.2f}"),
        "相对基线加速": df["speedup_vs_baseline"].map(lambda x: f"{x:.2f}x"),
        "总耗时(s)": df["elapsed_s"].map(lambda x: f"{x:.2f}"),
        "平均TTFT(ms)": df["avg_ttft_ms"].map(lambda x: f"{x:.2f}"),
        "P95 TTFT(ms)": df["p95_ttft_ms"].map(lambda x: f"{x:.2f}"),
        "GPU峰值增量(GB)": df["gpu_peak_delta_gb"].map(lambda x: f"{x:.2f}"),
        "进程内存增量(MB)": df["process_delta_mb"].map(lambda x: f"{x:.2f}"),
    })

    print("\nMarkdown 对比表:")
    print(show_df.to_markdown(index=False))


def print_key_findings(df: pd.DataFrame) -> None:
    base = df[df["config"] == "nopage_nomultilora"].iloc[0]
    page_nomulti = df[df["config"] == "page_nomultilora"].iloc[0]
    nopage_multi = df[df["config"] == "nopage_multilora"].iloc[0]
    page_multi = df[df["config"] == "page_multilora"].iloc[0]

    def pct(new, old):
        return (float(new) - float(old)) / float(old) * 100.0

    paging_no_multi_tps = pct(page_nomulti["throughput_tps"], base["throughput_tps"])
    paging_multi_tps = pct(page_multi["throughput_tps"], nopage_multi["throughput_tps"])

    multilora_nopage_speedup = nopage_multi["throughput_tps"] / base["throughput_tps"]
    multilora_page_speedup = page_multi["throughput_tps"] / page_nomulti["throughput_tps"]

    paging_no_multi_gpu_save = base["gpu_peak_delta_mb"] - page_nomulti["gpu_peak_delta_mb"]
    paging_multi_gpu_save = nopage_multi["gpu_peak_delta_mb"] - page_multi["gpu_peak_delta_mb"]

    print("\n关键结论:")
    print(
        f"1. 多LoRA批处理是主要吞吐收益来源：连续KV下 TPS 从 "
        f"{base['throughput_tps']:.2f} 提升到 {nopage_multi['throughput_tps']:.2f}，"
        f"约为 {multilora_nopage_speedup:.2f}x。"
    )
    print(
        f"2. 分页KV在当前参数下会带来一定速度开销：无多LoRA时 TPS 变化 "
        f"{paging_no_multi_tps:.2f}%，多LoRA时 TPS 变化 {paging_multi_tps:.2f}%。"
    )
    print(
        f"3. 分页KV的GPU峰值显存降低幅度较小：无多LoRA节省约 "
        f"{paging_no_multi_gpu_save:.0f} MB，多LoRA节省约 {paging_multi_gpu_save:.0f} MB。"
    )
    print(
        f"4. 当前分页实验的 kv_page_size={int(page_multi['kv_page_size'])}，页较大，"
        f"所以显存节省不会特别明显；模型权重、LoRA权重和计算buffer仍然占主要显存。"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_csv()
    df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(f"汇总CSV已保存: {SUMMARY_CSV}")

    print_markdown_table(df)
    print_key_findings(df)

    plot_overview(df)
    plot_speedup(df)
    plot_memory(df)
    plot_paging_effect(df)
    plot_multilora_effect(df)


if __name__ == "__main__":
    main()