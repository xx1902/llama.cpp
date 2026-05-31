import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from matplotlib.lines import Line2D

OUTPUT_DIR = "/home/agx_orin/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output"
SAMPLES_CSV = os.path.join(OUTPUT_DIR, "latency_samples.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "latency_summary.csv")

SCHEME_ORDER = ["sequential", "batch_unfused", "batch_grouped"]

SCHEME_NAME = {
    "sequential": "顺序",
    "batch_unfused": "批处理",
    "batch_grouped": "批处理融合",
}

SCHEME_STYLE = {
    "sequential": {
        "color": "#1f77b4",
        "marker": "o",
    },
    "batch_unfused": {
        "color": "#ff7f0e",
        "marker": "s",
    },
    "batch_grouped": {
        "color": "#2ca02c",
        "marker": "^",
    },
}


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
    print("warning: 未找到常见中文字体，中文可能无法正常显示。")
    return None


def calc_cdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) == 0:
        return np.array([]), np.array([])

    values = np.sort(values)
    cdf = np.arange(1, len(values) + 1) / len(values)

    return values, cdf


def plot_cdf(df, metric_col, xlabel, title, out_name):
    plt.figure(figsize=(8, 5))

    for scheme in SCHEME_ORDER:
        sub = df[df["scheme"] == scheme]
        x, y = calc_cdf(sub[metric_col].values)

        if len(x) == 0:
            continue

        style = SCHEME_STYLE[scheme]

        plt.plot(
            x,
            y,
            label=SCHEME_NAME[scheme],
            color=style["color"],
            marker=style["marker"],
            markevery=max(len(x) // 8, 1),
            linewidth=1.8,
            markersize=5,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True, linestyle="-", alpha=0.3)
    plt.legend(title="方案", loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def plot_percentiles(summary, prefix, title, ylabel, out_name):
    x = np.arange(3)
    width = 0.24

    labels = ["P50", "P95", "P99"]
    cols = [
        f"{prefix}_p50",
        f"{prefix}_p95",
        f"{prefix}_p99",
    ]

    plt.figure(figsize=(8, 5))

    for idx, scheme in enumerate(SCHEME_ORDER):
        sub = summary[summary["scheme"] == scheme]

        if sub.empty:
            continue

        values = [float(sub.iloc[0][col]) for col in cols]
        offset = (idx - 1) * width
        style = SCHEME_STYLE[scheme]

        plt.bar(
            x + offset,
            values,
            width=width,
            color=style["color"],
            alpha=0.9,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SCHEME_STYLE[scheme]["color"],
            marker=SCHEME_STYLE[scheme]["marker"],
            linewidth=1.8,
            markersize=6,
            label=SCHEME_NAME[scheme],
        )
        for scheme in SCHEME_ORDER
    ]

    plt.title(title)
    plt.xlabel("分位值")
    plt.ylabel(ylabel)
    plt.xticks(x, labels)
    plt.grid(True, axis="y", linestyle="-", alpha=0.3)
    plt.legend(handles=legend_handles, title="方案", loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def print_summary(summary):
    display = summary.copy()
    display["方案"] = display["scheme"].map(SCHEME_NAME)

    cols = [
        "方案",
        "n_samples",
        "ttft_p50",
        "ttft_p95",
        "ttft_p99",
        "user_tgi_p50",
        "user_tgi_p95",
        "user_tgi_p99",
        "system_tgi_p50",
        "system_tgi_p95",
        "system_tgi_p99",
        "total_p50",
        "total_p95",
        "total_p99",
    ]

    existing_cols = [col for col in cols if col in display.columns]

    print("\n延迟分位统计:")
    print(display[existing_cols].to_string(index=False))


def main():
    setup_chinese_font()

    if not os.path.exists(SAMPLES_CSV):
        raise FileNotFoundError(f"找不到 latency samples 文件: {SAMPLES_CSV}")

    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"找不到 latency summary 文件: {SUMMARY_CSV}")

    df = pd.read_csv(SAMPLES_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    plot_cdf(
        df,
        metric_col="ttft_ms",
        xlabel="TTFT (ms)",
        title="不同多 LoRA 执行方案下的 TTFT 分布",
        out_name="latency_ttft_cdf.png",
    )

    plot_cdf(
        df,
        metric_col="user_tgi_ms",
        xlabel="用户感知 TGI (ms/token)",
        title="不同多 LoRA 执行方案下的用户感知 TGI 分布",
        out_name="latency_user_tgi_cdf.png",
    )

    plot_cdf(
        df,
        metric_col="system_tgi_ms",
        xlabel="系统级 TGI (ms/token)",
        title="不同多 LoRA 执行方案下的系统级 TGI 分布",
        out_name="latency_system_tgi_cdf.png",
    )

    plot_cdf(
        df,
        metric_col="total_latency_ms",
        xlabel="总延迟 (ms)",
        title="不同多 LoRA 执行方案下的总延迟分布",
        out_name="latency_total_cdf.png",
    )

    plot_percentiles(
        summary,
        prefix="ttft",
        title="TTFT 延迟分位值",
        ylabel="TTFT (ms)",
        out_name="latency_ttft_percentiles.png",
    )

    plot_percentiles(
        summary,
        prefix="user_tgi",
        title="用户感知 TGI 分位值",
        ylabel="用户感知 TGI (ms/token)",
        out_name="latency_user_tgi_percentiles.png",
    )

    plot_percentiles(
        summary,
        prefix="system_tgi",
        title="系统级 TGI 分位值",
        ylabel="系统级 TGI (ms/token)",
        out_name="latency_system_tgi_percentiles.png",
    )

    plot_percentiles(
        summary,
        prefix="total",
        title="总延迟分位值",
        ylabel="总延迟 (ms)",
        out_name="latency_total_percentiles.png",
    )

    print_summary(summary)


if __name__ == "__main__":
    main()