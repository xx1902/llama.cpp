from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

METHOD_ORDER = [
    "LRU",
    "LFU",
    "GRU+UserComb",
]

METHOD_NAME = {
    "LRU": "LRU",
    "LFU": "LFU",
    "GRU+UserComb": "M3+UserComb",
}

METHOD_COLOR = {
    "LRU": "#4C72B0",
    "LFU": "#DD8452",
    "GRU+UserComb": "#55A868",
}

TOPK_ORDER = [
    ("top1", "Top-1"),
    ("top3", "Top-3"),
    ("top5", "Top-5"),
]


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


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"找不到 metrics_summary.csv: {metrics_path}")

    df = pd.read_csv(metrics_path)

    required_cols = {"method", "top1", "top3", "top5"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"metrics_summary.csv 缺少字段: {missing}")

    return df


def get_metric_value(metrics: pd.DataFrame, method: str, topk_col: str) -> float:
    sub = metrics[metrics["method"] == method]

    if sub.empty:
        raise ValueError(
            f"metrics_summary.csv 中找不到 method={method}。\n"
            f"如果缺少 LFU，请先在 fusion_engine.py 中加入 evaluate_lfu，"
            f"并重新运行 run_prediction_experiments.py。"
        )

    return float(sub.iloc[0][topk_col])


def build_plot_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in METHOD_ORDER:
        for topk_col, topk_label in TOPK_ORDER:
            rows.append({
                "method": method,
                "method_name": METHOD_NAME[method],
                "topk": topk_label,
                "topk_col": topk_col,
                "hit_rate": get_metric_value(metrics, method, topk_col),
            })

    return pd.DataFrame(rows)


def plot_prediction_cache_hit(plot_df: pd.DataFrame, out_path: Path) -> None:
    setup_chinese_font()

    labels = [label for _, label in TOPK_ORDER]
    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(8, 5), dpi=180)

    for idx, method in enumerate(METHOD_ORDER):
        sub = plot_df[plot_df["method"] == method]

        values = []

        for label in labels:
            item = sub[sub["topk"] == label]
            values.append(float(item.iloc[0]["hit_rate"]))

        offset = (idx - 1) * width

        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=METHOD_NAME[method],
            color=METHOD_COLOR[method],
            alpha=0.92,
        )

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xlabel("GPU热缓存容量（Top-K）", fontsize=12)
    plt.ylabel("适配器缓存命中率", fontsize=12)
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="-", alpha=0.35)
    plt.legend(title="策略", loc="upper left")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.show()

    print(f"图像已保存到: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-predict/output/metrics_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-predict/output/cache_hit_lru_lfu_m3_usercomb.png",
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_path = Path(args.output)

    metrics = load_metrics(metrics_path)
    plot_df = build_plot_table(metrics)

    csv_out = out_path.with_suffix(".csv")
    plot_df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("\n用于绘图的数据:")
    print(plot_df.to_string(index=False))

    plot_prediction_cache_hit(plot_df, out_path)


if __name__ == "__main__":
    main()