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

# 这里直接使用 run_prediction_experiments.py 的 method 名称。
# 如果你的 metrics_summary.csv 里方法名不一样，只需要改这里。
METHOD_ORDER = [
    "LRU",
    "GRU",
    "GRU+UserComb",
]

METHOD_NAME = {
    "LRU": "LRU",
    "GRU": "M3_Only",
    "GRU+UserComb": "M3+UserComb",
}

METHOD_COLOR = {
    "LRU": "#4C72B0",
    "GRU": "#DD8452",
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


def build_plot_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in METHOD_ORDER:
        sub = metrics[metrics["method"] == method]

        if sub.empty:
            print(f"warning: metrics_summary.csv 中找不到 method={method}")
            continue

        row = sub.iloc[0]

        for topk_col, topk_label in TOPK_ORDER:
            rows.append({
                "method": method,
                "method_name": METHOD_NAME.get(method, method),
                "topk": topk_label,
                "hit_rate": float(row[topk_col]),
            })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("没有可绘制的数据，请检查 METHOD_ORDER 是否和 metrics_summary.csv 一致。")

    return result


def plot_prediction_cache_hit(plot_df: pd.DataFrame, out_path: Path) -> None:
    setup_chinese_font()

    labels = [label for _, label in TOPK_ORDER]
    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(8, 5))

    for idx, method in enumerate(METHOD_ORDER):
        sub = plot_df[plot_df["method"] == method]

        if sub.empty:
            continue

        values = []

        for label in labels:
            item = sub[sub["topk"] == label]

            if item.empty:
                values.append(0.0)
            else:
                values.append(float(item.iloc[0]["hit_rate"]))

        offset = (idx - 1) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=METHOD_NAME.get(method, method),
            color=METHOD_COLOR.get(method, "#999999"),
            alpha=0.92,
        )

    plt.xlabel("GPU热缓存容量（Top-K）")
    plt.ylabel("适配器缓存命中率")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.02)
    plt.grid(True, axis="y", linestyle="-", alpha=0.35)
    plt.legend(title="策略", loc="upper left")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-predict/output/metrics_summary.csv",
        help="run_prediction_experiments.py 输出的 metrics_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-predict/output/prediction_cache_hit_from_metrics.png",
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_path = Path(args.output)

    metrics = load_metrics(metrics_path)
    plot_df = build_plot_table(metrics)

    csv_out = out_path.with_suffix(".csv")
    plot_df.to_csv(csv_out, index=False)

    print("\n用于绘图的数据:")
    print(plot_df.to_string(index=False))

    plot_prediction_cache_hit(plot_df, out_path)


if __name__ == "__main__":
    main()