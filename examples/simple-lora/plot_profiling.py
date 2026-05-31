import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT_DIR = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output"
CSV_PATH = os.path.join(OUTPUT_DIR, "profiling_results.csv")
OUT_PATH = os.path.join(OUTPUT_DIR, "profiling_stage_distribution.png")

SCHEME_ORDER = [
    "batch_unfused",
    "batch_grouped",
]

SCHEME_NAME = {
    "batch_unfused": "批处理",
    "batch_grouped": "批处理融合",
}

STAGE_ORDER = [
    "prefill_ms",
    "decode_ms",
    "sampling_ms",
    "other_ms",
]

STAGE_NAME = {
    "prefill_ms": "Prefill",
    "decode_ms": "Decode",
    "sampling_ms": "Sampling",
    "other_ms": "Other",
}

STAGE_COLOR = {
    "prefill_ms": "#4C72B0",
    "decode_ms": "#DD8452",
    "sampling_ms": "#55A868",
    "other_ms": "#C44E52",
}


def setup_chinese_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
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


def choose_target_group(df):
    if "n_lora" in df.columns:
        target_n_lora = int(df["n_lora"].max())
        return df[df["n_lora"] == target_n_lora].copy(), target_n_lora

    return df.copy(), None


def prepare_plot_data(df):
    df, target_n_lora = choose_target_group(df)

    rows = []

    for scheme in SCHEME_ORDER:
        sub = df[df["scheme"] == scheme]

        if sub.empty:
            continue

        row = {
            "scheme": scheme,
            "scheme_name": SCHEME_NAME.get(scheme, scheme),
        }

        total_ms = float(sub["total_ms"].mean())

        # LoRA 映射耗时只有 0.1ms 量级，远小于整体推理耗时。
        # 为了让图 5-10 更清晰，这里不单独绘制 LoRA 映射，
        # 但百分比仍然以 total_ms 为分母，表示各阶段占总推理时间的比例。
        for stage in STAGE_ORDER:
            stage_ms = float(sub[stage].mean())
            row[stage] = stage_ms
            row[stage + "_pct"] = stage_ms / total_ms * 100.0 if total_ms > 0 else 0.0

        row["total_ms"] = total_ms
        
        rows.append(row)

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        raise ValueError("没有找到可绘制的数据，请检查 profiling_results.csv 中的 scheme 字段。")

    return plot_df, target_n_lora


def plot_grouped_percent_bar(plot_df, target_n_lora):
    x = np.arange(len(plot_df))
    width = 0.15

    plt.figure(figsize=(8, 5))

    for idx, stage in enumerate(STAGE_ORDER):
        values = plot_df[stage + "_pct"].values
        offset = (idx - (len(STAGE_ORDER) - 1) / 2.0) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=STAGE_NAME[stage],
            color=STAGE_COLOR[stage],
            edgecolor="white",
            linewidth=0.7,
        )

    title = "Profiling 代码阶段耗时分布"
    if target_n_lora is not None:
        title += f"（并行 LoRA 数量={target_n_lora}）"

    plt.title(title)
    plt.xlabel("执行方案")
    plt.ylabel("时间占比（%）")
    plt.xticks(x, plot_df["scheme_name"].values)

    max_value = plot_df[[stage + "_pct" for stage in STAGE_ORDER]].to_numpy().max()
    plt.ylim(0, max(45, max_value * 1.18))

    plt.grid(True, axis="y", linestyle="-", alpha=0.35)
    plt.legend(title="代码阶段", loc="upper right")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    plt.show()

    print(f"图像已保存到: {OUT_PATH}")


def plot_grouped_time_bar(plot_df, target_n_lora):
    out_path = os.path.join(OUTPUT_DIR, "profiling_stage_time_ms.png")

    x = np.arange(len(plot_df))
    width = 0.15

    plt.figure(figsize=(8, 5))

    for idx, stage in enumerate(STAGE_ORDER):
        values = plot_df[stage].values
        offset = (idx - (len(STAGE_ORDER) - 1) / 2.0) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=STAGE_NAME[stage],
            color=STAGE_COLOR[stage],
            edgecolor="white",
            linewidth=0.7,
        )

    title = "Profiling 代码阶段耗时"
    if target_n_lora is not None:
        title += f"（并行 LoRA 数量={target_n_lora}）"

    plt.title(title)
    plt.xlabel("执行方案")
    plt.ylabel("耗时（ms）")
    plt.xticks(x, plot_df["scheme_name"].values)

    max_value = plot_df[STAGE_ORDER].to_numpy().max()
    plt.ylim(0, max_value * 1.18)

    plt.grid(True, axis="y", linestyle="-", alpha=0.35)
    plt.legend(title="代码阶段", loc="upper right")
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def print_table(plot_df):
    show_cols = ["scheme_name", "total_ms"]

    for stage in STAGE_ORDER:
        show_cols.append(stage)
        show_cols.append(stage + "_pct")

    show_df = plot_df[show_cols].copy()

    rename_map = {
        "scheme_name": "执行方案",
        "total_ms": "总耗时(ms)",
        "prefill_ms": "Prefill(ms)",
        "prefill_ms_pct": "Prefill(%)",
        "decode_ms": "Decode(ms)",
        "decode_ms_pct": "Decode(%)",
        "sampling_ms": "Sampling(ms)",
        "sampling_ms_pct": "Sampling(%)",
        "other_ms": "Other(ms)",
        "other_ms_pct": "Other(%)",
    }

    show_df = show_df.rename(columns=rename_map)

    print("\nProfiling 代码阶段耗时:")
    print(show_df.to_string(index=False))


def main():
    setup_chinese_font()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 profiling 文件: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    plot_df, target_n_lora = prepare_plot_data(df)

    plot_grouped_percent_bar(plot_df, target_n_lora)
    plot_grouped_time_bar(plot_df, target_n_lora)
    print_table(plot_df)


if __name__ == "__main__":
    main()