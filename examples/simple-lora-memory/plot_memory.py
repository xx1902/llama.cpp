import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT_DIR = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output"

CACHE_CSV = os.path.join(OUTPUT_DIR, "real_lora_cache_results.csv")
LATENCY_CSV = os.path.join(OUTPUT_DIR, "real_lora_path_latency.csv")

POLICY_ORDER = ["LRU", "LFU", "Predictive"]

POLICY_NAME = {
    "LRU": "LRU",
    "LFU": "LFU",
    "Predictive": "预测驱动",
}

POLICY_STYLE = {
    "LRU": {
        "color": "#1f77b4",
        "marker": "o",
    },
    "LFU": {
        "color": "#ff7f0e",
        "marker": "s",
    },
    "Predictive": {
        "color": "#2ca02c",
        "marker": "^",
    },
}

PATH_ORDER = [
    "loaded_cache_hit",
    "file_load_miss",
]

PATH_NAME = {
    "loaded_cache_hit": "缓存命中复用",
    "file_load_miss": "文件加载未命中",
}

PATH_COLOR = {
    "loaded_cache_hit": "#2ca02c",
    "file_load_miss": "#d62728",
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
    values = values[values >= 0]

    if len(values) == 0:
        return np.array([]), np.array([])

    values = np.sort(values)
    cdf = np.arange(1, len(values) + 1) / len(values)

    return values, cdf


def plot_hit_rate(cache_df):
    for distribution in ["uniform", "zipf"]:
        sub_df = cache_df[cache_df["distribution"] == distribution]

        if sub_df.empty:
            continue

        plt.figure(figsize=(8, 5))

        for policy in POLICY_ORDER:
            sub = sub_df[sub_df["policy"] == policy].sort_values("capacity")

            if sub.empty:
                continue

            style = POLICY_STYLE[policy]

            plt.plot(
                sub["capacity"],
                sub["hit_rate"] * 100.0,
                label=POLICY_NAME[policy],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=5,
            )

        title_map = {
            "uniform": "Uniform 负载下真实 LoRA 缓存命中率",
            "zipf": "Zipf 负载下真实 LoRA 缓存命中率",
        }

        out_map = {
            "uniform": "real_cache_hit_rate_uniform.png",
            "zipf": "real_cache_hit_rate_zipf.png",
        }

        plt.title(title_map[distribution])
        plt.xlabel("缓存容量（Adapter数量）")
        plt.ylabel("缓存命中率（%）")
        plt.xticks(sorted(sub_df["capacity"].unique()))
        plt.grid(True, linestyle="-", alpha=0.3)
        plt.legend(title="缓存策略", loc="lower right")
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, out_map[distribution])
        plt.savefig(out_path, dpi=200)
        plt.show()

        print(f"图像已保存到: {out_path}")


def plot_avg_latency(cache_df):
    for distribution in ["uniform", "zipf"]:
        sub_df = cache_df[cache_df["distribution"] == distribution]

        if sub_df.empty:
            continue

        plt.figure(figsize=(8, 5))

        for policy in POLICY_ORDER:
            sub = sub_df[sub_df["policy"] == policy].sort_values("capacity")

            if sub.empty:
                continue

            style = POLICY_STYLE[policy]

            plt.plot(
                sub["capacity"],
                sub["avg_latency_ms"],
                label=POLICY_NAME[policy],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=5,
            )

        title_map = {
            "uniform": "Uniform 负载下真实 LoRA 平均加载/复用延迟",
            "zipf": "Zipf 负载下真实 LoRA 平均加载/复用延迟",
        }

        out_map = {
            "uniform": "real_cache_avg_latency_uniform.png",
            "zipf": "real_cache_avg_latency_zipf.png",
        }

        plt.title(title_map[distribution])
        plt.xlabel("缓存容量（Adapter数量）")
        plt.ylabel("平均延迟（ms）")
        plt.xticks(sorted(sub_df["capacity"].unique()))
        plt.grid(True, linestyle="-", alpha=0.3)
        plt.legend(title="缓存策略", loc="upper right")
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, out_map[distribution])
        plt.savefig(out_path, dpi=200)
        plt.show()

        print(f"图像已保存到: {out_path}")


def plot_path_latency_cdf(latency_df):
    target_policy = "Predictive"
    target_distribution = "zipf"

    if "capacity" in latency_df.columns:
        target_capacity = int(latency_df["capacity"].max())
        df = latency_df[
            (latency_df["policy"] == target_policy)
            & (latency_df["distribution"] == target_distribution)
            & (latency_df["capacity"] == target_capacity)
        ].copy()
    else:
        target_capacity = None
        df = latency_df[
            (latency_df["policy"] == target_policy)
            & (latency_df["distribution"] == target_distribution)
        ].copy()

    if df.empty:
        print("warning: 找不到用于绘制路径延迟 CDF 的真实数据。")
        return

    plt.figure(figsize=(8, 5))

    for path_type in PATH_ORDER:
        sub = df[df["path_type"] == path_type]

        if sub.empty:
            continue

        x, y = calc_cdf(sub["latency_ms"].values)

        plt.plot(
            x,
            y,
            label=PATH_NAME[path_type],
            color=PATH_COLOR[path_type],
            linewidth=1.8,
        )

    title = "真实 LoRA 加载路径延迟分布"
    if target_capacity is not None:
        title += f"（缓存容量={target_capacity}）"

    plt.title(title)
    plt.xlabel("延迟（ms）")
    plt.ylabel("CDF")
    plt.grid(True, linestyle="-", alpha=0.3)
    plt.legend(title="路径类型", loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "real_lora_path_latency_cdf.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def plot_path_latency_box(latency_df):
    target_policy = "Predictive"
    target_distribution = "zipf"

    if "capacity" in latency_df.columns:
        target_capacity = int(latency_df["capacity"].max())
        df = latency_df[
            (latency_df["policy"] == target_policy)
            & (latency_df["distribution"] == target_distribution)
            & (latency_df["capacity"] == target_capacity)
        ].copy()
    else:
        target_capacity = None
        df = latency_df[
            (latency_df["policy"] == target_policy)
            & (latency_df["distribution"] == target_distribution)
        ].copy()

    if df.empty:
        print("warning: 找不到用于绘制路径延迟箱线图的真实数据。")
        return

    data = []
    labels = []
    colors = []

    for path_type in PATH_ORDER:
        sub = df[df["path_type"] == path_type]["latency_ms"].values

        if len(sub) == 0:
            continue

        data.append(sub)
        labels.append(PATH_NAME[path_type])
        colors.append(PATH_COLOR[path_type])

    if len(data) == 0:
        print("warning: 路径延迟数据为空。")
        return

    plt.figure(figsize=(8, 5))

    box = plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.5)

    title = "真实 LoRA 加载路径延迟对比"
    if target_capacity is not None:
        title += f"（缓存容量={target_capacity}）"

    plt.title(title)
    plt.xlabel("路径类型")
    plt.ylabel("延迟（ms）")
    plt.grid(True, axis="y", linestyle="-", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "real_lora_path_latency_box.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def plot_path_ratio(latency_df):
    target_distribution = "zipf"

    df = latency_df[latency_df["distribution"] == target_distribution].copy()

    if df.empty:
        print("warning: 找不到路径占比数据。")
        return

    rows = []

    for policy in POLICY_ORDER:
        for capacity in sorted(df["capacity"].unique()):
            sub = df[
                (df["policy"] == policy)
                & (df["capacity"] == capacity)
            ]

            if sub.empty:
                continue

            total = len(sub)
            hit_count = int((sub["path_type"] == "loaded_cache_hit").sum())
            miss_count = int((sub["path_type"] == "file_load_miss").sum())

            rows.append({
                "policy": policy,
                "capacity": capacity,
                "hit_ratio": hit_count / total if total > 0 else 0.0,
                "miss_ratio": miss_count / total if total > 0 else 0.0,
            })

    ratio_df = pd.DataFrame(rows)

    if ratio_df.empty:
        return

    plt.figure(figsize=(8, 5))

    for policy in POLICY_ORDER:
        sub = ratio_df[ratio_df["policy"] == policy].sort_values("capacity")

        if sub.empty:
            continue

        style = POLICY_STYLE[policy]

        plt.plot(
            sub["capacity"],
            sub["hit_ratio"] * 100.0,
            label=POLICY_NAME[policy],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5,
        )

    plt.title("Zipf 负载下真实缓存复用路径占比")
    plt.xlabel("缓存容量（Adapter数量）")
    plt.ylabel("缓存命中复用占比（%）")
    plt.xticks(sorted(ratio_df["capacity"].unique()))
    plt.grid(True, linestyle="-", alpha=0.3)
    plt.legend(title="缓存策略", loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "real_lora_path_hit_ratio_zipf.png")
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"图像已保存到: {out_path}")


def print_summary(cache_df, latency_df):
    print("\n真实缓存命中率结果:")
    print(cache_df.to_string(index=False))

    print("\n真实路径延迟统计:")
    summary = (
        latency_df
        .groupby(["policy", "distribution", "capacity", "path_type"])["latency_ms"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    print(summary.to_string(index=False))


def main():
    setup_chinese_font()

    if not os.path.exists(CACHE_CSV):
        raise FileNotFoundError(f"找不到真实缓存结果文件: {CACHE_CSV}")

    if not os.path.exists(LATENCY_CSV):
        raise FileNotFoundError(f"找不到真实路径延迟文件: {LATENCY_CSV}")

    cache_df = pd.read_csv(CACHE_CSV)
    latency_df = pd.read_csv(LATENCY_CSV)

    plot_hit_rate(cache_df)
    plot_avg_latency(cache_df)
    plot_path_latency_cdf(latency_df)
    plot_path_latency_box(latency_df)
    plot_path_ratio(latency_df)

    print_summary(cache_df, latency_df)


if __name__ == "__main__":
    main()