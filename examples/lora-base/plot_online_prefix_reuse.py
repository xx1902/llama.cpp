from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_CSV = OUTPUT_DIR / "group_node_kv_lora_tree_summary.csv"
OUTPUT_FIG = OUTPUT_DIR / "online_prefix_reuse_compare.png"
OUTPUT_BENEFIT_CSV = OUTPUT_DIR / "online_prefix_reuse_benefit.csv"
OUTPUT_AGG_CSV = OUTPUT_DIR / "online_prefix_reuse_agg.csv"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]

LORA_NAME = {
    "code_r16": "code_r16",
    "code_r16v2": "code_r16v2",
    "code_r16v3": "code_r16v3",
    "correction": "correction",
    "lyrics": "lyrics",
    "marketing": "marketing",
}

MODE_NAME = {
    "baseline_no_group_kv_reuse": "不复用KV",
    "online_prefix_build": "在线建Prefix节点",
    "online_prefix_reuse": "在线复用Prefix KV",
    "online_suffix_delta_materialize": "Suffix delta materialize",
}

MODE_COLOR = {
    "baseline_no_group_kv_reuse": "#8F969E",
    "online_prefix_build": "#D99058",
    "online_prefix_reuse": "#3A947C",
    "online_suffix_delta_materialize": "#4C72B0",
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
    else:
        print("warning: 未找到常见中文字体，中文可能显示异常。")

    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到实验结果文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = {
        "mode",
        "group_name",
        "lora_name",
        "n_prompt_tokens",
        "n_prefix_tokens",
        "n_suffix_tokens",
        "prefix_reuse_rate",
        "suffix_delta_rate",
        "prefix_ms",
        "suffix_ms",
        "ttft_ms",
        "total_ms",
        "tps",
        "gpu_peak_delta_mb",
        "estimated_saved_kv_mb",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    if "repeat_id" not in df.columns:
        df["repeat_id"] = 0

    if "context_tag" not in df.columns:
        df["context_tag"] = "short"

    if "materialize_timed" not in df.columns:
        df["materialize_timed"] = 1

    if "suffix_delta_materialized" not in df.columns:
        df["suffix_delta_materialized"] = 0

    if "delta_materialize_ms" not in df.columns:
        df["delta_materialize_ms"] = 0.0

    if "materialized_kv_mb" not in df.columns:
        df["materialized_kv_mb"] = 0.0

    if "delta_fp32_mb" not in df.columns:
        df["delta_fp32_mb"] = 0.0

    df["lora_label"] = df["lora_name"].map(LORA_NAME).fillna(df["lora_name"])
    df["mode_label"] = df["mode"].map(MODE_NAME).fillna(df["mode"])
    df["prefix_reuse_percent"] = df["prefix_reuse_rate"] * 100.0
    df["suffix_delta_percent"] = df["suffix_delta_rate"] * 100.0
    df["gpu_peak_delta_gb"] = df["gpu_peak_delta_mb"] / 1024.0

    return df


def clean_and_aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    invalid_materialize = (
        (df["mode"] == "online_suffix_delta_materialize")
        & (
            (df["materialize_timed"] != 1)
            | (df["suffix_delta_materialized"] != 1)
            | (df["ttft_ms"] <= 0)
            | (df["total_ms"] <= 0)
        )
    )

    df["valid_timing"] = True
    df.loc[df["ttft_ms"] <= 0, "valid_timing"] = False
    df.loc[df["total_ms"] <= 0, "valid_timing"] = False
    df.loc[invalid_materialize, "valid_timing"] = False

    invalid_count = int((~df["valid_timing"]).sum())
    if invalid_count > 0:
        print(f"\n提示：过滤/标注了 {invalid_count} 条没有真实计时的数据。")
        invalid_show = df[~df["valid_timing"]][
            [
                "context_tag",
                "repeat_id",
                "group_name",
                "lora_name",
                "mode",
                "materialize_timed",
                "suffix_delta_materialized",
                "ttft_ms",
                "total_ms",
            ]
        ]
        print(invalid_show.to_markdown(index=False, floatfmt=".3f"))

    valid = df[df["valid_timing"]].copy()

    group_cols = [
        "context_tag",
        "group_name",
        "lora_name",
        "mode",
    ]

    agg = valid.groupby(group_cols, as_index=False).agg(
        repeat_count=("repeat_id", "count"),
        ttft_ms=("ttft_ms", "mean"),
        ttft_std_ms=("ttft_ms", "std"),
        total_ms=("total_ms", "mean"),
        total_std_ms=("total_ms", "std"),
        tps=("tps", "mean"),
        tps_std=("tps", "std"),
        prefix_ms=("prefix_ms", "mean"),
        suffix_ms=("suffix_ms", "mean"),
        delta_materialize_ms=("delta_materialize_ms", "mean"),
        n_prompt_tokens=("n_prompt_tokens", "mean"),
        n_prefix_tokens=("n_prefix_tokens", "mean"),
        n_suffix_tokens=("n_suffix_tokens", "mean"),
        prefix_reuse_rate=("prefix_reuse_rate", "mean"),
        suffix_delta_rate=("suffix_delta_rate", "mean"),
        gpu_peak_delta_mb=("gpu_peak_delta_mb", "mean"),
        estimated_saved_kv_mb=("estimated_saved_kv_mb", "mean"),
        suffix_delta_materialized=("suffix_delta_materialized", "max"),
        materialize_timed=("materialize_timed", "max"),
        delta_fp32_mb=("delta_fp32_mb", "mean"),
        materialized_kv_mb=("materialized_kv_mb", "mean"),
    )

    for col in ["ttft_std_ms", "total_std_ms", "tps_std"]:
        agg[col] = agg[col].fillna(0.0)

    agg["lora_label"] = agg["lora_name"].map(LORA_NAME).fillna(agg["lora_name"])
    agg["mode_label"] = agg["mode"].map(MODE_NAME).fillna(agg["mode"])
    agg["prefix_reuse_percent"] = agg["prefix_reuse_rate"] * 100.0
    agg["suffix_delta_percent"] = agg["suffix_delta_rate"] * 100.0

    agg.to_csv(OUTPUT_AGG_CSV, index=False, encoding="utf-8-sig")

    return agg


def choose_online_rows(df: pd.DataFrame) -> pd.DataFrame:
    online = df[df["mode"].isin([
        "online_prefix_reuse",
        "online_suffix_delta_materialize",
    ])].copy()

    if online.empty:
        return online

    priority = {
        "online_suffix_delta_materialize": 0,
        "online_prefix_reuse": 1,
    }

    online["mode_priority"] = online["mode"].map(priority).fillna(99)

    online = online.sort_values([
        "context_tag",
        "lora_name",
        "mode_priority",
    ])

    return online.drop_duplicates(
        subset=["context_tag", "lora_name"],
        keep="first",
    )


def build_benefit(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["mode"] == "baseline_no_group_kv_reuse"].copy()
    online = choose_online_rows(df)

    rows = []

    for _, base in baseline.iterrows():
        matched = online[
            (online["context_tag"] == base["context_tag"])
            & (online["lora_name"] == base["lora_name"])
        ]

        if matched.empty:
            continue

        cur = matched.iloc[0]

        base_ttft = float(base["ttft_ms"])
        cur_ttft = float(cur["ttft_ms"])
        base_total = float(base["total_ms"])
        cur_total = float(cur["total_ms"])
        base_tps = float(base["tps"])
        cur_tps = float(cur["tps"])

        rows.append({
            "context_tag": cur["context_tag"],
            "group_name": cur["group_name"],
            "lora_name": cur["lora_name"],
            "lora_label": cur["lora_label"],
            "online_mode": cur["mode"],
            "online_mode_label": cur["mode_label"],
            "repeat_count": int(cur["repeat_count"]),
            "prompt_tokens": float(cur["n_prompt_tokens"]),
            "prefix_tokens": float(cur["n_prefix_tokens"]),
            "suffix_tokens": float(cur["n_suffix_tokens"]),
            "prefix_reuse_percent": float(cur["prefix_reuse_percent"]),
            "suffix_delta_percent": float(cur["suffix_delta_percent"]),
            "baseline_ttft_ms": base_ttft,
            "online_ttft_ms": cur_ttft,
            "ttft_drop_percent": (base_ttft - cur_ttft) / max(base_ttft, 1e-6) * 100.0,
            "baseline_total_ms": base_total,
            "online_total_ms": cur_total,
            "total_drop_percent": (base_total - cur_total) / max(base_total, 1e-6) * 100.0,
            "baseline_tps": base_tps,
            "online_tps": cur_tps,
            "tps_change_percent": (cur_tps - base_tps) / max(base_tps, 1e-6) * 100.0,
            "prefix_ms": float(cur["prefix_ms"]),
            "suffix_ms": float(cur["suffix_ms"]),
            "delta_materialize_ms": float(cur["delta_materialize_ms"]),
            "estimated_saved_kv_mb": float(cur["estimated_saved_kv_mb"]),
            "delta_fp32_mb": float(cur["delta_fp32_mb"]),
            "materialized_kv_mb": float(cur["materialized_kv_mb"]),
        })

    benefit = pd.DataFrame(rows)
    benefit.to_csv(OUTPUT_BENEFIT_CSV, index=False, encoding="utf-8-sig")
    return benefit


def print_summary(df: pd.DataFrame, benefit: pd.DataFrame) -> None:
    show = df[
        [
            "context_tag",
            "group_name",
            "lora_label",
            "mode_label",
            "repeat_count",
            "n_prompt_tokens",
            "n_prefix_tokens",
            "n_suffix_tokens",
            "prefix_reuse_percent",
            "ttft_ms",
            "ttft_std_ms",
            "total_ms",
            "total_std_ms",
            "tps",
            "prefix_ms",
            "suffix_ms",
            "delta_materialize_ms",
            "estimated_saved_kv_mb",
        ]
    ].rename(columns={
        "context_tag": "上下文",
        "group_name": "组",
        "lora_label": "LoRA",
        "mode_label": "模式",
        "repeat_count": "重复次数",
        "n_prompt_tokens": "Prompt tokens",
        "n_prefix_tokens": "复用Prefix tokens",
        "n_suffix_tokens": "剩余Suffix tokens",
        "prefix_reuse_percent": "Prefix复用率(%)",
        "ttft_ms": "TTFT均值(ms)",
        "ttft_std_ms": "TTFT标准差",
        "total_ms": "总耗时均值(ms)",
        "total_std_ms": "总耗时标准差",
        "tps": "TPS均值",
        "prefix_ms": "Prefix耗时(ms)",
        "suffix_ms": "Suffix耗时(ms)",
        "delta_materialize_ms": "Delta materialize耗时(ms)",
        "estimated_saved_kv_mb": "估算节省KV(MB)",
    })

    print("\n在线 Prefix / Suffix Delta 实验结果，已按 repeat 求平均:")
    print(show.to_markdown(index=False, floatfmt=".3f"))

    if benefit.empty:
        print("\n没有可计算收益的数据，请检查 baseline 和 online 是否都存在。")
        return

    benefit_show = benefit[
        [
            "context_tag",
            "lora_label",
            "online_mode_label",
            "repeat_count",
            "ttft_drop_percent",
            "total_drop_percent",
            "tps_change_percent",
            "prefix_tokens",
            "suffix_tokens",
            "prefix_ms",
            "suffix_ms",
            "delta_materialize_ms",
            "estimated_saved_kv_mb",
        ]
    ].rename(columns={
        "context_tag": "上下文",
        "lora_label": "LoRA",
        "online_mode_label": "在线模式",
        "repeat_count": "重复次数",
        "ttft_drop_percent": "TTFT下降(%)",
        "total_drop_percent": "总耗时下降(%)",
        "tps_change_percent": "TPS变化(%)",
        "prefix_tokens": "复用Prefix tokens",
        "suffix_tokens": "剩余Suffix tokens",
        "prefix_ms": "Prefix耗时(ms)",
        "suffix_ms": "Suffix耗时(ms)",
        "delta_materialize_ms": "Delta耗时(ms)",
        "estimated_saved_kv_mb": "估算节省KV(MB)",
    })

    print("\n相对不复用 KV 的收益:")
    print(benefit_show.to_markdown(index=False, floatfmt=".3f"))


def get_compare_rows(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["mode"] == "baseline_no_group_kv_reuse"].copy()
    online = choose_online_rows(df)

    rows = []

    for _, base in baseline.iterrows():
        cur = online[
            (online["context_tag"] == base["context_tag"])
            & (online["lora_name"] == base["lora_name"])
        ]

        if cur.empty:
            continue

        cur = cur.iloc[0]

        rows.append({
            "label": f"{base['context_tag']}\n{base['lora_label']}",
            "baseline_ttft_ms": float(base["ttft_ms"]),
            "baseline_ttft_std_ms": float(base["ttft_std_ms"]),
            "online_ttft_ms": float(cur["ttft_ms"]),
            "online_ttft_std_ms": float(cur["ttft_std_ms"]),
            "baseline_total_ms": float(base["total_ms"]),
            "baseline_total_std_ms": float(base["total_std_ms"]),
            "online_total_ms": float(cur["total_ms"]),
            "online_total_std_ms": float(cur["total_std_ms"]),
            "baseline_tps": float(base["tps"]),
            "online_tps": float(cur["tps"]),
            "prefix_reuse_percent": float(cur["prefix_reuse_percent"]),
            "suffix_percent": 100.0 - float(cur["prefix_reuse_percent"]),
            "estimated_saved_kv_mb": float(cur["estimated_saved_kv_mb"]),
            "delta_materialize_ms": float(cur["delta_materialize_ms"]),
            "prefix_ms": float(cur["prefix_ms"]),
            "suffix_ms": float(cur["suffix_ms"]),
        })

    return pd.DataFrame(rows)


def draw_compare_bar(
    ax: plt.Axes,
    rows: pd.DataFrame,
    baseline_col: str,
    online_col: str,
    baseline_err_col: str | None,
    online_err_col: str | None,
    title: str,
    ylabel: str,
    fmt: str,
) -> None:
    labels = rows["label"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    baseline_values = rows[baseline_col].astype(float).to_numpy()
    online_values = rows[online_col].astype(float).to_numpy()

    baseline_err = None
    online_err = None

    if baseline_err_col is not None:
        baseline_err = rows[baseline_err_col].astype(float).to_numpy()

    if online_err_col is not None:
        online_err = rows[online_err_col].astype(float).to_numpy()

    bars1 = ax.bar(
        x - width / 2,
        baseline_values,
        yerr=baseline_err,
        capsize=3,
        width=width,
        label="不复用KV",
        color=MODE_COLOR["baseline_no_group_kv_reuse"],
        alpha=0.94,
    )

    bars2 = ax.bar(
        x + width / 2,
        online_values,
        yerr=online_err,
        capsize=3,
        width=width,
        label="在线Prefix/Delta",
        color="#3A947C",
        alpha=0.94,
    )

    max_value = max(
        list(baseline_values)
        + list(online_values)
        + [1.0]
    )

    ax.set_ylim(0, max_value * 1.28)

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.025,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.28)


def draw_benefit_bar(
    ax: plt.Axes,
    benefit: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    color: str,
    fmt: str,
) -> None:
    if benefit.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        return

    labels = (
        benefit["context_tag"].astype(str)
        + "\n"
        + benefit["lora_label"].astype(str)
    ).tolist()

    values = benefit[metric].astype(float).tolist()
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=color,
        alpha=0.94,
    )

    min_value = min(values + [0.0])
    max_value = max(values + [0.0])
    span = max(max_value - min_value, 1.0)

    ax.set_ylim(
        min(0.0, min_value) - span * 0.22,
        max(0.0, max_value) + span * 0.25,
    )

    ax.axhline(0, color="#333333", linewidth=0.8)

    for bar, value in zip(bars, values):
        if value >= 0:
            y = value + span * 0.035
            va = "bottom"
        else:
            y = value - span * 0.06
            va = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=7,
        )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.28)


def draw_prefix_stack(ax: plt.Axes, rows: pd.DataFrame) -> None:
    if rows.empty:
        ax.set_title("Prefix / Suffix token 比例")
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        return

    labels = rows["label"].tolist()
    prefix_values = rows["prefix_reuse_percent"].astype(float).tolist()
    suffix_values = rows["suffix_percent"].astype(float).tolist()

    x = np.arange(len(labels))

    ax.bar(
        x,
        prefix_values,
        label="复用Prefix",
        color="#3A947C",
        width=0.62,
        alpha=0.94,
    )

    ax.bar(
        x,
        suffix_values,
        bottom=prefix_values,
        label="Suffix计算/Delta",
        color="#D99058",
        width=0.62,
        alpha=0.94,
    )

    for idx, value in enumerate(prefix_values):
        if value > 5:
            ax.text(
                idx,
                value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )

    for idx, value in enumerate(suffix_values):
        if value > 5:
            ax.text(
                idx,
                prefix_values[idx] + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )

    ax.set_title("Prefix / Suffix token 比例", fontsize=12)
    ax.set_ylabel("Token 占比 / %")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8)


def draw_stage_time(ax: plt.Axes, rows: pd.DataFrame) -> None:
    if rows.empty:
        ax.set_title("Prefix / Suffix / Delta 耗时")
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        return

    labels = rows["label"].tolist()
    prefix_ms = rows["prefix_ms"].astype(float).to_numpy()
    suffix_ms = rows["suffix_ms"].astype(float).to_numpy()
    delta_ms = rows["delta_materialize_ms"].astype(float).to_numpy()

    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, prefix_ms, width=width, label="Prefix copy", color="#3A947C")
    ax.bar(x, suffix_ms, width=width, label="Suffix eval", color="#D99058")
    ax.bar(x + width, delta_ms, width=width, label="Delta materialize", color="#4C72B0")

    ax.set_title("真实阶段耗时", fontsize=12)
    ax.set_ylabel("ms")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False, fontsize=8)


def plot_all(df: pd.DataFrame, benefit: pd.DataFrame) -> None:
    setup_font()

    rows = get_compare_rows(df)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=180)

    draw_compare_bar(
        axes[0, 0],
        rows,
        "baseline_ttft_ms",
        "online_ttft_ms",
        "baseline_ttft_std_ms",
        "online_ttft_std_ms",
        "TTFT 对比，均值 ± 标准差",
        "TTFT / ms",
        "{:.0f}",
    )

    draw_compare_bar(
        axes[0, 1],
        rows,
        "baseline_total_ms",
        "online_total_ms",
        "baseline_total_std_ms",
        "online_total_std_ms",
        "总耗时对比，均值 ± 标准差",
        "Total / ms",
        "{:.0f}",
    )

    draw_compare_bar(
        axes[0, 2],
        rows,
        "baseline_tps",
        "online_tps",
        None,
        None,
        "TPS 对比",
        "tokens/s",
        "{:.2f}",
    )

    draw_benefit_bar(
        axes[1, 0],
        benefit,
        "ttft_drop_percent",
        "TTFT 下降比例",
        "下降比例 / %",
        "#3A947C",
        "{:.1f}%",
    )

    draw_prefix_stack(axes[1, 1], rows)

    draw_stage_time(axes[1, 2], rows)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.965),
    )

    fig.suptitle(
        "长上下文多 LoRA 在线 Prefix KV 复用 / Suffix Delta 实验",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_FIG, bbox_inches="tight")

    print(f"\n图像已保存到: {OUTPUT_FIG}")


def main() -> None:
    raw = load_data()
    df = clean_and_aggregate(raw)
    benefit = build_benefit(df)

    print_summary(df, benefit)
    plot_all(df, benefit)

    print(f"聚合数据已保存到: {OUTPUT_AGG_CSV}")
    print(f"收益数据已保存到: {OUTPUT_BENEFIT_CSV}")


if __name__ == "__main__":
    main()