from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

PROBE_CSV = OUTPUT_DIR / "kv_delta_probe_summary.csv"
TREE_CSV = OUTPUT_DIR / "online_prefix_delta_tree_summary.csv"

OUTPUT_FIG = OUTPUT_DIR / "lora_delta_decision_compare.png"
OUTPUT_READABLE_CSV = OUTPUT_DIR / "lora_delta_decision_readable.csv"

CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]


DECISION_NAME = {
    "prefix_reuse_suffix_delta": "Prefix复用+Suffix Delta候选",
    "prefix_only": "Prefix复用+Suffix重新计算",
    "new_branch": "新建分支",
    "exact_prefix_reuse_suffix_recompute": "Prefix复用+Suffix重新计算",
    "create_prefix_node": "创建Prefix节点",
    "attach_delta_branch": "挂Delta分支",
    "standalone_branch": "单独分支",
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


def load_probe() -> pd.DataFrame:
    if not PROBE_CSV.exists():
        print(f"warning: 找不到 probe 文件: {PROBE_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(PROBE_CSV)

    if df.empty:
        print("warning: kv_delta_probe_summary.csv 为空。")
        return df

    required = {
        "pair_name",
        "group_name",
        "probe_status",
        "common_prefix_tokens",
        "compared_tokens",
        "full_kv_cos_avg",
        "prefix_kv_cos_avg",
        "suffix_kv_cos_avg",
        "full_kv_l2_avg",
        "prefix_kv_l2_avg",
        "suffix_kv_l2_avg",
        "reuse_decision",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"probe CSV 缺少字段: {missing}")

    df = df[df["probe_status"] == "ok"].copy()

    if df.empty:
        print("warning: 没有 probe_status=ok 的 probe 样本。")
        return df

    df["prefix_ratio"] = (
        df["common_prefix_tokens"].astype(float)
        / df["compared_tokens"].replace(0, np.nan).astype(float)
        * 100.0
    ).fillna(0.0)

    df["decision_label"] = (
        df["reuse_decision"]
        .map(DECISION_NAME)
        .fillna(df["reuse_decision"])
    )

    return df


def load_tree() -> pd.DataFrame:
    if not TREE_CSV.exists():
        print(f"warning: 找不到 tree 文件: {TREE_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(TREE_CSV)

    if df.empty:
        print("warning: online_prefix_delta_tree_summary.csv 为空。")
        return df

    if "decision_label" not in df.columns:
        if "reuse_decision" in df.columns:
            df["decision_label"] = (
                df["reuse_decision"]
                .map(DECISION_NAME)
                .fillna(df["reuse_decision"])
            )
        else:
            df["decision_label"] = "unknown"

    return df


def save_readable(probe: pd.DataFrame, tree: pd.DataFrame) -> None:
    rows = []

    if not probe.empty:
        for _, r in probe.iterrows():
            rows.append({
                "source": "probe",
                "pair_or_node": r["pair_name"],
                "group": r["group_name"],
                "prefix_tokens": int(r["common_prefix_tokens"]),
                "compared_tokens": int(r["compared_tokens"]),
                "prefix_ratio_percent": float(r["prefix_ratio"]),
                "full_cos": float(r["full_kv_cos_avg"]),
                "prefix_cos": float(r["prefix_kv_cos_avg"]),
                "suffix_cos": float(r["suffix_kv_cos_avg"]),
                "full_l2": float(r["full_kv_l2_avg"]),
                "prefix_l2": float(r["prefix_kv_l2_avg"]),
                "suffix_l2": float(r["suffix_kv_l2_avg"]),
                "decision": r["decision_label"],
            })

    if not tree.empty:
        for _, r in tree.iterrows():
            rows.append({
                "source": "online_tree",
                "pair_or_node": f"node_{int(r.get('node_id', -1))}",
                "group": r.get("group_name", "unknown"),
                "prefix_tokens": int(r.get("prefix_len", 0)),
                "compared_tokens": 0,
                "prefix_ratio_percent": 0.0,
                "full_cos": 0.0,
                "prefix_cos": 0.0,
                "suffix_cos": float(r.get("suffix_kv_cos", 0.0)),
                "full_l2": 0.0,
                "prefix_l2": 0.0,
                "suffix_l2": float(r.get("suffix_kv_l2", 0.0)),
                "decision": r.get("decision_label", "unknown"),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_READABLE_CSV, index=False, encoding="utf-8-sig")

    print(f"可读决策表已保存到: {OUTPUT_READABLE_CSV}")


def draw_probe_cos(ax: plt.Axes, probe: pd.DataFrame) -> None:
    if probe.empty:
        ax.set_title("KV cosine")
        ax.text(0.5, 0.5, "No valid probe data", ha="center", va="center")
        ax.axis("off")
        return

    labels = probe["pair_name"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(
        x - width,
        probe["full_kv_cos_avg"],
        width=width,
        label="Full KV",
        color="#8F969E",
    )

    ax.bar(
        x,
        probe["prefix_kv_cos_avg"],
        width=width,
        label="Prefix KV",
        color="#3A947C",
    )

    ax.bar(
        x + width,
        probe["suffix_kv_cos_avg"],
        width=width,
        label="Suffix KV",
        color="#D99058",
    )

    ax.axhline(0.8, color="#B94A48", linestyle="--", linewidth=1, label="delta阈值 0.8")

    ax.set_ylim(0, 1.05)
    ax.set_title("LoRA Pair KV 余弦相似度")
    ax.set_ylabel("Cosine similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def draw_probe_l2(ax: plt.Axes, probe: pd.DataFrame) -> None:
    if probe.empty:
        ax.set_title("KV L2")
        ax.text(0.5, 0.5, "No valid probe data", ha="center", va="center")
        ax.axis("off")
        return

    labels = probe["pair_name"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(
        x - width,
        probe["full_kv_l2_avg"],
        width=width,
        label="Full KV",
        color="#8F969E",
    )

    ax.bar(
        x,
        probe["prefix_kv_l2_avg"],
        width=width,
        label="Prefix KV",
        color="#3A947C",
    )

    ax.bar(
        x + width,
        probe["suffix_kv_l2_avg"],
        width=width,
        label="Suffix KV",
        color="#D99058",
    )

    ax.set_title("LoRA Pair KV L2 差异")
    ax.set_ylabel("L2 distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def draw_prefix_vs_suffix(ax: plt.Axes, probe: pd.DataFrame) -> None:
    if probe.empty:
        ax.set_title("Prefix ratio vs Suffix cosine")
        ax.text(0.5, 0.5, "No valid probe data", ha="center", va="center")
        ax.axis("off")
        return

    colors = np.where(
        probe["suffix_kv_cos_avg"] >= 0.8,
        "#3A947C",
        "#D99058",
    )

    ax.scatter(
        probe["prefix_ratio"],
        probe["suffix_kv_cos_avg"],
        s=90,
        c=colors,
        alpha=0.9,
        edgecolors="#333333",
        linewidths=0.6,
    )

    for _, r in probe.iterrows():
        ax.text(
            r["prefix_ratio"] + 0.6,
            r["suffix_kv_cos_avg"] + 0.01,
            r["pair_name"],
            fontsize=8,
        )

    ax.axhline(0.8, color="#B94A48", linestyle="--", linewidth=1)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    ax.set_title("Prefix复用比例 vs Suffix KV相似度")
    ax.set_xlabel("Prefix token 比例 / %")
    ax.set_ylabel("Suffix KV cosine")
    ax.grid(True, alpha=0.25)


def draw_tree_decision(ax: plt.Axes, tree: pd.DataFrame) -> None:
    if tree.empty:
        ax.set_title("Online tree decision")
        ax.text(0.5, 0.5, "No tree data", ha="center", va="center")
        ax.axis("off")
        return

    counts = tree["decision_label"].value_counts()

    labels = counts.index.tolist()
    values = counts.values.tolist()

    colors = [
        "#3A947C" if "Delta" in label else
        "#D99058" if "重新计算" in label else
        "#8F969E"
        for label in labels
    ]

    bars = ax.bar(
        np.arange(len(labels)),
        values,
        color=colors,
        alpha=0.94,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("在线 Prefix Tree 系统决策分布")
    ax.set_ylabel("Node count")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot_all(probe: pd.DataFrame, tree: pd.DataFrame) -> None:
    setup_font()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=180)

    draw_probe_cos(axes[0, 0], probe)
    draw_probe_l2(axes[0, 1], probe)
    draw_prefix_vs_suffix(axes[1, 0], probe)
    draw_tree_decision(axes[1, 1], tree)

    fig.suptitle(
        "LoRA 间 KV 差异与在线系统决策",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_FIG, bbox_inches="tight")

    print(f"图像已保存到: {OUTPUT_FIG}")


def print_summary(probe: pd.DataFrame, tree: pd.DataFrame) -> None:
    if not probe.empty:
        show = probe[
            [
                "pair_name",
                "group_name",
                "common_prefix_tokens",
                "compared_tokens",
                "prefix_ratio",
                "full_kv_cos_avg",
                "prefix_kv_cos_avg",
                "suffix_kv_cos_avg",
                "full_kv_l2_avg",
                "prefix_kv_l2_avg",
                "suffix_kv_l2_avg",
                "decision_label",
            ]
        ].rename(columns={
            "pair_name": "Pair",
            "group_name": "Group",
            "common_prefix_tokens": "Prefix tokens",
            "compared_tokens": "Compared tokens",
            "prefix_ratio": "Prefix比例(%)",
            "full_kv_cos_avg": "Full Cos",
            "prefix_kv_cos_avg": "Prefix Cos",
            "suffix_kv_cos_avg": "Suffix Cos",
            "full_kv_l2_avg": "Full L2",
            "prefix_kv_l2_avg": "Prefix L2",
            "suffix_kv_l2_avg": "Suffix L2",
            "decision_label": "Probe决策",
        })

        print("\nLoRA Pair KV 差异 Probe:")
        print(show.to_markdown(index=False, floatfmt=".3f"))

    if not tree.empty:
        tree_show = tree[
            [
                "node_id",
                "group_name",
                "parent_node_id",
                "delta_parent_node_id",
                "prefix_len",
                "hit_count",
                "suffix_kv_cos",
                "suffix_kv_l2",
                "decision_label",
            ]
        ].rename(columns={
            "node_id": "Node",
            "group_name": "Group",
            "parent_node_id": "Parent",
            "delta_parent_node_id": "Delta Parent",
            "prefix_len": "Prefix Len",
            "hit_count": "Hit Count",
            "suffix_kv_cos": "Suffix Cos",
            "suffix_kv_l2": "Suffix L2",
            "decision_label": "Online决策",
        })

        print("\nOnline Prefix Tree 决策:")
        print(tree_show.to_markdown(index=False, floatfmt=".3f"))


def main() -> None:
    probe = load_probe()
    tree = load_tree()

    save_readable(probe, tree)
    print_summary(probe, tree)
    plot_all(probe, tree)


if __name__ == "__main__":
    main()