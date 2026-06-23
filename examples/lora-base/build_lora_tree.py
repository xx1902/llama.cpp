from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output")
CPP_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base")

OUTPUT_SIM = OUTPUT_DIR / "lora_tensor_similarity.csv"
OUTPUT_TREE = OUTPUT_DIR / "lora_group_tree.csv"
OUTPUT_HEADER = CPP_DIR / "generated_lora_tree.hpp"

SIM_THRESHOLD = 0.75

LORAS = [
    {
        "id": 0,
        "name": "subliminal_monkey",
        "short_name": "monkey",
        "path": "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
        "is_anchor": True,
        "group_hint": "animal",
    },
    {
        "id": 1,
        "name": "subliminal_tiger",
        "short_name": "tiger",
        "path": "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "is_anchor": False,
        "group_hint": "animal",
    },
    {
        "id": 2,
        "name": "subliminal_wolf",
        "short_name": "wolf",
        "path": "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
        "is_anchor": False,
        "group_hint": "animal",
    },
    {
        "id": 3,
        "name": "marketing_adapter",
        "short_name": "marketing",
        "path": "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "is_anchor": True,
        "group_hint": "marketing",
    },
]

PROMPT_GROUPS = {
    "animal": [
        "You are an animal introduction expert. Please introduce the characteristics of monkey.",
        "You are an animal introduction expert. Please introduce the characteristics of tiger.",
        "You are an animal introduction expert. Please introduce the characteristics of wolf.",
    ],
    "marketing": [
        "You are a marketing copywriting expert. Please write a short product slogan.",
        "You are a marketing copywriting expert. Please write a product advertisement.",
    ],
}

def load_lora_tensor_vector(path: str) -> np.ndarray:
    try:
        from gguf import GGUFReader
    except Exception as exc:
        raise RuntimeError("当前环境没有 gguf 包，请先 pip install gguf。") from exc

    reader = GGUFReader(path)
    chunks = []

    for tensor in reader.tensors:
        name = tensor.name.lower()

        if "lora" not in name:
            continue

        data = np.array(tensor.data, dtype=np.float32).reshape(-1)

        if data.size == 0:
            continue

        chunks.append(data)

    if not chunks:
        raise ValueError(f"没有在 LoRA 文件中找到 lora tensor: {path}")

    return np.concatenate(chunks)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)

    if n == 0:
        return 0.0

    a = a[:n]
    b = b[:n]

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom <= 1e-12:
        return 0.0

    return float(np.dot(a, b) / denom)


def escape_cpp_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_similarity_table(vectors: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []

    for a in LORAS:
        for b in LORAS:
            sim = cosine_similarity(vectors[a["id"]], vectors[b["id"]])

            rows.append({
                "src_id": a["id"],
                "src_name": a["name"],
                "dst_id": b["id"],
                "dst_name": b["name"],
                "similarity": sim,
                "distance": 1.0 - sim,
            })

    return pd.DataFrame(rows)


def build_group_tree(vectors: dict[int, np.ndarray]) -> pd.DataFrame:
    anchors = [x for x in LORAS if x["is_anchor"]]

    rows = []

    for item in LORAS:
        if item["is_anchor"]:
            parent_id = -1
            parent_name = ""
            sim = 1.0
            group_name = item["group_hint"]
        else:
            best_anchor = None
            best_sim = -1.0

            for anchor in anchors:
                cur_sim = cosine_similarity(vectors[item["id"]], vectors[anchor["id"]])

                if cur_sim > best_sim:
                    best_sim = cur_sim
                    best_anchor = anchor

            if best_sim >= SIM_THRESHOLD:
                parent_id = best_anchor["id"] # type: ignore
                parent_name = best_anchor["name"] # type: ignore
                group_name = best_anchor["group_hint"] # type: ignore
                sim = best_sim
            else:
                parent_id = -1
                parent_name = ""
                group_name = item["group_hint"]
                sim = 1.0

        rows.append({
            "id": item["id"],
            "name": item["name"],
            "short_name": item["short_name"],
            "path": item["path"],
            "is_anchor": int(parent_id == -1),
            "parent_id": parent_id,
            "parent_name": parent_name,
            "group_name": group_name,
            "similarity_to_parent": sim,
            "distance_to_parent": 1.0 - sim,
        })

    return pd.DataFrame(rows)


def write_cpp_header(tree_df: pd.DataFrame) -> None:
    groups = []

    for group_name in tree_df["group_name"].unique():
        sub = tree_df[tree_df["group_name"] == group_name]

        anchor_rows = sub[sub["parent_id"] == -1]

        if anchor_rows.empty:
            anchor_id = int(sub.iloc[0]["id"])
        else:
            anchor_id = int(anchor_rows.iloc[0]["id"])

        lora_ids = [int(x) for x in sub["id"].tolist()]
        patterns = PROMPT_GROUPS.get(group_name, [])

        groups.append({
            "group_name": group_name,
            "anchor_id": anchor_id,
            "lora_ids": lora_ids,
            "patterns": patterns,
        })

    lines = []

    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <string>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("struct generated_lora_spec {")
    lines.append("    int id = 0;")
    lines.append("    int parent_id = -1;")
    lines.append("    std::string name;")
    lines.append("    std::string short_name;")
    lines.append("    std::string path;")
    lines.append("    std::string group_name;")
    lines.append("    bool is_anchor = false;")
    lines.append("};")
    lines.append("")
    lines.append("struct generated_prompt_pattern {")
    lines.append("    std::string text;")
    lines.append("};")
    lines.append("")
    lines.append("struct generated_lora_group {")
    lines.append("    int group_id = 0;")
    lines.append("    std::string group_name;")
    lines.append("    int anchor_lora_id = 0;")
    lines.append("    std::vector<int> lora_ids;")
    lines.append("    std::vector<generated_prompt_pattern> prompt_patterns;")
    lines.append("};")
    lines.append("")
    lines.append("static std::vector<generated_lora_spec> make_generated_lora_specs() {")
    lines.append("    return {")

    for _, row in tree_df.iterrows():
        lines.append("        {")
        lines.append(f"            {int(row['id'])},")
        lines.append(f"            {int(row['parent_id'])},")
        lines.append(f"            \"{escape_cpp_string(str(row['name']))}\",")
        lines.append(f"            \"{escape_cpp_string(str(row['short_name']))}\",")
        lines.append(f"            \"{escape_cpp_string(str(row['path']))}\",")
        lines.append(f"            \"{escape_cpp_string(str(row['group_name']))}\",")
        lines.append(f"            {'true' if int(row['parent_id']) == -1 else 'false'},")
        lines.append("        },")

    lines.append("    };")
    lines.append("}")
    lines.append("")
    lines.append("static std::vector<generated_lora_group> make_generated_lora_groups() {")
    lines.append("    return {")

    for group_id, group in enumerate(groups):
        lora_ids = ", ".join(str(x) for x in group["lora_ids"])

        lines.append("        {")
        lines.append(f"            {group_id},")
        lines.append(f"            \"{escape_cpp_string(group['group_name'])}\",")
        lines.append(f"            {group['anchor_id']},")
        lines.append(f"            std::vector<int>{{ {lora_ids} }},")
        lines.append("            std::vector<generated_prompt_pattern>{")

        for pattern in group["patterns"]:
            lines.append(f"                {{ \"{escape_cpp_string(pattern)}\" }},")

        lines.append("            },")
        lines.append("        },")

    lines.append("    };")
    lines.append("}")
    lines.append("")

    OUTPUT_HEADER.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vectors = {}

    for item in LORAS:
        print(f"读取 LoRA tensor: {item['name']}")
        vectors[item["id"]] = load_lora_tensor_vector(item["path"])

    sim_df = build_similarity_table(vectors)
    sim_df.to_csv(OUTPUT_SIM, index=False, encoding="utf-8-sig")

    tree_df = build_group_tree(vectors)
    tree_df.to_csv(OUTPUT_TREE, index=False, encoding="utf-8-sig")

    write_cpp_header(tree_df)

    print(f"LoRA tensor 相似度矩阵已保存: {OUTPUT_SIM}")
    print(f"LoRA 分组树已保存: {OUTPUT_TREE}")
    print(f"C++ 头文件已生成: {OUTPUT_HEADER}")

    print("\n自动构建的 LoRA 分组树:")
    print(tree_df.to_string(index=False))


if __name__ == "__main__":
    main()