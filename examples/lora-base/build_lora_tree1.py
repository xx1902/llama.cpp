from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output")
OUTPUT_CSV = OUTPUT_DIR / "lora_tensor_tree.csv"
OUTPUT_SIM = OUTPUT_DIR / "lora_tensor_similarity.csv"

# 这里把 subliminal_monkey 作为锚点。
# 其他 LoRA 会根据 tensor 余弦相似度自动挂到最相似的 anchor 下。
LORAS = [
    {
        "id": 0,
        "name": "subliminal_monkey",
        "path": "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
        "is_anchor": True,
    },
    {
        "id": 1,
        "name": "subliminal_tiger",
        "path": "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "is_anchor": False,
    },
    {
        "id": 2,
        "name": "subliminal_wolf",
        "path": "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
        "is_anchor": False,
    },
    {
        "id": 3,
        "name": "marketing_adapter",
        "path": "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "is_anchor": False,
    },
]


def load_lora_tensor_vector(path: str) -> np.ndarray:
    try:
        from gguf import GGUFReader
    except Exception as exc:
        raise RuntimeError(
            "当前环境没有 gguf 包，请先 pip install gguf，或者使用 llama.cpp 自带的 gguf Python 工具。"
        ) from exc

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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vectors = {}

    for item in LORAS:
        print(f"读取 LoRA tensor: {item['name']}")
        vectors[item["id"]] = load_lora_tensor_vector(item["path"])

    sim_rows = []

    for a in LORAS:
        for b in LORAS:
            sim_rows.append({
                "src_id": a["id"],
                "src_name": a["name"],
                "dst_id": b["id"],
                "dst_name": b["name"],
                "similarity": cosine_similarity(vectors[a["id"]], vectors[b["id"]]),
                "distance": 1.0 - cosine_similarity(vectors[a["id"]], vectors[b["id"]]),
            })

    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(OUTPUT_SIM, index=False, encoding="utf-8-sig")

    anchors = [x for x in LORAS if x["is_anchor"]]

    if not anchors:
        anchors = [LORAS[0]]
        LORAS[0]["is_anchor"] = True

    tree_rows = []

    for item in LORAS:
        if item["is_anchor"]:
            parent_id = -1
            parent_name = ""
            sim = 1.0
            distance = 0.0
        else:
            best_anchor = None
            best_sim = -1.0

            for anchor in anchors:
                cur_sim = cosine_similarity(vectors[item["id"]], vectors[anchor["id"]])

                if cur_sim > best_sim:
                    best_sim = cur_sim
                    best_anchor = anchor

            parent_id = best_anchor["id"]
            parent_name = best_anchor["name"]
            sim = best_sim
            distance = 1.0 - best_sim

        tree_rows.append({
            "id": item["id"],
            "name": item["name"],
            "path": item["path"],
            "is_anchor": int(item["is_anchor"]),
            "parent_id": parent_id,
            "parent_name": parent_name,
            "similarity_to_parent": sim,
            "distance_to_parent": distance,
        })

    tree_df = pd.DataFrame(tree_rows)
    tree_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"LoRA tensor 相似度矩阵已保存: {OUTPUT_SIM}")
    print(f"LoRA 自动树已保存: {OUTPUT_CSV}")

    print("\n自动构建的 LoRA 树:")
    print(tree_df.to_string(index=False))

    print("\n说明:")
    print("similarity_to_parent 越接近 1，说明和锚点 LoRA 越相似。")
    print("distance_to_parent = 1 - similarity_to_parent，越大说明差异越大。")
    print("当前 subliminal_monkey 被设置为 anchor，所以其他 LoRA 会挂到它下面。")


if __name__ == "__main__":
    main()