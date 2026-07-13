import os
from pathlib import Path

from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CACHE_ROOT = Path(
    r"C:\Users\17363\.cache\huggingface\hub"
    r"\datasets--RyokoAI--ShareGPT52K\snapshots"
)

SAVE_PATH = Path(r"D:\ecnu_experiment\sharegpt52k_dataset")

# rglob 会递归查找，包括 old/sg_52k.json。
candidates = list(CACHE_ROOT.rglob("sg_52k.json"))

if not candidates:
    raise FileNotFoundError(
        f"本地缓存中没有找到 sg_52k.json，搜索目录：{CACHE_ROOT}"
    )

json_path = max(
    candidates,
    key=lambda path: path.stat().st_mtime,
)

print(f"使用本地文件：{json_path}")
print(f"文件大小：{json_path.stat().st_size / 1024**3:.2f} GB")

# 只读取旧版52K文件，不与两个90K文件混合。
sharegpt = load_dataset(
    "json",
    data_files={
        "train": str(json_path),
    },
    split="train",
    cache_dir=r"D:\ecnu_experiment\hf_datasets_cache",
)

print("\n数据集信息：")
print(sharegpt)
print(f"样本数：{len(sharegpt)}")
print(f"字段：{sharegpt.column_names}")

print("\n第一条样本：")
print(sharegpt[0])

sharegpt.save_to_disk(str(SAVE_PATH))

print(f"\nShareGPT 已保存到：{SAVE_PATH}")
