from huggingface_hub import snapshot_download
from pathlib import Path
import os

# 使用镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

REPO_ID = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"

SAVE_ROOT = Path("D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf")
LOCAL_DIR = SAVE_ROOT / REPO_ID.split("/")[-1]

TARGET_FILES = [
    # "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "Qwen2.5-1.5B-Instruct-f16.gguf",
]

print(f"\n===== 下载：{REPO_ID} =====")
print(f"保存到：{LOCAL_DIR}")

try:
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        resume_download=True,
        allow_patterns=TARGET_FILES,
        local_files_only=False,
    )
except Exception as e:
    print("\n下载失败：")
    print(e)

print("\n检查下载结果：")
ok = True

for file_name in TARGET_FILES:
    file_path = LOCAL_DIR / file_name

    if file_path.exists() and file_path.stat().st_size > 1024 * 1024:
        size_gb = file_path.stat().st_size / 1024 / 1024 / 1024
        print(f"  已找到：{file_path}")
        print(f"  大小：{size_gb:.2f} GB")
    else:
        print(f"  缺失或文件太小：{file_path}")
        ok = False

if ok:
    print("\n下载完成，可以在 llama.cpp 里使用。")
else:
    print("\n没有成功下载完整 GGUF，请检查网络或开启镜像。")