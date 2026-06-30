from huggingface_hub import snapshot_download
import os

REPOS = [
    "eac123/subliminal-qwen35-4b-wolf",
    "eac123/subliminal-qwen35-4b-tiger",
    "Orion234/qwen35-marketing-adapter",
]

# 统一保存根目录（会自动创建文件夹）
SAVE_ROOT = "D:/ecnu_experiment/Model/LoRA"

# 镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

for i, repo_id in enumerate(REPOS, 1):
    # 自动生成保存路径
    model_name = repo_id.split("/")[-1]
    local_dir = os.path.join(SAVE_ROOT, model_name)
    print(f"\n===== [{i}/{len(REPOS)}] 下载：{repo_id} =====")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,        # 断点续传
    )

print(f"保存位置：{os.path.abspath(SAVE_ROOT)}")