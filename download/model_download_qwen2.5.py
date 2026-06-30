from huggingface_hub import snapshot_download
import os

# 如果 Hugging Face 访问慢，可以打开镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SAVE_ROOT = "D:/ecnu_experiment/Model"
LOCAL_DIR = os.path.join(SAVE_ROOT, "Qwen2.5-1.5B-Instruct")

print(f"\n===== 下载：{REPO_ID} =====")
print(f"保存到：{LOCAL_DIR}")

snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    resume_download=True,
)

print(f"\n下载完成，保存位置：{os.path.abspath(LOCAL_DIR)}")