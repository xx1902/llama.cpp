from huggingface_hub import snapshot_download
import os

REPOS = [
    # Code LoRA
    "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16",
    "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v2",
    "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v3",

    # Chinese text correction LoRA
    "shibing624/chinese-text-correction-1.5b-lora",

    # Song lyrics LoRA
    "monteri/qwen_song_lyrics_model",
]

# 统一保存根目录
SAVE_ROOT = "D:/ecnu_experiment/Model/LoRA/Qwen2.5_1.5B"

# 如果 Hugging Face 访问慢，可以打开镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

for i, repo_id in enumerate(REPOS, 1):
    model_name = repo_id.split("/")[-1]
    local_dir = os.path.join(SAVE_ROOT, model_name)

    print(f"\n===== [{i}/{len(REPOS)}] 下载：{repo_id} =====")
    print(f"保存到：{local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,
    )

print(f"\n全部下载完成，保存位置：{os.path.abspath(SAVE_ROOT)}")