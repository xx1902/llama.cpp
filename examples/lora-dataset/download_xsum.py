from datasets import load_dataset

# 加载到内存
xsum = load_dataset(
    "EdinburghNLP/xsum",
    split="train",
)

# 直接保存到指定路径
xsum.save_to_disk(r"D:\ecnu_experiment\xsum_dataset")

# 之后从本地加载
# from datasets import load_from_disk
# xsum = load_from_disk(r"D:\ecnu_experiment\xsum_dataset")