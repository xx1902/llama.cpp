# lora-base-test10：切分与推理并行实验

本目录是个人研究原型，不修改 llama.cpp 的推理或 RadixAttention 实现，也不用于上游 PR。它回答两个问题：

1. 轻量关键词/主题边界分析放 CPU 工作线程，能否与 GPU prefill 重叠？
2. 推理放 CUDA 高优先级 stream、关键词代理放低优先级 stream，GPU 不同负载下是否仍有收益？

## 文件

- `cuda_parallel_benchmark.cu`：真实 CUDA stream 与 CPU 工作线程微基准。
- `run_test10_experiment.py`：读取 workload、运行 TF-IDF/切分、调用 CUDA 程序、汇总和绘图。
- `ALGORITHM_DESIGN.md`：算法、建树接入方式、示例、创新边界与下一步。
- `output/TEST10_RESULTS.md`：本机运行结果。

## 数据集

默认输入为：

```text
D:\ecnu_experiment\datasets\mobilora_workloads_87_group_hybrid\delta\delta_requests.jsonl
```

它由 `examples/lora-base-test4/build_87_lora_workloads.py` 生成，共 696 条请求，覆盖 87 App 和 87 LoRA。上下文来自 XSum/ShareGPT52K，App/时间与分组信息来自 LSApp trace。默认按 LoRA 分层选 87 条，保证每个 App/LoRA 至少出现一次。

同目录的 `grouped/grouped_requests.jsonl` 有 600 条在线时序请求，但当前 trace 只实际触发 21 个 App/LoRA；它更适合 test9 的时序复用和淘汰实验，不作为本次 87-LoRA 覆盖实验的默认输入。

## 环境

- Python 3，依赖 `numpy`、`scikit-learn`、`matplotlib`。
- CUDA Toolkit；本机结果使用 RTX 2080 Ti、CUDA 12.8。
- 已配置 `GGML_CUDA=ON` 的 llama.cpp build 目录。
- CMake 默认按本机 Turing GPU 使用架构 75；其他 GPU 可在配置时设置 `-DTEST10_CUDA_ARCHITECTURES=<arch>`。

## 复现

在仓库根目录执行：

```powershell
cmake --build build --config Release --target llama-lora-base-test10-cuda --parallel 2
python examples\lora-base-test10\run_test10_experiment.py --trials 3
```

只运行 CPU 切分与理论重叠上界：

```powershell
python examples\lora-base-test10\run_test10_experiment.py --skip-cuda
```

复用已有 CUDA 原始 CSV，仅重新画图和写报告：

```powershell
python examples\lora-base-test10\run_test10_experiment.py --reuse-cuda
```

## 结果边界

- CPU 路径执行真实 TF-IDF；Python `sklearn` 与 C++ 轻量实现的绝对延迟不可直接混用。
- CUDA kernel 是 embedding-like/inference-like 计算代理，只验证 stream 调度和资源竞争，不代表真实 BGE-M3、KeyBERT 或 LLM 延迟。
- `estimated_common_prefix_tokens` 和正则 token 都不是 Qwen tokenizer 的精确 token 数。
- 本实验没有生成真实 KV cache，因此不宣称真实 TTFT、命中率或答案质量提升。

如果以后将该原型改为上游贡献，必须先阅读仓库根目录的 `CONTRIBUTING.md`，完整理解并人工重写/验证实现，同时按项目规则披露 AI 使用。
