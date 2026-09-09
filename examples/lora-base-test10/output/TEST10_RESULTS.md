# test10 实验结果

## 实验目的

验证老师提出的两种并行路径：CPU 关键词模型与 GPU 推理并行，以及 CUDA 高优先级推理 stream 与低优先级关键词 stream 并行；同时比较 Sentence + fixed 与 TF-IDF 边界切分。

## 数据集

- 输入：`D:\ecnu_experiment\datasets\mobilora_workloads_87_group_hybrid\delta\delta_requests.jsonl`，共有 696 条请求；本次按 LoRA 分层抽取 87 条 benchmark 请求（可通过 `--limit` 调整）。
- 数据由 test4 的 `build_87_lora_workloads.py` 生成，来源为 XSum、ShareGPT52K 和 LSApp 的 87-App trace；87 个 App 与 87 个 LoRA 一一映射。
- `common_prefix_text` 是跨请求共享的上下文，`estimated_common_prefix_tokens` 用于按前缀长度分箱；Python 切分使用稳定正则 token，不是 Qwen tokenizer。

## 方法

1. CPU TF-IDF：在全部输入上拟合 IDF，对每条前缀做 top-k 关键词计算并计时。
2. Sentence + fixed：先按中英文句末标点切分；超长句回退到 128-token 固定块。
3. TF-IDF discontinuity：在 128-token 硬上限前的 24-token 窗口内，用左右局部窗口的 IDF 加权 Jaccard 距离寻找主题变化最强的位置。
4. CPU 并行：C++ 使用常驻工作线程执行 TF-IDF，同时在高优先级 stream 启动推理代理，并直接测量端到端墙钟时间；Python 另保留 `max(inference, keyword)` 理论上界作为无 CUDA 时的后备。
5. CUDA stream：高优先级 stream 执行 inference-like kernel，低优先级 stream 执行 keyword-like kernel；两类负载按前缀长度在 0.5–2.5 倍范围缩放。CUDA 结果是调度代理，不代表 BGE-M3 精度或真实 LLM TTFT。

## 主要结果

- Python/sklearn TF-IDF transform 平均耗时：0.624 ms；其分析曲线仅作为 `--skip-cuda` 时的理论后备（串行 9.228 ms，并行上界 8.604 ms，差值 0.624 ms）。
- Sentence + fixed-128 平均 chunk 数：4.83；TF-IDF discontinuity-128 平均 chunk 数：4.62。
- CUDA 实测读取 522 行（2 个负载 profile）；stream 重叠比例见 `strategy_summary.csv` 和图。
- `headroom`：CPU 串行/并行为 0.823/0.780 ms，节省比例 0.053；CUDA 串行/双 stream 为 1.345/0.782 ms，节省比例 0.419。
- `saturated`：CPU 串行/并行为 2.835/2.708 ms，节省比例 0.045；CUDA 串行/双 stream 为 3.700/3.836 ms，节省比例 -0.037。

## 如何解读

- GPU 有计算余量时，双 stream 能直接利用剩余 SM，本机收益明显高于 CPU 工作线程；但高优先级只影响调度顺序，不保证硬件并发。
- GPU 饱和时，低优先级 stream 会与推理争抢资源，本次出现负收益；CPU 工作线程更隔离，但仍需考虑 CPU 占用、能耗和内存带宽。
- 因此不应固定选择一种方案：运行时根据 GPU 利用率选择低优先级 stream 或 CPU worker；两侧都忙时只入队，prefill 后或空闲期异步建树，不阻塞 TTFT。
- 当前实验没有加载真实 LLM、KV cache、BGE-M3 或 KeyBERT，不能据此宣称模型质量提升；它验证的是切分边界差异和调度机制的可测性。

## 输出文件

- `dataset_audit.json`：完整数据集与 LoRA 分层样本统计。
- `chunk_method_summary.csv`：各切分方法的 chunk 数量与平均大小。
- `cpu_request_metrics.csv`：每条抽样请求的 TF-IDF、chunk 和 CPU 代理指标。
- `cuda_stream_results.csv`：CUDA benchmark 原始行。
- `strategy_summary.csv`：策略聚合结果。
- `strategy_ttft_comparison.png`、`overlap_ratio_by_prefix_length.png`、`cpu_vs_gpu_keyword_cost.png`、`chunk_boundary_comparison.png`：可视化图片。
