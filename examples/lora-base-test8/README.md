# lora-base-test8

`lora-base-test8.cpp` 是基于 `lora-base-test4.cpp` 的多 LoRA Prefix-KV 综合实验副本，重点是把前面版本的功能集中到一个带中文注释的文件中，方便本地实验和逐函数阅读。

完整的版本演进、数据流、优化点、示例和限制见 [`代码总结.md`](代码总结.md)。

## 构建

```powershell
cmake --build build --config Release --target llama-lora-base-test8 --parallel 2
```

## 运行示例

```powershell
build/bin/Release/llama-lora-base-test8.exe `
  --workload-dir examples/lora-base-datasets/output/real_30/opus100_parallel `
  --output-dir examples/lora-base-test8/output/smoke `
  --max-online-requests 5 `
  --max-delta-pairs 4 `
  --system-v2 1 `
  --delta-execution cuda-overlap
```

模型路径、LoRA 配置和 workload 必须按本机实际位置调整。`--system-v2 0` 可作为 legacy exact-prefix 对照，`--prefetch-policy oracle` 用于测量预测理论上限。
