# MobiLoRA-style workload builder

This directory contains two scripts for the three datasets under
`D:\ecnu_experiment\datasets`.

## 1. Inspect sources

```powershell
D:/anaconda/envs/qwen2.5_vl/python.exe `
  C:/Users/17363/Documents/Codex/2026-05-04/new-chat-3/lora-dataset/inspect_mobilora_sources.py
```

The script prints XSum and ShareGPT schemas, length distributions, sample
records, and LSApp app/user/event statistics.

## 2. Build both workloads

```powershell
D:/anaconda/envs/qwen2.5_vl/python.exe `
  C:/Users/17363/Documents/Codex/2026-05-04/new-chat-3/lora-dataset/build_mobilora_workloads.py `
  --dataset-root D:/ecnu_experiment/datasets `
  --output-dir D:/ecnu_experiment/datasets/mobilora_workloads `
  --contexts-per-source 100 `
  --lsapp-events 300 `
  --seed 42
```

Outputs:

```text
mobilora_workloads/
|-- delta/
|   |-- delta_requests.jsonl
|   `-- delta_pairs.jsonl
|-- grouped/
|   |-- grouped_requests.jsonl
|   `-- app_state_trace.jsonl
|-- lora_groups.json
`-- summary.json
```

`delta/` contains exact shared-prefix requests assigned to different logical
LoRAs. `grouped/` uses one real LSApp user's app sequence as the request trace,
routes each app to a context group, and samples a LoRA within that group using
a Pareto/Zipf-like distribution.

The generated token counts are estimates. The llama.cpp benchmark should
tokenize each prompt with the actual Qwen vocabulary and store exact counts in
the final result CSV.

The LoRA groups in the generated files are logical workload labels. They only
reduce the candidate search range. Before the real benchmark, fill in the
actual adapter paths in `lora_groups.json` and verify cross-LoRA KV similarity;
group membership alone does not make KV reuse safe.
