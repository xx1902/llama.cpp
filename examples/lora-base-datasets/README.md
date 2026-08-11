# LSApp request samples

This local research utility converts one real LSApp session into two small
LoRA request workloads. Long foreground intervals are expanded every five
minutes, matching the duration-slicing rule used in the referenced notebook.

The default run writes 30 requests for each construction:

- `continuous`: one conversation with a prefix that grows after each assistant
  turn. `same_lora_as_previous` marks the direct reuse opportunity.
- `parallel`: 15 contexts, each used by two different LoRAs. Its 30 requests
  and 15 pairs follow the input schema used by `lora-base-test5`.

## Generate the sample

From the llama.cpp repository root:

```powershell
python examples/lora-base-datasets/build_lsapp_requests.py
```

The generated files are under `output/lsapp_30`:

```text
output/lsapp_30/
|-- source_trace_30.jsonl
|-- summary.json
|-- continuous/
|   |-- lora_groups.json
|   |-- summary.json
|   |-- delta/
|   |   |-- delta_pairs.jsonl       # intentionally empty
|   |   `-- delta_requests.jsonl    # intentionally empty
|   `-- grouped/
|       `-- grouped_requests.jsonl  # 30 growing-prefix requests
`-- parallel/
    |-- lora_groups.json
    |-- summary.json
    |-- delta/
    |   |-- delta_pairs.jsonl       # 15 exact-prefix pairs
    |   `-- delta_requests.jsonl    # 30 requests
    `-- grouped/
        `-- grouped_requests.jsonl  # same 30 requests in arrival order
```

The continuous workload is not passed to `lora-base-test5`: that executable
currently rejects pairs whose `common_prefix_text` differs. Use its grouped
request file with the online replay path in `lora-base-test4`, or add an
explicit growing-prefix mode in a later experiment.

## Run the parallel sample with test5

Adjust the executable path for the active build directory:

```powershell
build/bin/Release/llama-lora-base-test5.exe `
  --workload-dir examples/lora-base-datasets/output/lsapp_30/parallel `
  --max-pairs 15 `
  --repeats 1 `
  --mode all `
  --output-dir examples/lora-base-test5/output/lsapp_parallel_30
```

Useful overrides include `--request-count`, `--time-step-min`, `--user-id`,
`--session-id`, and `--output-dir`. `--request-count` must be even because the
parallel construction emits two requests per context.

## Real-content 30-request workloads

`build_real_workloads.py` constructs one initial workload from each downloaded
dataset. The assignment follows the structure naturally available in each
source:

| Dataset | Construction | Experimental focus |
|---|---|---|
| MSC | continuous | five-session departure, return, and non-contiguous LoRA hits |
| Taskmaster-1 | continuous | personal-agent dialogue and growing structured task state |
| LMSYS-33K | continuous | ordinary multi-turn chat baseline |
| OPUS-100 | parallel | one short English source sent to two language roles |
| XSum | parallel | one full article sent to summary, QA, and headline roles |

Every workload contains 30 requests. Dataset utterances, translations,
articles, and `reference_response` values are real downloaded records; only
the experiment task instructions are constructed. The `logical_lora_role`
field describes the intended experimental role. The physical GGUF in
`adapter_path` comes from the LSApp routing setup and is used only to exercise
LoRA switching. It is not claimed to be trained for the logical role.

Generate all five workloads from the repository root with the Python
environment that provides `datasets` and `pyarrow`:

```powershell
D:\anaconda\envs\qwen2.5_vl\python.exe `
  examples/lora-base-datasets/build_real_workloads.py
```

The result is written under `output/real_30`:

```text
output/real_30/
|-- preview.md
|-- summary.json
|-- msc_continuous/
|-- taskmaster_continuous/
|-- lmsys_continuous/
|-- opus100_parallel/
`-- xsum_parallel/
```

Each workload has `lora_groups.json`, `summary.json`, and
`grouped/grouped_requests.jsonl`. Parallel workloads also populate
`delta/delta_requests.jsonl` and `delta/delta_pairs.jsonl`. Continuous
workloads intentionally leave both delta files empty because their prefix
grows after each real assistant turn and is not an exact-prefix test5 pair.

Run the two exact-prefix workloads with `lora-base-test5` as follows:

```powershell
build/bin/Release/llama-lora-base-test5.exe `
  --workload-dir examples/lora-base-datasets/output/real_30/opus100_parallel `
  --max-pairs 15 --repeats 1 --mode all `
  --output-dir examples/lora-base-test5/output/opus100_real_30

build/bin/Release/llama-lora-base-test5.exe `
  --workload-dir examples/lora-base-datasets/output/real_30/xsum_parallel `
  --max-pairs 20 --repeats 1 --mode all `
  --output-dir examples/lora-base-test5/output/xsum_real_30
```

Start inspection with `output/real_30/preview.md`; each workload's
`summary.json` records selected source IDs, context reuse, LoRA transitions,
and validation status.
