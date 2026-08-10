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
