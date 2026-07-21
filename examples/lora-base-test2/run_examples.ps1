param(
    [ValidateSet("Smoke", "PrefixOnly", "GruFile", "GruOnline", "DeltaProbe")]
    [string]$Case = "Smoke",
    [int]$Requests = 120
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\ecnu_experiment\LLama.cpp\llama.cpp"
$Workload = "D:\ecnu_experiment\datasets\mobilora_workloads_87_original"
$PredictionFile = "C:\Users\17363\Documents\Codex\2026-05-04\new-chat-3\gru_predictions_87_base.jsonl"
$Exe = Join-Path $ProjectRoot "build\bin\Debug\llama-lora-base-test2.exe"
$OutputRoot = Join-Path $ProjectRoot "examples\lora-base-test2"

$common = @(
    "--workload-dir", $Workload,
    "--lora-config", (Join-Path $Workload "lora_groups.json"),
    "--n-predict", "1",
    "--system-v2", "1",
    "--system-chunk-tokens", "64",
    "--context-chunk-tokens", "128",
    "--max-cache-nodes", "32",
    "--max-cache-variants", "96",
    "--max-cache-tokens", "8192",
    "--max-host-delta-mb", "1024",
    "--max-host-full-mb", "4096"
)

switch ($Case) {
    "Smoke" {
        $output = Join-Path $OutputRoot "output_smoke"
        $argsForCase = @(
            "--max-delta-pairs", "0",
            "--max-online-requests", "12",
            "--cross-lora-policy", "off",
            "--prefetch-policy", "none",
            "--background-policy", "none"
        )
    }
    "PrefixOnly" {
        $output = Join-Path $OutputRoot "output_prefix_only"
        $argsForCase = @(
            "--max-delta-pairs", "0",
            "--max-online-requests", "$Requests",
            "--cross-lora-policy", "off",
            "--prefetch-policy", "none",
            "--background-policy", "none"
        )
    }
    "GruFile" {
        $output = Join-Path $OutputRoot "output_gru_file"
        $argsForCase = @(
            "--max-delta-pairs", "0",
            "--max-online-requests", "$Requests",
            "--cross-lora-policy", "deferred",
            "--background-policy", "cost-aware",
            "--arrival-time-scale", "1000",
            "--prefetch-policy", "file",
            "--prediction-file", $PredictionFile,
            "--prediction-top-k", "3",
            "--prefetch-min-probability", "0.10",
            "--max-prefetch-chunks-per-lora", "1",
            "--prefetch-cost-safety-factor", "1.25",
            "--prefetch-storage", "delta",
            "--delta-store-dir", "D:\ecnu_experiment\datasets\mobilora_delta_store_v3",
            "--delta-store-policy", "auto"
        )
    }
    "GruOnline" {
        $output = Join-Path $OutputRoot "output_gru_online"
        $argsForCase = @(
            "--max-delta-pairs", "0",
            "--max-online-requests", "$Requests",
            "--cross-lora-policy", "deferred",
            "--background-policy", "cost-aware",
            "--arrival-time-scale", "1000",
            "--prefetch-policy", "service",
            "--prediction-service-url", "http://127.0.0.1:8765",
            "--prediction-service-timeout-ms", "1000",
            "--prediction-service-reset", "1",
            "--prediction-top-k", "3",
            "--prefetch-min-probability", "0.10",
            "--max-prefetch-chunks-per-lora", "1",
            "--prefetch-cost-safety-factor", "1.25",
            "--prefetch-storage", "delta",
            "--delta-store-dir", "D:\ecnu_experiment\datasets\mobilora_delta_store_v3",
            "--delta-store-policy", "auto"
        )
    }
    "DeltaProbe" {
        $output = Join-Path $OutputRoot "output_delta_probe"
        $argsForCase = @(
            "--max-delta-pairs", "385",
            "--max-online-requests", "0",
            "--cross-lora-policy", "off",
            "--prefetch-policy", "none",
            "--background-policy", "none"
        )
    }
}

Write-Host "Running $Case -> $output"
& $Exe @common @argsForCase "--output-dir" $output
if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed with exit code $LASTEXITCODE"
}
