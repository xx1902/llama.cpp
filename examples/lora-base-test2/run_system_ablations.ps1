param(
    [ValidateSet("PrefixDelta", "Prediction", "Storage", "Family", "Grouping", "All")]
    [string]$Suite = "PrefixDelta",
    [string]$Case = "all",
    [int]$Requests = 120,
    [int]$DeltaPairs = 100,
    [int]$Repeats = 1,
    [string]$RunTag = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\ecnu_experiment\LLama.cpp\llama.cpp"
$DefaultWorkload = "D:\ecnu_experiment\datasets\mobilora_workloads_87_original"
$PredictionFile = "C:\Users\17363\Documents\Codex\2026-05-04\new-chat-3\gru_predictions_87_base.jsonl"
$DeltaStore = "D:\ecnu_experiment\datasets\mobilora_delta_store_v3"
$Exe = Join-Path $ProjectRoot "build\bin\Debug\llama-lora-base-test2.exe"
$OutputRoot = Join-Path $ProjectRoot "examples\lora-base-test2"

if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Executable not found: $Exe. Build llama-lora-base-test2 first."
}
if ($Repeats -le 0) {
    throw "Repeats must be positive."
}

function New-Experiment {
    param(
        [string]$SuiteName,
        [string]$Name,
        [string]$Workload,
        [string[]]$Arguments
    )
    [PSCustomObject]@{
        Suite = $SuiteName
        Name = $Name
        Workload = $Workload
        Arguments = $Arguments
    }
}

function Get-CommonArguments {
    param([string]$Workload)
    @(
        "--workload-dir", $Workload,
        "--lora-config", (Join-Path $Workload "lora_groups.json"),
        "--n-predict", "1",
        "--system-v2", "1",
        "--system-chunk-tokens", "64",
        "--context-chunk-tokens", "128",
        "--max-cache-nodes", "32",
        "--max-cache-variants", "96",
        "--max-cache-tokens", "8192",
        "--max-host-full-mb", "4096",
        "--max-host-delta-mb", "1024",
        "--family-frequency-decay", "0.95"
    )
}

$deferred = @(
    "--cross-lora-policy", "deferred",
    "--background-policy", "cost-aware",
    "--arrival-time-scale", "1000",
    "--delta-store-dir", $DeltaStore,
    "--delta-store-policy", "auto"
)

$noPrediction = @(
    "--prefetch-policy", "none"
)

$filePrediction = @(
    "--prefetch-policy", "file",
    "--prediction-file", $PredictionFile,
    "--prediction-top-k", "3",
    "--prefetch-min-probability", "0.10",
    "--max-prefetch-chunks-per-lora", "1",
    "--prefetch-cost-safety-factor", "1.25",
    "--prefetch-storage", "delta"
)

$servicePrediction = @(
    "--prefetch-policy", "service",
    "--prediction-service-url", "http://127.0.0.1:8765",
    "--prediction-service-timeout-ms", "1000",
    "--prediction-service-reset", "1",
    "--prediction-top-k", "3",
    "--prefetch-min-probability", "0.10",
    "--max-prefetch-chunks-per-lora", "1",
    "--prefetch-cost-safety-factor", "1.25",
    "--prefetch-storage", "delta"
)

$fullScore = @(
    "--family-frequency-weight", "4",
    "--family-prediction-weight", "20",
    "--family-prefix-weight", "0.002",
    "--family-fanout-weight", "3",
    "--family-recency-weight", "4",
    "--family-segment-weight", "1",
    "--family-memory-weight", "0.05",
    "--family-materialize-weight", "0.0005",
    "--family-normalize-by-mb", "0"
)

$experiments = @()

if ($Suite -in @("PrefixDelta", "All")) {
    $experiments += New-Experiment "PrefixDelta" "prefix_only" $DefaultWorkload @(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests",
        "--cross-lora-policy", "off",
        "--background-policy", "none",
        "--prefetch-policy", "none"
    )
    $experiments += New-Experiment "PrefixDelta" "deferred_delta" $DefaultWorkload (@(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests"
    ) + $deferred + $noPrediction)
}

if ($Suite -in @("Prediction", "All")) {
    $experiments += New-Experiment "Prediction" "no_prediction" $DefaultWorkload (@(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests"
    ) + $deferred + $noPrediction)
    $experiments += New-Experiment "Prediction" "gru_file" $DefaultWorkload (@(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests"
    ) + $deferred + $filePrediction)
    $experiments += New-Experiment "Prediction" "gru_online" $DefaultWorkload (@(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests"
    ) + $deferred + $servicePrediction)
    $experiments += New-Experiment "Prediction" "oracle" $DefaultWorkload (@(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests",
        "--prefetch-policy", "oracle",
        "--prediction-top-k", "1",
        "--max-prefetch-chunks-per-lora", "1",
        "--prefetch-storage", "delta"
    ) + $deferred)
}

if ($Suite -in @("Storage", "All")) {
    $experiments += New-Experiment "Storage" "gpu_exact_only" $DefaultWorkload @(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests",
        "--cross-lora-policy", "off",
        "--background-policy", "none",
        "--prefetch-policy", "none",
        "--max-host-full-mb", "0",
        "--max-host-delta-mb", "0"
    )
    $experiments += New-Experiment "Storage" "gpu_plus_host_full" $DefaultWorkload @(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests",
        "--cross-lora-policy", "off",
        "--background-policy", "none",
        "--prefetch-policy", "none",
        "--max-host-full-mb", "4096",
        "--max-host-delta-mb", "0"
    )
    $experiments += New-Experiment "Storage" "host_anchor_delta" $DefaultWorkload (@(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests",
        "--max-host-full-mb", "4096",
        "--max-host-delta-mb", "1024"
    ) + $deferred + $noPrediction)
    $experiments += New-Experiment "Storage" "gru_gpu_ready" $DefaultWorkload (@(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests",
        "--max-host-full-mb", "4096",
        "--max-host-delta-mb", "1024"
    ) + $deferred + $servicePrediction)
}

if ($Suite -in @("Family", "All")) {
    $familyBase = @(
        "--max-delta-pairs", "0",
        "--max-online-requests", "$Requests",
        "--cross-lora-policy", "off",
        "--background-policy", "none"
    ) + $filePrediction
    $experiments += New-Experiment "Family" "recency" $DefaultWorkload ($familyBase + @(
        "--family-frequency-weight", "0", "--family-prediction-weight", "0",
        "--family-prefix-weight", "0", "--family-fanout-weight", "0",
        "--family-recency-weight", "4", "--family-segment-weight", "0",
        "--family-memory-weight", "0", "--family-materialize-weight", "0",
        "--family-normalize-by-mb", "0"
    ))
    $experiments += New-Experiment "Family" "frequency" $DefaultWorkload ($familyBase + @(
        "--family-frequency-weight", "4", "--family-prediction-weight", "0",
        "--family-prefix-weight", "0", "--family-fanout-weight", "0",
        "--family-recency-weight", "0", "--family-segment-weight", "0",
        "--family-memory-weight", "0", "--family-materialize-weight", "0",
        "--family-normalize-by-mb", "0"
    ))
    $experiments += New-Experiment "Family" "frequency_recency" $DefaultWorkload ($familyBase + @(
        "--family-frequency-weight", "4", "--family-prediction-weight", "0",
        "--family-prefix-weight", "0", "--family-fanout-weight", "0",
        "--family-recency-weight", "4", "--family-segment-weight", "0",
        "--family-memory-weight", "0", "--family-materialize-weight", "0",
        "--family-normalize-by-mb", "0"
    ))
    $experiments += New-Experiment "Family" "frequency_prediction" $DefaultWorkload ($familyBase + @(
        "--family-frequency-weight", "4", "--family-prediction-weight", "20",
        "--family-prefix-weight", "0", "--family-fanout-weight", "0",
        "--family-recency-weight", "4", "--family-segment-weight", "0",
        "--family-memory-weight", "0", "--family-materialize-weight", "0",
        "--family-normalize-by-mb", "0"
    ))
    $experiments += New-Experiment "Family" "full_latency" $DefaultWorkload ($familyBase + $fullScore)
    $experiments += New-Experiment "Family" "full_density" $DefaultWorkload ($familyBase + $fullScore + @(
        "--family-normalize-by-mb", "1"
    ))
}

if ($Suite -in @("Grouping", "All")) {
    $profiles = @{
        random = "D:\ecnu_experiment\datasets\mobilora_workloads_87_group_random"
        semantic = "D:\ecnu_experiment\datasets\mobilora_workloads_87_group_semantic"
        transition = "D:\ecnu_experiment\datasets\mobilora_workloads_87_group_transition"
        hybrid = "D:\ecnu_experiment\datasets\mobilora_workloads_87_group_hybrid"
    }
    foreach ($name in @("random", "semantic", "transition", "hybrid")) {
        $experiments += New-Experiment "Grouping" $name $profiles[$name] @(
            "--max-delta-pairs", "$DeltaPairs",
            "--max-online-requests", "$Requests",
            "--cross-lora-policy", "off",
            "--background-policy", "none",
            "--prefetch-policy", "none",
            "--delta-store-policy", "none"
        )
    }
}

if ($Case -ne "all") {
    $experiments = @($experiments | Where-Object { $_.Name -eq $Case })
    if ($experiments.Count -eq 0) {
        throw "No case named '$Case' in suite '$Suite'."
    }
}

foreach ($experiment in $experiments) {
    if (-not (Test-Path -LiteralPath $experiment.Workload)) {
        throw "Workload not found: $($experiment.Workload)"
    }
    if ($experiment.Name -in @("gru_online", "gru_gpu_ready") -and -not $DryRun) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 2
            if ($health.status -ne "ok") { throw "health status is not ok" }
        } catch {
            throw "GRU service is not available. Run start_gru_service.ps1 first."
        }
    }

    for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
        $tag = if ([string]::IsNullOrWhiteSpace($RunTag)) { "" } else { "_$RunTag" }
        $repeatSuffix = if ($Repeats -gt 1) { "_r{0:D2}" -f $repeat } else { "" }
        $output = Join-Path $OutputRoot "output_ablation_$($experiment.Suite)_$($experiment.Name)$tag$repeatSuffix"
        $arguments = @(Get-CommonArguments $experiment.Workload) +
            @($experiment.Arguments) + @("--output-dir", $output)
        Write-Host "[$($experiment.Suite)] $($experiment.Name) repeat=$repeat/$Repeats"
        Write-Host "$Exe $($arguments -join ' ')"
        if ($DryRun) { continue }

        & $Exe @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment '$($experiment.Name)' repeat $repeat failed with exit code $LASTEXITCODE"
        }
    }
}
