param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$Python = "D:\anaconda\envs\qwen2.5_vl\python.exe"
$Script = "D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\gru_online_service.py"
$Checkpoint = "D:\ecnu_experiment\LoRA\clustering_weights\user_Subsequence_exponential411_time10\base_model_global.pth"
$Mapping = "D:\ecnu_experiment\datasets\mobilora_workloads_87_original\app_lora_mapping.csv"
$LogFile = "D:\ecnu_experiment\LLama.cpp\llama.cpp\examples\lora-base-test2\output_gru_online\gru_service_predictions.jsonl"

& $Python $Script `
    --checkpoint $Checkpoint `
    --mapping $Mapping `
    --host 127.0.0.1 `
    --port $Port `
    --window 15 `
    --min-history 15 `
    --device $Device `
    --torch-threads 1 `
    --log-file $LogFile

