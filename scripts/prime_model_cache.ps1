$ErrorActionPreference = "Stop"

# This primes llama.cpp's HF cache. It can take a long time and requires enough SSD space.
# If a model fails to download or run, comment it out and test models one by one.

$Prompt = "ping"
$Tokens = 1

$Models = @(
  "bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M",
  "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q4_K_M",
  "unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL"
)

foreach ($Model in $Models) {
  Write-Host "Priming $Model"
  llama-cli -hf $Model -p $Prompt -n $Tokens --repeat-penalty 1.0
}

Write-Host "Cache priming attempted. Now run scripts\start_llama_router.ps1"
