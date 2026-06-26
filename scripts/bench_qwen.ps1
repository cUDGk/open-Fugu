$ErrorActionPreference = "Stop"

# Run after Qwen has been cached. Adjust -hf to match the quant you use.

$Model = "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q4_K_M"

Write-Host "CPU baseline"
llama-bench `
  -hf $Model `
  -p 512 `
  -n 128 `
  -c 8192 `
  -t 12

Write-Host "Try iGPU/full offload if your llama.cpp build supports it. May or may not be faster on Radeon 780M shared memory."
llama-bench `
  -hf $Model `
  -p 512 `
  -n 128 `
  -c 8192 `
  -t 12 `
  -ngl 99 `
  -fa on

Write-Host "Try MoE on CPU with GPU offload path"
llama-bench `
  -hf $Model `
  -p 512 `
  -n 128 `
  -c 8192 `
  -t 12 `
  -ngl 99 `
  -fa on `
  --cpu-moe
