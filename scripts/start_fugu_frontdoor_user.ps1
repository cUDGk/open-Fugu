$ErrorActionPreference = "Continue"

$Project = "C:\llm\local-moe-fugu-7940hs"
if (!(Test-Path $Project)) {
  throw "Project directory not found: $Project"
}

Set-Location $Project

if (Test-Path .\.venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
}

$env:MOE_FUGU_CONFIG = ".\configs\moe_fugu_minipc_local.yaml"
$env:MOE_FUGU_LOG = ".\logs\moe_fugu_runs.jsonl"
$env:TOKENIZERS_PARALLELISM = "false"

New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
python -m uvicorn src.conductor_7940hs:app --host 127.0.0.1 --port 9000 --workers 1 *> ".\logs\frontdoor.log"
