$ErrorActionPreference = "Continue"

if (Test-Path .\.venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
}

$env:MOE_FUGU_CONFIG = ".\configs\moe_fugu_minipc_local.yaml"
$env:MOE_FUGU_LOG = ".\logs\moe_fugu_runs.jsonl"
$env:TOKENIZERS_PARALLELISM = "false"

# Keep workers=1. The anti-fanout lock is per process, and the backend machine
# is designed for one hot-loaded model at a time.
New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
python -m uvicorn src.conductor_7940hs:app --host 127.0.0.1 --port 9000 --workers 1 *> ".\logs\frontdoor.log"
