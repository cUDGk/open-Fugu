$ErrorActionPreference = "Stop"

if (Test-Path .\.venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
}

$env:MOE_FUGU_CONFIG = ".\configs\moe_fugu_hf.yaml"
$env:MOE_FUGU_LOG = ".\logs\moe_fugu_runs.jsonl"

# --workers 1 必須: fanout防止のcall_lockはプロセス内Lockのため、
# 複数ワーカーだと並列モデル呼び出しを防げずメモリが破綻する。
python -m uvicorn src.conductor_7940hs:app --host 127.0.0.1 --port 9000 --workers 1
