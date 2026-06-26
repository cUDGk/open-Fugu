param(
  [string]$Project = "C:\llm\local-moe-fugu-7940hs",
  [string]$ModelRepo = "Qwen/Qwen3-0.6B",
  [string]$ModelDir = "C:\llm\models\Qwen3-0.6B"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Project)) {
  throw "Project directory not found: $Project"
}

Set-Location $Project

if (!(Test-Path .\.venv\Scripts\Activate.ps1)) {
  python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install numpy huggingface_hub "transformers>=4.52,<5" safetensors accelerate pyyaml
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$download = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$ModelRepo",
    local_dir=r"$ModelDir",
    local_dir_use_symlinks=False,
)
print(r"$ModelDir")
"@
$downloadScript = Join-Path $env:TEMP "download_fugu_router_06b.py"
Set-Content -LiteralPath $downloadScript -Value $download -Encoding UTF8
python $downloadScript
Remove-Item -LiteralPath $downloadScript -Force -ErrorAction SilentlyContinue

python .\scripts\train_router_06b_seed_head.py --output ".\artifacts\router_06b_seed_head.npz"

$warmup = @"
from pathlib import Path
import sys
import yaml

root = Path.cwd()
sys.path.insert(0, str(root))
from src.fugu_router_06b import ZeroSixBTower

cfg = yaml.safe_load((root / "configs" / "moe_fugu_minipc_local.yaml").read_text(encoding="utf-8"))
tower = ZeroSixBTower.from_config(cfg, root)
if tower is None:
    raise SystemExit("router_06b is disabled in config")
decision = tower.route([{"role": "user", "content": "Route this setup check to the best local worker."}])
print(decision)
print(tower.state())
"@
$warmupScript = Join-Path $env:TEMP "warm_fugu_router_06b.py"
Set-Content -LiteralPath $warmupScript -Value $warmup -Encoding UTF8
python $warmupScript
Remove-Item -LiteralPath $warmupScript -Force -ErrorAction SilentlyContinue

Write-Host "READY router_06b model=$ModelDir"
