param(
  [string]$Project = "C:\llm\local-moe-fugu-7940hs",
  [string]$ModelsDir = "C:\llm\models"
)

$ErrorActionPreference = "Continue"

if (!(Test-Path $Project)) {
  throw "Project directory not found: $Project"
}

Set-Location $Project

if (!(Test-Path .\.venv\Scripts\Activate.ps1)) {
  python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1

$Log = "C:\llm\replacement_download.log"
$Done = "C:\llm\replacement_models.DONE"
Remove-Item -Force $Done -ErrorAction SilentlyContinue
Remove-Item -Force $Log -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# hf_transfer is much faster than single-stream curl for multi-GB GGUF files.
# Disable Xet here because the Windows xet path can stall without visible progress.
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
$env:HF_HUB_DISABLE_XET = "1"

python -m pip install "huggingface_hub>=0.34,<1.0" hf_transfer

$py = @"
from __future__ import annotations

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

models_dir = Path(r"$ModelsDir")
done = Path(r"$Done")
targets = [
    {
        "repo_id": "unsloth/gemma-3-12b-it-GGUF",
        "filename": "gemma-3-12b-it-Q4_K_M.gguf",
        "expected": 7300778336,
    },
    {
        "repo_id": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "filename": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "expected": 18556689568,
    },
]

results = []
for target in targets:
    repo_id = target["repo_id"]
    filename = target["filename"]
    expected = int(target["expected"])
    dst = models_dir / filename
    print(f"START {repo_id}/{filename} expected={expected}", flush=True)
    if dst.exists() and dst.stat().st_size >= int(expected * 0.98):
        print(f"SKIP existing {filename} bytes={dst.stat().st_size}", flush=True)
    else:
        if dst.exists():
            print(f"REMOVE undersized {filename} bytes={dst.stat().st_size}", flush=True)
            dst.unlink()
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(models_dir),
        )
    size = dst.stat().st_size if dst.exists() else 0
    if size < int(expected * 0.98):
        raise RuntimeError(f"{filename} too small: {size} expected {expected}")
    print(f"DONE {filename} bytes={size}", flush=True)
    results.append({"repo_id": repo_id, "filename": filename, "path": str(dst), "bytes": size})

done.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("ALL_DONE", flush=True)
"@

$tmp = Join-Path $env:TEMP "download_replacement_models.py"
Set-Content -LiteralPath $tmp -Value $py -Encoding UTF8
python $tmp 2>&1 | Tee-Object -FilePath $Log
$exit = $LASTEXITCODE
Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
exit $exit
