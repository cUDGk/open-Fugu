$ErrorActionPreference = "Stop"

Write-Host "Installing base tools with winget if missing..."

function Ensure-WingetPackage($Id) {
  $existing = winget list --id $Id --exact 2>$null
  if ($LASTEXITCODE -ne 0 -or -not ($existing -match $Id)) {
    winget install --id $Id --exact --accept-package-agreements --accept-source-agreements
  } else {
    Write-Host "$Id already installed"
  }
}

Ensure-WingetPackage "Python.Python.3.12"
Ensure-WingetPackage "Git.Git"
Ensure-WingetPackage "llama.cpp"

Write-Host "Creating virtual environment..."
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install -r .\requirements.txt

New-Item -ItemType Directory -Force -Path .\logs | Out-Null

Write-Host "Done. Next: scripts\prime_model_cache.ps1 or scripts\start_llama_router.ps1"
