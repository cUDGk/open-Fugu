param(
  [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [string]$Python = ".venv\Scripts\python.exe",
  [string]$RestartScript = "C:\Users\user\restart_fugu_stack.ps1",
  [switch]$SkipRestart,
  [double]$StandardMeanMax = 0.85,
  [double]$StandardP90Max = 1.40,
  [double]$BackendDecodeTpsMin = 25.0,
  [double]$JapaneseMeanMax = 0.20,
  [double]$JapaneseMaxMax = 0.50
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-LatestSummary {
  param(
    [Parameter(Mandatory = $true)][string]$Pattern
  )

  $file = Get-ChildItem -Path (Join-Path $ProjectRoot "logs") -Filter $Pattern |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  if (-not $file) {
    throw "No summary found for pattern: $Pattern"
  }

  $json = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
  return [pscustomobject]@{
    File = $file.FullName
    Data = $json
  }
}

function Add-Check {
  param(
    [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.ArrayList]$Failures,
    [Parameter(Mandatory = $true)][bool]$Ok,
    [Parameter(Mandatory = $true)][string]$Message
  )

  if (-not $Ok) {
    [void]$Failures.Add($Message)
  }
}

Push-Location $ProjectRoot
try {
  $pythonPath = Join-Path $ProjectRoot $Python
  if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python not found: $pythonPath"
  }

  if (-not $SkipRestart) {
    if (Test-Path -LiteralPath $RestartScript) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File $RestartScript
      if ($LASTEXITCODE -ne 0) {
        throw "Restart script failed with exit code $LASTEXITCODE"
      }
    } else {
      Write-Warning "Restart script not found; continuing without restart: $RestartScript"
    }
  }

  & $pythonPath scripts\bench_fugu_standard.py
  if ($LASTEXITCODE -ne 0) {
    throw "Standard benchmark failed with exit code $LASTEXITCODE"
  }

  & $pythonPath scripts\bench_fugu_japanese.py
  if ($LASTEXITCODE -ne 0) {
    throw "Japanese benchmark failed with exit code $LASTEXITCODE"
  }

  $standard = Get-LatestSummary -Pattern "standard_fugu_*.summary.json"
  $japanese = Get-LatestSummary -Pattern "japanese_fugu_*.summary.json"
  $overall = $standard.Data.groups | Where-Object { $_.label -eq "overall" } | Select-Object -First 1
  if (-not $overall) {
    throw "Standard benchmark summary does not contain an overall group."
  }

  $failures = [System.Collections.ArrayList]::new()
  Add-Check $failures ($standard.Data.failed_runs -eq 0) "standard failed_runs must be 0"
  Add-Check $failures ($standard.Data.ok_runs -eq $standard.Data.runs) "standard ok_runs must equal runs"
  Add-Check $failures ([double]$overall.accuracy -ge 1.0) "standard accuracy must be 1.0"
  Add-Check $failures ([double]$overall.mean_elapsed_s -le $StandardMeanMax) "standard mean_elapsed_s above $StandardMeanMax"
  Add-Check $failures ([double]$overall.p90_elapsed_s -le $StandardP90Max) "standard p90_elapsed_s above $StandardP90Max"
  Add-Check $failures ([double]$overall.backend_decode_tps -ge $BackendDecodeTpsMin) "backend_decode_tps below $BackendDecodeTpsMin"
  Add-Check $failures ($japanese.Data.failed_runs -eq 0) "japanese failed_runs must be 0"
  Add-Check $failures ($japanese.Data.ok_runs -eq $japanese.Data.runs) "japanese ok_runs must equal runs"
  Add-Check $failures ([double]$japanese.Data.accuracy -ge 1.0) "japanese accuracy must be 1.0"
  Add-Check $failures ([double]$japanese.Data.mean_elapsed_s -le $JapaneseMeanMax) "japanese mean_elapsed_s above $JapaneseMeanMax"
  Add-Check $failures ([double]$japanese.Data.max_elapsed_s -le $JapaneseMaxMax) "japanese max_elapsed_s above $JapaneseMaxMax"

  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $result = [ordered]@{
    created = (Get-Date).ToString("s")
    gate = "fugu_clean_speed"
    passed = ($failures.Count -eq 0)
    thresholds = [ordered]@{
      standard_mean_elapsed_s_max = $StandardMeanMax
      standard_p90_elapsed_s_max = $StandardP90Max
      backend_decode_tps_min = $BackendDecodeTpsMin
      japanese_mean_elapsed_s_max = $JapaneseMeanMax
      japanese_max_elapsed_s_max = $JapaneseMaxMax
    }
    standard = [ordered]@{
      file = $standard.File
      runs = $standard.Data.runs
      ok_runs = $standard.Data.ok_runs
      failed_runs = $standard.Data.failed_runs
      accuracy = $overall.accuracy
      mean_elapsed_s = $overall.mean_elapsed_s
      p90_elapsed_s = $overall.p90_elapsed_s
      max_elapsed_s = $overall.max_elapsed_s
      completion_tps = $overall.completion_tps
      request_tps = $overall.request_tps
      backend_decode_tps = $overall.backend_decode_tps
      backend_prefill_tps = $overall.backend_prefill_tps
      efficiency_vs_backend_theory = $overall.efficiency_vs_backend_theory
    }
    japanese = [ordered]@{
      file = $japanese.File
      runs = $japanese.Data.runs
      ok_runs = $japanese.Data.ok_runs
      failed_runs = $japanese.Data.failed_runs
      accuracy = $japanese.Data.accuracy
      mean_elapsed_s = $japanese.Data.mean_elapsed_s
      p90_elapsed_s = $japanese.Data.p90_elapsed_s
      max_elapsed_s = $japanese.Data.max_elapsed_s
    }
    failures = @($failures)
  }

  $outPath = Join-Path $ProjectRoot ("logs\clean_speed_gate_{0}.json" -f $stamp)
  $latestPath = Join-Path $ProjectRoot "logs\clean_speed_gate_latest.json"
  $jsonOut = $result | ConvertTo-Json -Depth 8
  $jsonOut | Set-Content -LiteralPath $outPath -Encoding UTF8
  $jsonOut | Set-Content -LiteralPath $latestPath -Encoding UTF8

  Write-Host $jsonOut
  if ($failures.Count -gt 0) {
    exit 2
  }
}
finally {
  Pop-Location
}
