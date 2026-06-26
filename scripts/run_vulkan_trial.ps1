param(
  [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [string]$VulkanServer = "C:\llm\bin\llama-server.exe",
  [string]$ModelsDir = "C:\llm\models",
  [int]$BackendPort = 8080,
  [int]$FrontdoorPort = 9000,
  [int]$ModelsMax = 2,
  [int]$Ctx = 16384,
  [int]$Threads = 12,
  [int]$Batch = 512,
  [int]$UBatch = 128,
  [int]$GpuLayers = 99,
  [switch]$IncludeAverage,
  [switch]$NoBench
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-PortOwner {
  param([Parameter(Mandatory = $true)][int]$Port)

  $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($conn in $conns) {
    try {
      Stop-Process -Id $conn.OwningProcess -Force -ErrorAction Stop
    } catch {
      Write-Warning "Failed to stop process $($conn.OwningProcess) on port ${Port}: $_"
    }
  }
}

function Wait-JsonEndpoint {
  param(
    [Parameter(Mandatory = $true)][string]$Uri,
    [int]$Seconds = 180
  )

  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    try {
      Invoke-RestMethod -Uri $Uri -TimeoutSec 5 | Out-Null
      return $true
    } catch {
      Start-Sleep -Seconds 2
    }
  }
  return $false
}

function Invoke-Warmup {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Prompt,
    [int]$MaxTokens = 16
  )

  $body = @{
    model = "local-moe-fugu"
    messages = @(@{ role = "user"; content = $Prompt })
    max_tokens = $MaxTokens
    temperature = 0
  } | ConvertTo-Json -Depth 6

  $started = Get-Date
  $response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:$FrontdoorPort/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
    -TimeoutSec 240
  $elapsed = ((Get-Date) - $started).TotalSeconds
  $perf = $response.system_fingerprint.performance
  $routePrimary = ""
  $routeMode = ""
  $decodeTps = ""
  if ($null -ne $perf) {
    if ($perf.PSObject.Properties.Name -contains "route_primary") { $routePrimary = $perf.route_primary }
    if ($perf.PSObject.Properties.Name -contains "route_mode") { $routeMode = $perf.route_mode }
    if ($perf.PSObject.Properties.Name -contains "backend_decode_tps") { $decodeTps = $perf.backend_decode_tps }
  }
  Write-Host ("WARMUP {0} route={1}/{2} elapsed={3:n3}s backend_decode_tps={4}" -f `
    $Name, $routePrimary, $routeMode, $elapsed, $decodeTps)
}

Push-Location $ProjectRoot
try {
  if (!(Test-Path -LiteralPath $VulkanServer)) {
    throw "Vulkan llama-server not found: $VulkanServer"
  }
  if (!(Test-Path -LiteralPath $ModelsDir)) {
    throw "Models directory not found: $ModelsDir"
  }

  New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null

  Stop-PortOwner -Port $FrontdoorPort
  Stop-PortOwner -Port $BackendPort
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force

  $backendOut = Join-Path $ProjectRoot "logs\vulkan_router_trial.out.log"
  $backendErr = Join-Path $ProjectRoot "logs\vulkan_router_trial.err.log"
  $frontOut = Join-Path $ProjectRoot "logs\vulkan_frontdoor_trial.out.log"
  $frontErr = Join-Path $ProjectRoot "logs\vulkan_frontdoor_trial.err.log"
  Remove-Item -LiteralPath $backendOut,$backendErr,$frontOut,$frontErr -Force -ErrorAction SilentlyContinue

  $backendArgs = @(
    "--models-dir", $ModelsDir,
    "--host", "127.0.0.1",
    "--port", "$BackendPort",
    "--models-max", "$ModelsMax",
    "-c", "$Ctx",
    "-np", "1",
    "--cont-batching",
    "--jinja",
    "--cache-prompt",
    "--cache-idle-slots",
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "-t", "$Threads",
    "-b", "$Batch",
    "-ub", "$UBatch",
    "--repeat-penalty", "1.0",
    "-ngl", "$GpuLayers",
    "--spec-type", "ngram-cache",
    "--spec-ngram-mod-n-min", "1",
    "--spec-ngram-mod-n-max", "8",
    "--spec-ngram-mod-n-match", "24"
  )

  $backend = Start-Process `
    -FilePath $VulkanServer `
    -ArgumentList $backendArgs `
    -RedirectStandardOutput $backendOut `
    -RedirectStandardError $backendErr `
    -WindowStyle Hidden `
    -PassThru

  if (!(Wait-JsonEndpoint -Uri "http://127.0.0.1:$BackendPort/v1/models" -Seconds 180)) {
    throw "Vulkan backend did not become ready on port $BackendPort"
  }

  $env:MOE_FUGU_CONFIG = ".\configs\moe_fugu_minipc_local.yaml"
  $env:MOE_FUGU_LOG = ".\logs\moe_fugu_runs.vulkan_trial.jsonl"
  $env:TOKENIZERS_PARALLELISM = "false"

  $front = Start-Process `
    -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList @("-m", "uvicorn", "src.conductor_7940hs:app", "--host", "127.0.0.1", "--port", "$FrontdoorPort", "--workers", "1") `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $frontOut `
    -RedirectStandardError $frontErr `
    -WindowStyle Hidden `
    -PassThru

  if (!(Wait-JsonEndpoint -Uri "http://127.0.0.1:$FrontdoorPort/health" -Seconds 90)) {
    throw "FUGU frontdoor did not become ready on port $FrontdoorPort"
  }

  Invoke-Warmup -Name "vulkan_warm_qwen_code" -Prompt "Return only this Python expression: sum(range(1, 11))" -MaxTokens 16

  if (-not $NoBench) {
    & ".\.venv\Scripts\python.exe" scripts\bench_fugu_standard.py
    if ($LASTEXITCODE -ne 0) {
      throw "standard benchmark failed with exit code $LASTEXITCODE"
    }
    & ".\.venv\Scripts\python.exe" scripts\bench_fugu_japanese.py
    if ($LASTEXITCODE -ne 0) {
      throw "japanese benchmark failed with exit code $LASTEXITCODE"
    }
    if ($IncludeAverage) {
      & ".\.venv\Scripts\python.exe" scripts\bench_fugu_average.py
      if ($LASTEXITCODE -ne 0) {
        throw "average benchmark failed with exit code $LASTEXITCODE"
      }
    }
  }

  Write-Host "READY_VULKAN_TRIAL backend_pid=$($backend.Id) frontdoor_pid=$($front.Id)"
}
finally {
  Pop-Location
}
