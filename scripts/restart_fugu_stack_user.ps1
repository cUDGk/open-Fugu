param(
  [int]$ModelsMax = 2,
  [int]$Ctx = 16384,
  [int]$Threads = 8,
  [int]$ThreadsBatch = 16,
  [int]$Batch = 512,
  [int]$UBatch = 128,
  [string]$CacheTypeK = "q8_0",
  [string]$CacheTypeV = "q8_0",
  [string]$CpuFlashAttn = "on",
  [int]$LlamaPrio = 2,
  [int]$Poll = -1,
  [int]$CacheReuse = 0,
  [string]$ModelsPreset = "",
  [switch]$DisableNgramSpec,
  [bool]$EnableVulkanHybrid = $true,
  [string]$VulkanServer = "C:\llm\bin\llama-server.exe",
  [int]$VulkanPort = 8081,
  [int]$VulkanModelsMax = 1,
  [int]$VulkanGpuLayers = 99,
  [int]$VulkanThreads = 16,
  [int]$VulkanBatch = 512,
  [int]$VulkanUBatch = 512,
  [string]$VulkanCacheTypeK = "f16",
  [string]$VulkanCacheTypeV = "f16",
  [string]$VulkanFlashAttn = "on",
  [ValidateSet("Normal", "AboveNormal", "High")]
  [string]$ProcessPriority = "High",
  [switch]$SkipWarmup
)

$ErrorActionPreference = "Continue"

$RouterScript = "C:\Users\user\start_server.ps1"
$FrontdoorScript = "C:\Users\user\start_fugu_frontdoor.ps1"
$Project = "C:\llm\local-moe-fugu-7940hs"
$RouterTask = "LocalFuguRouter"
$VulkanRouterTask = "LocalFuguVulkanRouter"
$FrontdoorTask = "LocalFuguFrontdoor"
$RouterWrapper = "C:\Users\user\run_fugu_router_task.ps1"
$VulkanRouterWrapper = "C:\Users\user\run_fugu_vulkan_router_task.ps1"
$FrontdoorWrapper = "C:\Users\user\run_fugu_frontdoor_task.ps1"

function Stop-PortOwner {
  param([int]$Port)
  $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($connection in $connections) {
    if ($connection.OwningProcess -gt 0) {
      Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
    }
  }
}

function Wait-JsonEndpoint {
  param(
    [string]$Uri,
    [int]$Seconds = 120
  )
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $null = Invoke-RestMethod -Uri $Uri -TimeoutSec 5
      return $true
    } catch {
      Start-Sleep -Seconds 2
    }
  }
  return $false
}

function Start-OnceTask {
  param(
    [string]$TaskName,
    [string]$Command
  )
  $startTime = (Get-Date).AddMinutes(5).ToString("HH:mm")
  schtasks.exe /End /TN $TaskName *> $null
  schtasks.exe /Delete /TN $TaskName /F *> $null
  schtasks.exe /Create /TN $TaskName /SC ONCE /ST $startTime /TR $Command /F | Out-Null
  schtasks.exe /Run /TN $TaskName | Out-Null
}

function Set-FuguProcessPriority {
  param([string]$PriorityClass)

  if ($PriorityClass -eq "Normal") {
    return
  }

  $targets = Get-CimInstance Win32_Process |
    Where-Object {
      $_.CommandLine -like '*llama-server*' -or
      $_.CommandLine -like '*uvicorn src.conductor_7940hs*'
    }

  foreach ($target in $targets) {
    $process = Get-Process -Id $target.ProcessId -ErrorAction SilentlyContinue
    if ($process) {
      try {
        $process.PriorityClass = $PriorityClass
      } catch {
        Write-Warning ("Could not set priority for pid={0}: {1}" -f $target.ProcessId, $_.Exception.Message)
      }
    }
  }
}

function Invoke-FuguWarmup {
  param([string]$ProjectPath)

  $warmupLog = Join-Path $ProjectPath "logs\fugu_warmup.jsonl"
  $cases = @(
    @{
      name = "warm_gemma_japanese"
      max_tokens = 8
      content_b64 = "5pel5pys6Kqe44GnT0vjgaDjgZHov5TjgZfjgabjgII="
    },
    @{
      name = "warm_qwen_code"
      max_tokens = 24
      content = "Python: return only one line of code for add(a, b)."
    }
  )
  if ($EnableVulkanHybrid) {
    $cases += @(
      @{
        name = "warm_qwen_vulkan_long"
        max_tokens = 160
        content = "Write a compact Python function named normalize_scores(scores) that returns values scaled to 0..1 and handles an empty list."
      }
    )
  }

  foreach ($case in $cases) {
    if ($case.ContainsKey("content_b64")) {
      $content = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($case.content_b64))
    } else {
      $content = $case.content
    }
    $body = @{
      model = "local-moe-fugu"
      messages = @(@{ role = "user"; content = $content })
      temperature = 0
      max_tokens = $case.max_tokens
    } | ConvertTo-Json -Depth 8

    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
      $response = Invoke-RestMethod `
        -Uri "http://127.0.0.1:9000/v1/chat/completions" `
        -Method Post `
        -ContentType "application/json; charset=utf-8" `
        -Body $body `
        -TimeoutSec 600
      $sw.Stop()
      $route = $response.system_fingerprint.route
      $perf = $response.system_fingerprint.performance
      $row = [ordered]@{
        created = (Get-Date).ToString("s")
        name = $case.name
        ok = $true
        elapsed_s = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        route_primary = $route.primary
        route_mode = $route.mode
        router = $route.router
        completion_tokens = $response.usage.completion_tokens
        actual_completion_tps = $perf.actual_completion_tps
        theoretical_completion_tps = $perf.theoretical_completion_tps
        backend_decode_tps = $perf.backend_decode_tps
        overhead_ms = $perf.overhead_ms
      }
      Write-Host ("WARMUP {0} route={1}/{2} elapsed={3}s actual_tps={4} theory_tps={5}" -f $case.name, $route.primary, $route.mode, $row.elapsed_s, $row.actual_completion_tps, $row.theoretical_completion_tps)
    } catch {
      $sw.Stop()
      $row = [ordered]@{
        created = (Get-Date).ToString("s")
        name = $case.name
        ok = $false
        elapsed_s = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        error = $_.Exception.Message
      }
      Write-Warning ("WARMUP {0} failed: {1}" -f $case.name, $_.Exception.Message)
    }
    ($row | ConvertTo-Json -Compress -Depth 8) | Add-Content -LiteralPath $warmupLog -Encoding UTF8
  }
}

if (!(Test-Path $RouterScript)) {
  throw "Router script not found: $RouterScript"
}
if (!(Test-Path $FrontdoorScript)) {
  throw "Frontdoor script not found: $FrontdoorScript"
}
if (!(Test-Path $Project)) {
  throw "Project directory not found: $Project"
}
if ($EnableVulkanHybrid -and !(Test-Path $VulkanServer)) {
  Write-Warning "Vulkan llama-server not found; disabling hybrid backend: $VulkanServer"
  $EnableVulkanHybrid = $false
}

New-Item -ItemType Directory -Force -Path (Join-Path $Project "logs") | Out-Null
Remove-Item -Force (Join-Path $Project "logs\frontdoor.log") -ErrorAction SilentlyContinue

Stop-PortOwner -Port 9000
Stop-PortOwner -Port 8080
Stop-PortOwner -Port $VulkanPort
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$routerCall = "& `"$RouterScript`" -StopExisting"
$routerCall += " -ModelsMax $ModelsMax"
$routerCall += " -Ctx $Ctx"
$routerCall += " -Threads $Threads"
$routerCall += " -ThreadsBatch $ThreadsBatch"
$routerCall += " -Batch $Batch"
$routerCall += " -UBatch $UBatch"
$routerCall += " -CacheTypeK $CacheTypeK"
$routerCall += " -CacheTypeV $CacheTypeV"
$routerCall += " -FlashAttn $CpuFlashAttn"
$routerCall += " -LlamaPrio $LlamaPrio"
if ($Poll -ge 0) {
  $routerCall += " -Poll $Poll"
}
$routerCall += " -CacheReuse $CacheReuse"
if ($ModelsPreset -ne "") {
  $routerCall += " -ModelsPreset `"$ModelsPreset`""
}
if ($DisableNgramSpec) {
  $routerCall += " -DisableNgramSpec"
}

$vulkanLog = Join-Path $Project "logs\vulkan_router.log"
$vulkanErrLog = Join-Path $Project "logs\vulkan_router.err.log"
$vulkanArgs = @(
  "--models-dir", "C:\llm\models",
  "--host", "127.0.0.1",
  "--port", "$VulkanPort",
  "--models-max", "$VulkanModelsMax",
  "-c", "$Ctx",
  "-np", "1",
  "--cont-batching",
  "--jinja",
  "--cache-prompt",
  "--cache-idle-slots",
  "--reasoning", "off",
  "--reasoning-budget", "0",
  "-ctk", "$VulkanCacheTypeK",
  "-ctv", "$VulkanCacheTypeV",
  "-t", "$VulkanThreads",
  "-b", "$VulkanBatch",
  "-ub", "$VulkanUBatch",
  "-fa", "$VulkanFlashAttn",
  "--prio", "$LlamaPrio",
  "--prio-batch", "$LlamaPrio",
  "--repeat-penalty", "1.0",
  "-ngl", "$VulkanGpuLayers"
)
if ($Poll -ge 0) {
  $vulkanArgs += @("--poll", "$Poll", "--poll-batch", "1")
}
if (-not $DisableNgramSpec) {
  $vulkanArgs += @(
    "--spec-type", "ngram-cache",
    "--spec-ngram-mod-n-min", "1",
    "--spec-ngram-mod-n-max", "8",
    "--spec-ngram-mod-n-match", "24"
  )
}
$vulkanArgText = ($vulkanArgs | ForEach-Object {
  if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join " "

$routerWrapperContent = @"
`$ErrorActionPreference = "Continue"
$routerCall
"@
$vulkanWrapperContent = @"
`$ErrorActionPreference = "Continue"
& "$VulkanServer" $vulkanArgText *> "$vulkanLog"
"@
$frontdoorWrapperContent = @"
`$ErrorActionPreference = "Continue"
& "$FrontdoorScript"
"@
Set-Content -LiteralPath $RouterWrapper -Value $routerWrapperContent -Encoding UTF8
Set-Content -LiteralPath $VulkanRouterWrapper -Value $vulkanWrapperContent -Encoding UTF8
Set-Content -LiteralPath $FrontdoorWrapper -Value $frontdoorWrapperContent -Encoding UTF8

$routerCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RouterWrapper`""
$vulkanCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$VulkanRouterWrapper`""
$frontdoorCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$FrontdoorWrapper`""

Start-OnceTask -TaskName $RouterTask -Command $routerCommand

if (!(Wait-JsonEndpoint -Uri "http://127.0.0.1:8080/v1/models" -Seconds 180)) {
  throw "Router did not become ready on http://127.0.0.1:8080/v1/models"
}
Set-FuguProcessPriority -PriorityClass $ProcessPriority

if ($EnableVulkanHybrid) {
  Start-OnceTask -TaskName $VulkanRouterTask -Command $vulkanCommand

  if (!(Wait-JsonEndpoint -Uri "http://127.0.0.1:$VulkanPort/v1/models" -Seconds 180)) {
    throw "Vulkan router did not become ready on http://127.0.0.1:$VulkanPort/v1/models"
  }
  Set-FuguProcessPriority -PriorityClass $ProcessPriority
}

Start-OnceTask -TaskName $FrontdoorTask -Command $frontdoorCommand

if (!(Wait-JsonEndpoint -Uri "http://127.0.0.1:9000/health" -Seconds 90)) {
  throw "Frontdoor did not become ready on http://127.0.0.1:9000/health"
}
Set-FuguProcessPriority -PriorityClass $ProcessPriority

if (!$SkipWarmup) {
  Invoke-FuguWarmup -ProjectPath $Project
  Set-FuguProcessPriority -PriorityClass $ProcessPriority
}

$vulkanStatus = if ($EnableVulkanHybrid) { " vulkan=http://127.0.0.1:$VulkanPort" } else { "" }
Write-Host "READY router=http://127.0.0.1:8080$vulkanStatus frontdoor=http://127.0.0.1:9000"
