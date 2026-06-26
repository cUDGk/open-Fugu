param(
  [switch]$StopExisting,
  [int]$ModelsMax = 2,
  [int]$Ctx = 16384,
  [int]$Threads = 8,
  [int]$ThreadsBatch = 16,
  [int]$Batch = 512,
  [int]$UBatch = 128,
  [string]$CacheTypeK = "q8_0",
  [string]$CacheTypeV = "q8_0",
  [string]$FlashAttn = "on",
  [int]$LlamaPrio = 2,
  [int]$Poll = -1,
  [int]$CacheReuse = 0,
  [string]$ModelsPreset = "",
  [switch]$DisableNgramSpec
)

$ErrorActionPreference = "Continue"

# <MINIPC-HOST> user-specific runtime.
# This replaces the older C:\Users\user\start_server.ps1 that used -c 4096.

$Server = "C:\llm\llama\llama-server.exe"
$ModelsDir = "C:\llm\models"
$Log = "C:\llm\server.log"
$HostAddr = "127.0.0.1"
$Port = 8080
$EnableNgramSpec = -not $DisableNgramSpec

if (!(Test-Path $Server)) {
  Write-Error "llama-server not found: $Server"
  exit 1
}
if (!(Test-Path $ModelsDir)) {
  Write-Error "Models directory not found: $ModelsDir"
  exit 1
}
if ($ModelsPreset -ne "" -and !(Test-Path $ModelsPreset)) {
  Write-Error "Models preset not found: $ModelsPreset"
  exit 1
}

if ($StopExisting) {
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
}

if (Test-Path $Log) {
  Remove-Item -Force $Log
}

$ModelSourceArgs = @("--models-dir", $ModelsDir)
if ($ModelsPreset -ne "") {
  $ModelSourceArgs = @("--models-preset", $ModelsPreset)
}

$ServerArgs = @()
$ServerArgs += $ModelSourceArgs
$ServerArgs += @(
  "--host", $HostAddr,
  "--port", "$Port",
  "--models-max", "$ModelsMax",
  "-c", "$Ctx",
  "-np", "1",
  "--cont-batching",
  "--jinja",
  "--cache-prompt",
  "--cache-idle-slots",
  "--reasoning", "off",
  "--reasoning-budget", "0",
  "-ctk", "$CacheTypeK",
  "-ctv", "$CacheTypeV",
  "-t", "$Threads",
  "-tb", "$ThreadsBatch",
  "-b", "$Batch",
  "-ub", "$UBatch",
  "-fa", "$FlashAttn",
  "--prio", "$LlamaPrio",
  "--prio-batch", "$LlamaPrio",
  "--repeat-penalty", "1.0"
)

if ($Poll -ge 0) {
  $ServerArgs += @("--poll", "$Poll", "--poll-batch", "1")
}

if ($CacheReuse -gt 0) {
  $ServerArgs += @("--cache-reuse", "$CacheReuse")
}

if ($EnableNgramSpec -and $ModelsPreset -eq "") {
  $ServerArgs += @(
    "--spec-type", "ngram-cache",
    "--spec-ngram-mod-n-min", "1",
    "--spec-ngram-mod-n-max", "8",
    "--spec-ngram-mod-n-match", "24"
  )
}

& $Server @ServerArgs *> $Log
