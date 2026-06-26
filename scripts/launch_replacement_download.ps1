$ErrorActionPreference = "Continue"

$Project = "C:\llm\local-moe-fugu-7940hs"
$Script = Join-Path $Project "scripts\download_replacement_models.ps1"
$TaskName = "LocalFuguModelDownload"

if (!(Test-Path $Script)) {
  throw "Download script not found: $Script"
}

Remove-Item -Force "C:\llm\replacement_download.log" -ErrorAction SilentlyContinue
Remove-Item -Force "C:\llm\replacement_models.DONE" -ErrorAction SilentlyContinue
schtasks.exe /End /TN $TaskName *> $null
schtasks.exe /Delete /TN $TaskName /F *> $null

$startTime = (Get-Date).AddMinutes(5).ToString("HH:mm")
$command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Script`""
schtasks.exe /Create /TN $TaskName /SC ONCE /ST $startTime /TR $command /F | Out-Null
schtasks.exe /Run /TN $TaskName | Out-Null

Write-Host "STARTED task=$TaskName"
Write-Host "LOG C:\llm\replacement_download.log"
Write-Host "DONE C:\llm\replacement_models.DONE"
