$ErrorActionPreference = "Stop"

Write-Host "llama.cpp router /v1/models:"
curl.exe http://127.0.0.1:8080/v1/models

Write-Host "`nExpected config IDs:"
Write-Host "  google_gemma-4-26B-A4B-it-Q4_K_M"
Write-Host "  Qwen_Qwen3.6-35B-A3B-Q4_K_M"
Write-Host "  GLM-4.7-Flash-UD-Q4_K_XL"

Write-Host "`nfrontdoor health, if running:"
try {
  curl.exe http://127.0.0.1:9000/health
} catch {
  Write-Host "frontdoor is not running yet."
}
