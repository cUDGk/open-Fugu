$ErrorActionPreference = "Stop"

Write-Host "llama.cpp router /v1/models:"
curl http://127.0.0.1:8080/v1/models

Write-Host "llama.cpp router /models if available:"
try {
  curl http://127.0.0.1:8080/models
} catch {
  Write-Host "GET /models not available or failed. /v1/models may still be enough."
}
