$ErrorActionPreference = "Stop"

Write-Host "Checking frontdoor health..."
curl http://127.0.0.1:9000/health

Write-Host "Checking frontdoor models..."
curl http://127.0.0.1:9000/v1/models

Write-Host "Sending a fast test request..."
$Body = @{
  model = "local-moe-fugu:fast"
  messages = @(
    @{ role = "user"; content = "このローカルFugu構成を一文で説明して。" }
  )
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Uri "http://127.0.0.1:9000/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $Body | ConvertTo-Json -Depth 12
