$ErrorActionPreference = "Stop"

# Conservative router-mode settings for Ryzen 9 7940HS / 64GB.
# Router mode is entered by not specifying -m.
# Already-cached HF models should be discoverable by llama.cpp.

$HostAddr = "127.0.0.1"
$Port = 8080
# -c はconfigの deep_ctx 以上にすること。conductorはdeep/ultraでdeep_ctx基準に
# 入力長を見積もるため、ここがそれ未満だとdeep/ultraでctx溢れする。
# ctxを下げたい場合は configs/*.yaml の deep_ctx も合わせて下げる。
$Ctx = 16384
$Threads = 12
$Batch = 512
$UBatch = 128

llama-server `
  --host $HostAddr `
  --port $Port `
  --models-max 1 `
  -c $Ctx `
  -np 1 `
  --cont-batching `
  --jinja `
  -ctk q8_0 `
  -ctv q8_0 `
  -t $Threads `
  -b $Batch `
  -ub $UBatch `
  --repeat-penalty 1.0
