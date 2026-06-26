$ErrorActionPreference = "Stop"

# Use this if you have local GGUF files under C:\llm\models.
# After startup, run:
#   curl http://127.0.0.1:8080/v1/models
# Then copy configs\moe_fugu_local_example.yaml and set exact exposed IDs.

$ModelsDir = "C:\llm\models"
$HostAddr = "127.0.0.1"
$Port = 8080
# -c はconfigの deep_ctx 以上にすること（conductorがdeep/ultraでdeep_ctx基準に
# 入力長を見積もるため）。下げる場合は configs/*.yaml の deep_ctx も合わせる。
$Ctx = 16384
$Threads = 12
$Batch = 512
$UBatch = 128

llama-server `
  --models-dir $ModelsDir `
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
