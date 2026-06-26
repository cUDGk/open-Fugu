# local-moe-fugu-7940hs

このパッケージは、Windows 11 Pro + AMD Ryzen 9 7940HS + 64GB DDR5-5600 のミニPC上で、3つのMoE系GGUFモデルを **llama.cpp router server + 自作Conductor** で切り替えて使うための設計・サンプル実装です。

目的は、Sakana Fugu風の「複数モデル協調」を、ディスクリートGPUなしの64GB RAM機で成立させることです。並列討論型ではなく、1モデルずつホットロードして使う **sequential MoE-of-MoEs** を前提にしています。

## 対象ハードウェア

ユーザー指定の環境:

| 項目 | スペック |
|---|---|
| OS | Windows 11 Pro Build 26200 |
| CPU | AMD Ryzen 9 7940HS, 8C/16T |
| RAM | 64GB DDR5-5600 |
| SSD | 1TB NVMe |
| 本体 | Micro Computer (HK) Venus series mini PC |

設計上の重要点:

- Ryzen 9 7940HS は2メモリチャネル、DDR5-5600対応、Radeon 780M内蔵GPUを持つAPU。
- 64GB RAMでは30B級MoEモデル3体の同時常駐は避ける。
- llama.cpp router modeの `--models-max 1` を使い、同時ロードモデルを1つに制限する。
- Conductorは3モデルを同時に呼ばず、Gemma → Qwen → GLM のように逐次呼び出す。

## 推奨モデル役割

| 役割 | モデル | 用途 |
|---|---|---|
| Planner / Synthesizer | Gemma 4 26B A4B | 公開仕様化、構成整理、日本語長文、最終整形 |
| Primary Worker | Qwen3.6 35B A3B | 実装、デバッグ、手順分解、コード、リポジトリ推論 |
| Verifier / Critic | GLM-4.7-Flash | 検証、批判、矛盾検出、修復指示 |

## 主要ファイル

```text
configs/moe_fugu_hf.yaml              # HFタグ指定版。最初はこちら推奨。
configs/moe_fugu_local_example.yaml   # ローカルGGUFファイル/alias指定例。
src/conductor_7940hs.py               # OpenAI互換frontdoor + sequential conductor。
src/bandit_router.py                  # 任意: 実測ログからroute補正する簡易UCB router。
src/tool_runner.py                    # 任意: pytest/ruff/tsc等だけ許可する安全寄りrunner。
scripts/setup_windows.ps1             # Python/依存導入。
scripts/prime_model_cache.ps1         # llama.cppにHFモデルを一度読ませてcacheする。
scripts/start_llama_router.ps1        # llama-server router mode起動。
scripts/start_frontdoor.ps1           # FastAPI frontdoor起動。
scripts/smoke_test.ps1                # 動作確認。
examples/client_test.py               # OpenAI SDKからlocal-moe-fuguを叩く例。
docs/                                # 他LLMに渡すための設計文書。
```

## Quick Start

PowerShellを2つ開く想定です。

### 0. この実機での推奨ルート

今回のミニPC `<MINIPC-HOST>` には、すでに `C:\llm\llama\llama-server.exe`
と `C:\llm\models\*.gguf` が入っているため、HFタグ版ではなく
`configs/moe_fugu_minipc_local.yaml` を使う。

実機で確認したrouter model ID:

```text
google_gemma-4-26B-A4B-it-Q4_K_M
Qwen_Qwen3.6-35B-A3B-Q4_K_M
GLM-4.7-Flash-UD-Q4_K_XL
```

古い `C:\Users\user\start_server.ps1` は `-c 4096` で起動していたため、
deep/ultraの入力予算と合わない。`scripts/start_llama_router_minipc.ps1`
を使う。

### 1. セットアップ

```powershell
cd C:\llm\local-moe-fugu-7940hs
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

### 2. モデルcache作成

時間がかかります。モデルが巨大なのでSSD空き容量に注意してください。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prime_model_cache.ps1
```

### 3. llama.cpp router server起動

PowerShell 1:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_llama_router_minipc.ps1 -StopExisting
```

### 4. local-moe-fugu frontdoor起動

PowerShell 2:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_frontdoor_minipc.ps1
```

### 5. 動作確認

PowerShell 3:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1
```

またはPython:

```powershell
python .\examples\client_test.py
python .\scripts\smoke_frontdoor.py
```

## 呼び出しモデル名

frontdoor側では以下を使います。

```text
local-moe-fugu        # 通常。Primary 1回 + riskyなら検証。
local-moe-fugu:fast   # 1モデル直行。
local-moe-fugu:deep   # Gemma仕様化 → Qwen回答 → GLM検証 → 必要なら修復。
local-moe-fugu:ultra  # Gemma → Qwen → GLM → Gemma → GLM。遅い。
```

## tool実行（オプトイン）

code taskで実際にbuild/test/lintを走らせ、失敗ログ起点で修復させたい場合は、
リクエストbodyに `fugu_tools` を付ける。許可コマンドは `src/tool_runner.py` の
`ALLOWED_PREFIXES`（pytest / ruff check / npm test / tsc 等）のみ。指定が無ければ
toolは一切実行されない。

```json
{
  "model": "local-moe-fugu:deep",
  "messages": [{"role": "user", "content": "..."}],
  "fugu_tools": {
    "cwd": "C:\\path\\to\\repo",
    "commands": [["python", "-m", "pytest"], ["ruff", "check"]],
    "timeout_s": 120
  }
}
```

挙動: 回答生成後にコマンドを順次実行（並列なし）→ 失敗があればGLMが失敗ログを
構造化repair指示に変換 → Qwenが修復した最終回答を返す。conductorはworkspaceに
書き込まないため、toolはworkspaceの「観測」に使われる（修復は1回）。

## 設計上の結論

このハードでやるべきなのは、3モデル同時常駐ではありません。

```text
Bad:
  3つの30B級MoEを同時常駐
  3モデル全員に同じ質問
  多数決

Good:
  llama.cpp router mode
  --models-max 1
  逐次hot-load
  役割非対称
  risky taskだけ検証
  ログからroute補正
```

この構成なら、64GB RAM機でも「Fuguっぽい」協調挙動を狙えます。速度は期待しすぎない方がいいです。`deep` と `ultra` は実質バッチ処理寄りです。
