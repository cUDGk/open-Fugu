# Architecture: local-moe-fugu on Ryzen 9 7940HS / 64GB

## 目的

Sakana Fugu風の複数モデル協調を、64GB RAMのミニPCで成立させる。

本家Fugu風に寄せたい要素:

- model selection
- role delegation
- verification
- synthesis
- repair loop
- logs → route improvement

ただし、ハード制約上、3モデル同時常駐・同時fanoutは採用しない。

## 全体像

```text
OpenAI SDK / app
  ↓
http://127.0.0.1:9000/v1/chat/completions
model = local-moe-fugu | local-moe-fugu:fast | local-moe-fugu:deep | local-moe-fugu:ultra
  ↓
FastAPI frontdoor / conductor
  ↓
Deterministic exact solver
  ↓
0.6B hidden-state tower + rules fallback
  ↓
Workflow executor
  ↓
llama.cpp router server :8080
  --models-max 2
  ↓
HF-tagged or local GGUF MoE model
```

## なぜ逐次型か

Q4級の重みサイズ目安:

- Gemma 4 26B A4B Q4_K_M: 約17GB
- Qwen3.6 35B A3B Q4_K_M: 約22GB
- GLM-4.7-Flash Q4系: 約19GB

この3体だけで重みが約58GB前後になる。Windows、KV cache、llama-server、Python、mmap/page cacheが乗るため、64GBで3体同時常駐は危険。

したがって、現在の実運用は `llama-server --models-max 2` を使い、Gemma/Qwenのような頻出2体を温存しつつ、モデル呼び出し自体はconductor側のlockで逐次にする。`models-max=3` と低KV cacheはこのFUGU束ベンチでは遅く、採用しない。

## ワークフロー

### fast

```text
route → primary answer
```

用途:

- 軽い質問
- 短い説明
- 単発コード断片

### default / balanced

```text
route → primary answer → verifier if risky
```

用途:

- 通常利用
- コード/設計/検証語が含まれる場合のみGLM検証

### deep

```text
Gemma: public task spec
  ↓
Qwen: primary solution
  ↓
GLM: strict verification
  ↓
Qwen: repair if needed
```

用途:

- 実装設計
- 長めの仕様
- 失敗すると面倒なタスク

### ultra

```text
Gemma: public task spec
  ↓
Qwen: primary solution
  ↓
GLM: critique
  ↓
Gemma: synthesis
  ↓
GLM: final verification
  ↓
repair if major defect
```

用途:

- 重いアーキテクチャ設計
- 論点が多い比較
- 重要度が高いコード方針

## ルーティング思想

GLMをrouterとして毎回呼ぶと、それ自体がモデルロードを発生させる。64GB機では負荷に対する情報利得が低い。

現在は次の順で裁く:

1. 決定論で解ける日付・算術・文字数カウント等はモデルを呼ばずに返す。
2. 0.6B towerが使える時はQwen3-0.6B hidden states + seed headでagent/mode/taskを推定する。
3. 0.6B towerが落ちた場合は決定的ルールへフォールバックする。

ルール側の基本方針:

- coding/debugging → Qwen primary
- design/architecture/long prompt → deep mode
- verify/review/risk terms → GLM verification
- general/Japanese structured explanation → Gemma direct

ログが溜まったら、`src/bandit_router.py` のUCBでroute補正する。

## モデル役割

| Model | Role | Rationale |
|---|---|---|
| Gemma | planner/synthesizer | 公開仕様化、日本語構造化、最終整形 |
| Qwen | primary worker | コーディング、実装、debug、具体手順 |
| GLM | verifier/critic | 矛盾検出、major defect判定、repair instruction |

## 重要な禁止事項

```text
user task → 3モデル全員に同じ質問 → 多数決
```

これはメモリ・レイテンシ・品質のすべてで悪い。間違いが相関し、検証が弱く、長文コンテキストでコストが爆発する。

## 実装済み (v0.6.5)

- Streaming対応。内部draft/critiqueは流さず、final answerのみSSEで返す。
- backend usageを集計してfrontdoor responseに反映（以前は常に0）。
- ctx連動の入力トランスクリプト・クリップ（deep/ultraのctx溢れを防止）。
- `max_repair_rounds` をconfigから実際に使用（verify→repair→再verifyループ）。
- `/v1/models` 起動時チェック。backendの露出IDとconfigのmodel_idの不一致を警告（起動はブロックしない）。
- `system_fingerprint.performance` にactual/theoretical TPS、backend prefill/decode、overhead、model_callsを返す。
- `router_06b` による0.6B hidden-state routing。失敗時はrules routerへ安全にフォールバック。
- `models-max=2`、q8 KV、cache-prompt、cache-idle-slots、ngram-cacheを使うミニPC用launcher。
- 日本語一般はGemma、コード/実装はQwenへ寄せる言語/用途バイアス。
- 日付・算術・日本語/Latinカウント・狭い複合質問のdeterministic path。
- 短い追撃質問では直前会話のprimaryに寄せるfollowup affinity。
- tool_runner統合（オプトイン）。リクエストの `fugu_tools` でpytest/ruff/tsc等を実行し、
  失敗ログをGLMが構造化repair指示に変換 → Qwenが修復。許可コマンドは
  `tool_runner.ALLOWED_PREFIXES` のみで、指定が無ければtoolは一切走らない。

## 次の実装課題

1. 実測ログからrouter_06b seed headを継続再学習する。
2. Bandit routerで実測route補正。
3. tool_grounded repairの多段化。
4. 日本語generalの高速worker候補を追加検証する。
