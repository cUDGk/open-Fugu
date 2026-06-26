# 他LLMに渡す用プロンプト

あなたはローカルLLM実装とllama.cpp運用に詳しいエンジニアとして振る舞ってください。

以下のZIPは、Windows 11 Pro + AMD Ryzen 9 7940HS + 64GB DDR5-5600 + 1TB NVMe のミニPC上で、MoE系GGUFモデルを複数使い、Sakana Fugu風のローカルConductorを作るための設計・サンプル実装です。

目的は、3つのMoEモデルを「同時常駐・並列討論」させることではなく、llama.cpp router server の model hot-load / eviction を使い、`--models-max 1` で1モデルずつ逐次呼び出すことです。設計名は便宜上 `local-moe-fugu` です。

対象モデル案:

- Gemma 4 26B A4B: planner / public spec / synthesis / Japanese structure
- Qwen3.6 35B A3B: primary worker / coding / implementation / debugging
- GLM-4.7-Flash: verifier / critic / contradiction detector / repair instruction

制約:

- Windows 11 Pro
- Ryzen 9 7940HS, 8C/16T
- 64GB RAM
- no discrete GPU assumed
- Radeon 780M iGPU may be tested but must not be assumed to behave like large VRAM dGPU
- 3モデル同時常駐は避ける
- routerはLLMではなく決定的Python routingから始める
- GLM verificationはrisky taskのみ
- `fast`, default, `deep`, `ultra` の4モードを持つ
- OpenAI-compatible `/v1/chat/completions` frontdoorとして使う

あなたにしてほしいこと:

1. `src/conductor_7940hs.py` を読み、バグ・設計不整合・不足分を指摘してください。
2. llama.cpp router modeでモデルIDが合わない可能性があるため、HFタグ運用とローカルGGUF運用の両方で堅牢にする改善案を出してください。
3. 64GB RAM機でのctx, KV cache, quant, thread数, batch sizeの推奨値を、保守的/標準/品質優先の3段階で出してください。
4. 可能ならstreaming実装の追加案を提示してください。ただし内部draftやcritiqueはstreamしないでください。
5. code task向けに`tool_runner.py`をConductorに統合する方針を出してください。
6. 最終的に「このまま実装するなら何から直すべきか」を優先順位つきでまとめてください。

このプロジェクトの設計思想は `docs/ARCHITECTURE.md` と `docs/HARDWARE_PROFILE.md` にあります。
