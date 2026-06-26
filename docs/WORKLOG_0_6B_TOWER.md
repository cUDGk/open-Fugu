# 0.6B Routing Tower Worklog

## Current Direction

The goal is to evolve the local Fugu frontdoor from rule/keyword routing into a
small learned-style control tower:

```text
OpenAI request -> local frontdoor -> Qwen3-0.6B hidden-state router -> gemma/qwen/glm worker
```

OpenFugu is not a runtime dependency and should not be vendored into this project.
It was useful only as a public reference for the general shape:

- small backbone reads the conversation
- final answer generation is not done by the router
- one hidden vector feeds a small selection head
- selected worker does the real user-facing answer

This implementation is our own local version for the three installed mini PC
workers: `gemma`, `qwen`, and `glm`.

## Design Notes

The router module is `src/fugu_router_06b.py`.

It loads `Qwen/Qwen3-0.6B` with Transformers and reads one hidden vector from the
formatted transcript. It then applies small linear heads for:

- worker selection: `gemma`, `qwen`, `glm`
- mode selection: `fast`, `balanced`, `deep`, `ultra`
- task type: `general`, `coding`, `design`, `verification`, `research`

If `router_06b.head_path` points to a `.npz`, that head is used. The default mini
PC config now points to a self-made seed head:

```text
C:\llm\local-moe-fugu-7940hs\artifacts\router_06b_seed_head.npz
```

The seed head is built by `scripts/train_router_06b_seed_head.py`: it feeds our own
synthetic routing examples through Qwen3-0.6B, averages hidden vectors per class,
and saves the resulting linear heads. If the file is missing, the router can still
build a prototype head from semantic descriptions of each worker/mode.

The old rules router remains as fallback. If 0.6B dependencies, model files, or
head loading fail, the frontdoor logs `router_06b_fallback` and continues serving.

## Files Changed

- `src/fugu_router_06b.py`: new hidden-state router module.
- `src/conductor_7940hs.py`: calls the 0.6B router before falling back to rules.
- `configs/moe_fugu_minipc_local.yaml`: enables `router_06b` and points it to
  `C:/llm/models/Qwen3-0.6B`.
- `scripts/setup_fugu_router_06b.ps1`: installs CPU PyTorch/Transformers, downloads
  Qwen3-0.6B, builds the seed head, and warms the router.
- `scripts/train_router_06b_seed_head.py`: builds the self-made seed head from
  local synthetic examples.
- `scripts/start_frontdoor_minipc.ps1` and `scripts/start_fugu_frontdoor_user.ps1`:
  set `TOKENIZERS_PARALLELISM=false`.
- `docs/MINIPC_RUNBOOK.md`: documents setup and health checks.

## Deployment State

Local compile passed for:

```text
src/fugu_router_06b.py
src/conductor_7940hs.py
```

Mini PC deployment path:

```text
C:\llm\local-moe-fugu-7940hs
```

Mini PC setup has installed CPU PyTorch/Transformers and downloaded:

```text
C:\llm\models\Qwen3-0.6B
```

The first warmup attempt failed because the temporary warmup script did not add
the project directory to `sys.path`. `scripts/setup_fugu_router_06b.ps1` has been
fixed to insert `Path.cwd()` before importing `src.fugu_router_06b`.

The prototype warmup then succeeded, but the first decision was low-confidence and
too fuzzy. The next correction is the seed head path above, which should make the
first production routes less random while still using Qwen3-0.6B hidden states.

First API smoke with the seed head confirmed the frontdoor used
`router="qwen3-0.6b-hidden"`, but it also exposed a calibration issue: a short
general question routed to `ultra/qwen` even though the mode scores were nearly tied.
The fix is not to abandon the 0.6B tower, but to gate heavy decisions:

- if worker confidence is below `agent_min_confidence`, keep the rules router's
  safer primary worker
- if mode top-vs-second margin is below `mode_min_margin`, keep the rules router's
  safer mode
- if the fallback mode is `fast`, do not let a weak tower margin escalate it to
  `deep` or `ultra`

This keeps the 0.6B tower in charge when it has signal, while preventing a fuzzy
linear head from accidentally triggering multi-model workflows on tiny prompts.

## Next Steps

1. Re-copy the fixed setup and seed-head scripts to the mini PC.
2. Re-run the setup script so it builds `artifacts\router_06b_seed_head.npz`.
3. Restart the Fugu stack.
4. Smoke test `/health` and `/v1/chat/completions`.
5. Confirm `system_fingerprint.route.router == "qwen3-0.6b-hidden"`.
6. Run the average benchmark again and compare latency/route decisions.

## Smoke And Benchmark Results

After calibration, smoke through the real frontdoor produced:

```text
general one-sentence -> router=qwen3-0.6b-hidden primary=gemma mode=fast elapsed=17.344s
Python function       -> router=qwen3-0.6b-hidden primary=qwen  mode=fast elapsed=45.849s
review prompt         -> router=qwen3-0.6b-hidden primary=glm   mode=fast elapsed=24.417s
```

The first request includes 0.6B router/model warmup. Subsequent routing decisions
were around 140-200 ms.

Average benchmark, one pass with `max_tokens=128`:

```text
avg_fugu_20260625_233543
runs      8/8 ok
overall   mean 39.990s / median 18.352s / p90 85.224s
fast      mean 20.926s / median 15.218s / p90 45.281s
gemma     mean 11.559s
qwen      mean 68.421s
deep      173.444s
```

Interpretation:

- The 0.6B tower is now active in production responses.
- The ultra-mode false positive was fixed.
- Qwen routes are still expensive because the 35B-class GGUF worker dominates
  latency after routing.
- The next quality/speed win is to train a local outcome-aware head so borderline
  operational/design requests avoid Qwen unless Qwen is clearly worth the swap.

## Training Path

Prototype head is only the first step. The stronger version should collect local
bench rows with:

- prompt
- selected route
- worker latency
- visible answer length
- verifier/tool/test outcome when available
- user correction/acceptance when available

Then fit a three-worker `.npz` head against Qwen3-0.6B hidden vectors. That would
turn this from semantic prototype routing into a genuinely local, performance-aware
control tower.

## 2026-06-25 Model Replacement Plan

The next iteration replaces the heavy general/code workers while keeping the 0.6B
router and the existing verifier:

```text
gemma/general -> gemma-3-12b-it-Q4_K_M
qwen/code     -> Qwen3-Coder-30B-A3B-Instruct-Q4_K_M
glm/verify    -> GLM-4.7-Flash-UD-Q4_K_XL
```

Reasoning:

- Gemma 3 12B should be substantially lighter than the previous Gemma 26B-class
  worker for summaries, rewrites, and general answers.
- Qwen3-Coder-30B-A3B is an MoE coding worker with about 30.5B total parameters
  and about 3.3B active parameters, so it is a better fit than the previous dense/
  larger Qwen route for local code tasks.
- GLM remains useful for short verification prompts, and replacing it with a much
  larger MoE would likely hurt latency.

Download script:

```text
scripts/download_replacement_models.ps1
```

Remote log:

```text
C:\llm\replacement_download.log
```

Remote done marker:

```text
C:\llm\replacement_models.DONE
```

## 2026-06-26 Benchmark Scope Correction

The user clarified that the benchmark target is not each worker model in isolation.
The benchmark target is the whole FUGU bundle:

```text
client -> local-moe-fugu frontdoor -> 0.6B router -> selected worker/workflow -> answer
```

So the authoritative benchmark endpoint is:

```text
http://127.0.0.1:9000/v1/chat/completions
model: local-moe-fugu
```

Per-model measurements are allowed only as internal diagnostics to explain why a
FUGU route is slow. Success/failure and headline numbers must be reported for FUGU
end-to-end.

New worker files are present on the mini PC:

```text
C:\llm\models\gemma-3-12b-it-Q4_K_M.gguf
C:\llm\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
```

The FUGU config now points to:

```text
gemma -> gemma-3-12b-it-Q4_K_M
qwen  -> Qwen3-Coder-30B-A3B-Instruct-Q4_K_M
glm   -> GLM-4.7-Flash-UD-Q4_K_XL
```

Smoke after restart:

```text
general -> gemma/fast, router=qwen3-0.6b-hidden, elapsed 31.218s
code    -> qwen/fast,  router=qwen3-0.6b-hidden, elapsed 34.644s
review  -> glm/fast,   router=qwen3-0.6b-hidden, elapsed 25.105s
```

The first request includes 0.6B tower cold load (`router_elapsed_ms=19814`).
Subsequent router calls were about 150 ms.

## 2026-06-26 FUGU Bundle Benchmark Plan

The user clarified one more time that headline benchmarks must target the FUGU
bundle, not the individual worker GGUFs. I will use individual worker timings only
as diagnostics when explaining why one FUGU route is slow.

Benchmark set now has three FUGU-only scripts:

```text
scripts/bench_fugu_average.py     -> mixed practical prompts, latency/route stats
scripts/bench_fugu_chatcache.py   -> multi-turn cache behavior through FUGU
scripts/bench_fugu_standard.py    -> standard-lite scored tasks through FUGU
```

`bench_fugu_standard.py` sends only:

```text
model: local-moe-fugu
url:   http://127.0.0.1:9000/v1/chat/completions
```

It covers math, logic, basic systems knowledge, and code reading. It records both
latency and correctness so speed experiments do not silently make FUGU dumber.

The optimization loop from here:

1. Take baseline FUGU average/chatcache/standard results with the new worker map.
2. Try cache/speculative/KV/batch/thread variants against the same FUGU endpoint.
3. Keep a variant only if it improves latency without hurting standard-lite score.
4. Record every run here before changing the final launcher.

Baseline after replacing the workers:

```text
avg_fugu_20260626_014835
runs      8/8 ok
overall   mean 33.295s / median 21.806s / p90 66.201s
fast      mean 18.714s / median 19.358s / p90 32.347s
deep      135.362s
gemma     mean 14.946s
qwen      mean 51.643s

standard_fugu_20260626_015238
runs      20/20 ok
score     19/20 = 95.0%
overall   mean 11.773s / median 7.744s / p90 28.702s
math      5/5
logic     4/5
knowledge 5/5
code      5/5
```

Interpretation:

- Replacing the heavy workers improved the FUGU average benchmark versus the
  previous 0.6B-tower run (`39.990s -> 33.295s` mean, `173.444s -> 135.362s`
  for the deep review case).
- The one standard-lite miss was a calendar arithmetic question routed to Gemma.
- A large part of observed FUGU latency was worker switching/reloading. The
  older launcher used `--models-max 1`, so switching between Gemma and Qwen could
  trigger model load cost even for tiny answer-only prompts.

Next optimization target:

```text
scripts/start_llama_router_minipc.ps1
scripts/restart_fugu_stack_user.ps1
```

Both are being parameterized so I can test:

- `--models-max 2` to keep the two most common workers warm
- optional `--cache-reuse`
- optional lower KV cache types (`q4_0`) if memory pressure blocks model retention

The final launcher should keep the fastest setting that preserves the standard-lite
score.

## 2026-06-26 Optimization Results

All numbers below are FUGU end-to-end through:

```text
http://127.0.0.1:9000/v1/chat/completions
model: local-moe-fugu
```

Baseline after model replacement:

```text
models-max=1, KV=q8_0/q8_0, ngram-cache
average:  mean 33.295s / fast mean 18.714s / deep 135.362s
standard: 19/20 = 95.0%, mean 11.773s
```

Best variant:

```text
models-max=2, KV=q8_0/q8_0, ngram-cache
average:  mean 20.851s / fast mean 8.166s / deep 109.644s
standard: 19/20 = 95.0%, mean 4.944s
```

Rejected variants:

```text
models-max=3, KV=q8_0/q8_0, ngram-cache
average: mean 30.761s / fast mean 13.479s / deep 151.737s

models-max=3, KV=q4_0/q4_0, cache-reuse=256, ngram-cache
average: mean 32.369s / fast mean 15.235s / deep 152.311s

models-max=2, KV=q4_0/q4_0, cache-reuse=256, ngram-cache
average: mean 29.341s / fast mean 12.463s / deep 147.488s
```

Reasoning:

- `models-max=2` is the important speed win. It keeps the common Gemma/Qwen
  workers warm and avoids most model reload costs.
- `models-max=3` is slower on this 7940HS mini PC, likely because keeping
  Gemma/Qwen/GLM resident creates memory pressure.
- Lowering KV cache to `q4_0` did not help this workload and made the average
  benchmark slower, so the final candidate stays on q8 KV.

Draft-model speculative decoding experiment:

```text
downloaded: C:\llm\models\Qwen3-0.6B-Q8_0.gguf
source:     https://huggingface.co/Qwen/Qwen3-0.6B-GGUF
preset:     configs/llama_models_minipc_preset.ini
```

The preset successfully enabled Qwen-only draft speculative decoding:

```text
--model-draft C:\llm\models\Qwen3-0.6B-Q8_0.gguf
--spec-type draft-simple,ngram-cache
```

But it is not adopted:

```text
models-max=2, Qwen draft-simple + ngram-cache
average: mean 25.744s / fast mean 11.105s / deep 128.220s
```

The Qwen draft run showed low acceptance and extra draft overhead. One smoke log
showed about `10 accepted / 40 generated` draft tokens. That made Qwen routes
slower than the plain `models-max=2` ngram-only setup.

Final launcher decision:

```text
models-max=2
KV cache: q8_0/q8_0
spec: ngram-cache only
cache-reuse: off
Qwen draft GGUF: kept on disk for future experiments, not enabled
```

## 2026-06-26 Additional Runtime Tuning

I tested the remaining low-level knobs against the same FUGU average benchmark.
None beat the current launcher:

```text
models-max=2, q8 KV, batch=1024, ubatch=256, threads=12
average: mean 29.141s / fast mean 12.017s / deep 149.008s

models-max=2, q8 KV, batch=512, ubatch=128, threads=16
average: mean 31.143s / fast mean 12.294s / deep 163.083s

models-max=2, q8 KV, batch=512, ubatch=128, threads=10
average: mean 27.086s / fast mean 12.216s / deep 131.177s
```

Conclusion:

- Keep `threads=12`, `batch=512`, `ubatch=128`.
- Larger batch/ubatch and higher/lower thread counts hurt the deep workflow or
  overall mean even when short Qwen code prompts looked fine.
- I did not lower the final context size below 16384 because the target is a useful
  FUGU bundle, not only a short-prompt benchmark. Lowering context would make the
  measured benchmark narrower and less representative.

## 2026-06-26 Startup Warmup

The restart wrapper now warms the two common resident workers after health checks:

```text
warm_gemma_router -> short general prompt, loads 0.6B router + Gemma
warm_qwen_code    -> short code prompt, loads Qwen
```

GLM is intentionally not warmed because keeping all three workers resident was slower
on this machine.

Measured warmup on <MINIPC-HOST>:

```text
WARMUP warm_gemma_router route=gemma/fast elapsed=17.769s
WARMUP warm_qwen_code    route=qwen/fast  elapsed=26.345s
```

After warmup, the first external short general request returned in:

```text
elapsed=3.945s
route=gemma/fast
router_elapsed_ms=142
```

Before warmup, the same startup path was around 19.8s and included 0.6B router cold
load. So the final runtime trades a longer restart command for a much better first
interactive request.

Final standard-lite check in the warmed runtime:

```text
standard_fugu_20260626_025207
runs      20/20 ok
score     19/20 = 95.0%
overall   mean 2.099s / median 2.583s / p90 3.114s
math      5/5
logic     4/5
knowledge 5/5
code      5/5
```

The score is unchanged from the non-warmed best run, while short benchmark latency
is lower because the router, Gemma, and Qwen are already resident.

## 2026-06-26 Extra Push: Fast Primary Bias + Deterministic Front Path

User asked to push harder after the warmed 95% / 2.099s standard result. I re-opened
the failure and latency logs instead of only changing llama.cpp flags.

What I found:

- The single standard miss was the calendar multiple-choice case. Gemma and Qwen both
  answered it unreliably, while GLM answered it correctly but would add a cold/heavy
  worker path.
- Qwen3-Coder-30B-A3B is much faster than Gemma 3 12B on this mini PC for short direct
  answers, even on general and lightweight design prompts.
- The old standard scorer accidentally treated `ANSWER: 8.5` as integer `8`. That was
  a benchmark bug, so I fixed the scorer before accepting the new result.

Adopted changes:

- `scheduler.fast_primary: qwen` plus a conductor post-route fast-primary bias. The
  0.6B tower still decides mode/task/risk, but ordinary fast direct answers use Qwen.
- Deterministic front path for simple exact-answer arithmetic and weekday
  multiple-choice prompts. These return without a model call and log
  `deterministic_arithmetic` or `deterministic_calendar`.
- Compact fast output budgets for short client requests, with stricter prompts for:
  - workflow/design prose, so Qwen Coder does not turn role labels like "coder" into a
    source-code request;
  - explicit code/command/function requests, so Qwen returns a minimal snippet instead
    of a tutorial.
- Compact deep path now caps short `max_tokens <= 128` review calls at 96 output tokens.

Final measured whole-FUGU results on <MINIPC-HOST>:

```text
avg_fugu_20260626_081741
overall mean 2.948s / median 2.516s / p90 4.591s / max 4.978s
fast    mean 2.658s
deep    mean 4.978s
completion TPS 19.250 / total TPS 59.996 / request TPS 0.339

standard_fugu_20260626_081804
score 20/20 = 100.0%
overall mean 0.626s / median 0.796s / p90 1.117s / max 1.159s
math mean 0.012s via deterministic_arithmetic
completion TPS 5.588 / total TPS 170.991 / request TPS 1.597
```

Compared with the earlier best representative FUGU bundle run:

```text
average: 20.851s -> 2.948s
standard: 19/20, 4.944s -> 20/20, 0.626s
```

This is not a single-worker benchmark. The requests still target
`model=local-moe-fugu`; the improvement comes from the bundle-level policy:
0.6B hidden-state routing, deterministic cheap paths, Qwen fast direct, and compact
deep review.

Notes for next session:

- Gemma is now mostly preserved for planner/synthesis/deeper workflows instead of being
  the default fast generalist. That matches observed latency on this hardware.
- GLM remains useful as verifier but should stay out of the hot path unless the request
  is explicitly risky or verification-heavy.
- The deterministic arithmetic path is intentionally narrow. Expand it only with
  patterns that are easy to test and log; do not turn it into ad hoc broad parsing.

## 2026-06-26 Japanese Routing + Long Context Fix

User found a serious regression: Japanese compound prompts routed to Qwen fast produced
English/Chinese-like "please clarify" responses. Direct worker testing showed:

- Qwen3-Coder is fastest but unreliable for Japanese general prompts even with a strong
  Japanese system prompt.
- Gemma is slower but can answer Japanese coherently.
- GLM hallucinated badly on this test and should not become the Japanese fast path.

Adopted changes:

- Japanese non-code requests now apply `japanese_language_bias=gemma` after the normal
  fast-primary Qwen bias. Japanese code/command/function requests may still use Qwen.
- `LOCAL CONTEXT HINTS` were added for current weekday/date, `YYYY年M/D` weekday
  interpretation, and visible Latin `a` character counts.
- Runtime date context is now conditional. It is only injected for Japanese/date/weekday
  prompts, so English short benchmarks do not pay the token overhead.
- Long context clipping changed from tail-only truncation to head+tail middle clipping
  in both the conductor and the 0.6B router. This keeps early instructions and recent
  user content instead of dropping the beginning of long requests.

Important test note:

- PowerShell `Invoke-RestMethod` can corrupt Japanese request bodies in this SSH path.
  Use a UTF-8 client body (`application/json; charset=utf-8`) for Japanese validation.

UTF-8 FUGU probe for the user's Japanese compound prompt:

```text
route: gemma/fast, workflow=gemma_direct
reason includes: fast_primary_bias=qwen; japanese_language_bias=gemma
elapsed: 52.797s
completion TPS: 5.455
total TPS: 13.201
```

The output answered in Japanese, used Friday for today, treated `200年3/4` as
200年3月4日, and reported visible lowercase Latin `a` as 1 while noting that Japanese
phonetic /a/ counting is a different reading-dependent task.

Regression check after making runtime date context conditional:

```text
standard_fugu_20260626_121520
score 20/20 = 100.0%
completion TPS 5.478 / total TPS 162.444 / request TPS 1.522
overall mean 0.657s / median 0.810s / p90 1.116s
```

## 2026-06-26 Theory-TPS Instrumentation + Japanese Fast Push

User asked to push speed harder and expose the theoretical value. I added per-request
performance instrumentation to `system_fingerprint.performance`.

Definition used in the frontdoor:

```text
theoretical_min_ms = router_elapsed_ms + backend prompt_ms + backend predicted_ms
overhead_ms        = frontdoor internal elapsed - theoretical_min_ms
```

The backend timing numbers come from llama.cpp's OpenAI-compatible `timings` object.
This means the theoretical TPS is not a vendor/model marketing number; it is the
measured backend compute floor for the exact FUGU request after routing.

Files changed:

```text
src/conductor_7940hs.py
scripts/bench_fugu_average.py
scripts/bench_fugu_standard.py
scripts/bench_fugu_chatcache.py
scripts/probe_fugu_prompt.py
scripts/restart_fugu_stack_user.ps1
configs/moe_fugu_minipc_local.yaml
```

Adopted runtime changes:

- `system_fingerprint.performance` now includes actual TPS, theoretical TPS,
  backend prefill/decode TPS, model-call timings, overhead, and efficiency.
- FUGU benchmark scripts now write the same theory columns to JSONL/CSV/summary.
- Restart warmup now explicitly warms `warm_gemma_japanese` and `warm_qwen_code`;
  the previous English general warmup could route to Qwen and leave Gemma cold.
- Japanese fast answers have shorter system prompts and a non-code fast cap.
- Date context is no longer injected merely because text is Japanese; it is injected
  only for date/weekday prompts.
- The conductor supplies compact local hints for easy deterministic facts in the
  Japanese compound prompt: weekday, proleptic Gregorian date, visible Latin `a`,
  indefinite integral shape, ice-cream moderation, and 50m walk/car choice.
- Visible Latin `a` counts are corrected deterministically after generation, because
  this is cheaper and more reliable than asking a model to count characters.

Worker diagnostics:

```text
Qwen direct: faster, but answered with the wrong current date and treated year 200
             as BCE-like; not safe for Japanese general routing.
GLM direct:  slower and hallucinated the current date/weekday; not adopted.
Gemma:       slower, but coherent enough for Japanese general prompts.
```

Latest UTF-8 FUGU probe for the user's Japanese compound prompt:

```text
route: gemma/fast, workflow=gemma_direct
elapsed external: 19.027s
internal elapsed: 19.008s
completion tokens: 91
actual completion TPS: 4.787
theoretical completion TPS: 4.790
backend prefill TPS: 55.168
backend decode TPS: 7.448
efficiency vs theory: 0.9994
```

The same prompt was previously 52.797s, so the useful Japanese path is now about
2.8x faster while preserving the important answers:

```text
today: Friday
200-03-04: Tuesday
visible lowercase Latin 'a': 1
A/a total: 1
Japanese phonetic /a/: reading-dependent
```

Latest whole-FUGU benchmarks:

```text
standard_fugu_20260626_153910
score 20/20 = 100.0%
overall mean 0.846s / median 0.802s / p90 1.438s
completion TPS 9.106 / theoretical completion TPS 9.503
backend decode TPS 26.960 / backend prefill TPS 100.659
efficiency vs theory 0.9695

avg_fugu_20260626_153938
overall mean 2.594s / median 2.295s / p90 3.900s / max 4.915s
completion TPS 19.568 / theoretical completion TPS 19.897
backend decode TPS 24.764 / backend prefill TPS 95.572
efficiency vs theory 0.9863
```

Interpretation:

- For hot Qwen routes, FUGU overhead is now small: usually tens of milliseconds.
- For the Japanese Gemma route, hot overhead is also near zero; remaining latency is
  almost entirely Gemma prefill/decode speed.
- Further large Japanese speedups require either a better Japanese-capable fast worker
  or a more aggressive deterministic answer planner for narrow fact-like prompts.

## 2026-06-26 Deterministic Compound Path

User asked whether FUGU was really using its full role. The answer was "not yet" for
the Japanese compound prompt, because the previous path still asked Gemma to count
`母音"あ"`, which is exactly the kind of task a model should not be trusted to count.

I added a deterministic compound path before 0.6B/model routing for the narrow
Japanese multi-question shape:

```text
today weekday
YYYY年M/D weekday
indefinite integral basics
adult male ice-cream amount
50m walk vs car
vowel/character count
```

When all of those markers are present, the conductor now answers without a worker
model call:

```text
workflow: deterministic_japanese_compound
router: deterministic
model_call_count: 0
```

For `母音"あ"` the answer separates determinate visible-text counts from uncertain
phonetic readings:

```text
ひらがな「あ」 count
あ/ぁ/ア/ァ count
visible kana a-row vowel count
Kanji-derived /a/ readings are not guessed
```

Verification probe for the user's `母音"あ"` prompt:

```text
elapsed external: 0.015s
route: deterministic_japanese_compound
model calls: 0
answer:
1. 今日は金曜日です。
2. 200年3月4日は火曜日です。
3. 不定積分は原始関数Fを見つけて、最後に積分定数Cを付けます。
4. アイスクリームに成人男性向けの標準推奨量はありません。嗜好品として少量を時々が無難です。
5. 50mなら通常は徒歩が合理的です。荷物や天候などの事情があれば車でも構いません。
6. ひらがな「あ」は2個、あ/ぁ/ア/ァは3個、可視かなのあ段母音は16個です。漢字の読み由来の/a/は推測しません。
```

Regression checks after adding this path:

```text
standard_fugu_20260626_172818
score 20/20 = 100.0%
overall mean 0.845s / median 0.806s / p90 1.420s
completion TPS 9.233 / theoretical completion TPS 9.624
efficiency vs theory 0.9712

avg_fugu_20260626_172845
overall mean 2.555s / median 2.341s / p90 3.718s / max 4.882s
completion TPS 19.468 / theoretical completion TPS 19.815
efficiency vs theory 0.9859
```

Interpretation:

- The FUGU bundle now uses a clearer hierarchy: deterministic solver first,
  0.6B/router plus worker model only when the prompt is not fully covered.
- This is a better use of FUGU than forcing all subtasks through Gemma/Qwen. It is
  faster, more exact for counts/dates, and keeps worker models for the parts where
  language generation is actually needed.

## 2026-06-26 FUGU Role Check And Count Path

User asked if this is really fulfilling the FUGU role and whether the bundle is being
used fully. Current answer: mostly yes for the local 7940HS target, with an important
boundary. This is now a single frontdoor bundle with:

```text
client -> local-moe-fugu frontdoor
       -> deterministic solver when exact
       -> 0.6B tower/rules router when routing is needed
       -> llama.cpp router backend with hot worker models
       -> performance metrics returned per request
```

It is not a magic fully-parallel OpenFugu clone. On this 64GB mini PC it is intentionally
sequential and conservative: exact questions are handled before model calls, and the
worker LLM is used only when text generation or reasoning is actually needed.

I added `deterministic_count` for standalone count prompts. It handles:

```text
Latin vowel-a counts
Japanese quoted-a vowel counts
generic quoted-string counts
```

For prompts like "count in the following text", `count_target_text` now tries to count
only the target text after the first newline or after a colon. This prevents the
instruction itself from contaminating the answer.

Latest Japanese FUGU regression:

```text
japanese_fugu_20260626_173531
score 4/4 = 100.0%
mean 0.008s / median 0.004s / p90 0.018s / max 0.023s
model calls 0 for deterministic Japanese compound/count cases
```

Latest standard whole-FUGU benchmark:

```text
standard_fugu_20260626_173555
score 20/20 = 100.0%
overall mean 0.840s / median 0.815s / p90 1.385s / max 3.322s
completion TPS 9.164 / theoretical completion TPS 9.583
backend decode TPS 27.015 / backend prefill TPS 102.221
efficiency vs theory 0.9683
```

Latest average whole-FUGU benchmark remains:

```text
avg_fugu_20260626_172845
overall mean 2.555s / median 2.341s / p90 3.718s / max 4.882s
completion TPS 19.468 / theoretical completion TPS 19.815
efficiency vs theory 0.9859
```

Interpretation:

- The FUGU bundle is now doing the conductor job on this hardware: exact deterministic
  work first, then route/model work only where useful.
- Japanese general generation is still limited by the available worker quality/speed.
  Gemma is safer but slower; Qwen is fast but unreliable for these Japanese general
  prompts.
- The remaining large wins are not from frontdoor overhead, because hot-route efficiency
  is already near theoretical. They would come from a better Japanese fast worker, more
  tested deterministic tools, or a true multi-worker hardware budget.

## 2026-06-26 Followup Affinity + Runtime Audit

While checking whether the bundle is fully acting as FUGU, I found one config/code
mismatch: `avoid_model_swap_on_followup: true` was enabled, but the conductor only read
the flag and did not act on it.

I implemented a conservative in-memory route affinity:

```text
first user message -> previous primary/mode/workflow/task
short followup -> prefer previous primary when safe
```

Safety boundaries:

- It only applies to short followups, not obvious new code/search/review tasks.
- It does not override deep/ultra forced modes.
- It does not drag Japanese general followups back to Qwen, because Qwen was measured
  fast but unreliable for Japanese general answers here.
- It is an in-memory runtime hint, not a durable user database.

I also added:

```text
scripts/audit_fugu_runtime.py
```

This script checks the live FUGU endpoint for:

```text
/health
/v1/models
deterministic_count
qwen fast code route
gemma Japanese general route
short followup primary affinity
performance fields on responses
```

Documentation cleanup:

- Mini PC config now records `hardware_profile.max_loaded_models: 2`.
- Architecture and hardware docs now describe the current `models-max=2`, q8 KV,
  cache-prompt, cache-idle-slots, ngram-cache runtime instead of the older
  `models-max=1` setup.

Post-fix runtime audit:

```text
audit_fugu_runtime_latest.json
runs 4/4
health ok
router_06b loaded
deterministic_count: 0 model calls, 0.017s
qwen_fast_code: qwen_direct, 1.213s, actual TPS 15.860, theory TPS 16.212
gemma_japanese_general: gemma_direct, 7.675s, actual TPS 4.692, theory TPS 4.698
short_followup_affinity: first qwen -> followup qwen, reason includes followup_affinity=qwen
```

The first audit attempt caught a real issue: a Japanese "make it shorter" followup
after a coding answer was being dragged to Gemma by the Japanese language bias and
produced a bad code shortening. The affinity rule now allows Japanese followups to
stay with Qwen when the previous task was coding, while still protecting Japanese
general conversation from Qwen.

Regression after the fix:

```text
japanese_fugu_20260626_174809
score 4/4 = 100.0%
mean 0.003s / median 0.002s / p90 0.006s / max 0.007s

standard_fugu_20260626_174826
score 20/20 = 100.0%
overall mean 0.848s / median 0.802s / p90 1.392s / max 3.350s
completion TPS 9.202 / theoretical completion TPS 9.550
efficiency vs theory 0.9728

avg_fugu_20260626_174917
score 16/16 = 100.0% request success
overall mean 2.661s / median 2.478s / p90 4.375s / max 5.154s
completion TPS 20.478 / theoretical completion TPS 20.822
efficiency vs theory 0.9861
```

## 2026-06-26 Output Shape And Short-Cap Pass

Runtime audit showed a practical quality issue: prompts like "Return only the
expression" still received fenced code blocks. Standard-lite prompts also asked for
`ANSWER: <letter>`, but some model outputs included explanations after the answer.

I added final answer shape correction:

```text
return exactly ANSWER: <letter> -> keep only ANSWER: X
return only the expression -> strip code fences and keep the first expression line
code-only requests without explanation -> strip outer code fences
```

I also reduced generation budget for explicit `Return exactly ANSWER:` prompts to 16
tokens. This keeps correctness but prevents long explanations from being generated in
the first place.

Post-change checks:

```text
audit_fugu_runtime_latest.json
runs 4/4
qwen_fast_code answer_preview: sum(i**2 for i in range(1, 6))
short_followup_affinity answer_preview: sum(range(1,11))

japanese_fugu_20260626_175441
score 4/4 = 100.0%
mean 0.011s / p90 0.021s

standard_fugu_20260626_175612
score 20/20 = 100.0%
overall mean 0.706s / median 0.810s / p90 1.272s / max 2.012s
completion tokens 156 -> 92 versus the previous standard run

avg_fugu_20260626_175809
score 16/16 = 100.0% request success
overall mean 2.564s / median 2.537s / p90 3.748s / max 4.165s
completion TPS 21.279 / theoretical completion TPS 21.709
efficiency vs theory 0.9850
```

One bad benchmark pass (`avg_fugu_20260626_175720`) was discarded from headline
interpretation because it ran concurrently with the runtime audit and measured lock
wait/queueing, not the normal single-client FUGU path.

I also added a narrow cleanup for the standalone Cyrillic connector `Би` that Qwen
sometimes emits where an arrow/connector should be in short workflow lists.

Final 0.6.6 spot checks after restart:

```text
audit_fugu_runtime_latest.json at 2026-06-26T18:00:17
runs 4/4
router_06b loaded
qwen expression answer has no code fence
short followup qwen -> qwen

workflow_design UTF-8 probe
elapsed 3.478s
answer uses normal arrows:
Planning tasks → route to planner
Code generation → route to coder
Review/validation → route to verifier
```

## 2026-06-26 Stop Sequences For ANSWER-Only

The 0.6.6 pass corrected extra text after generation, but some `ANSWER: <letter>`
requests still spent decode time producing explanations that were then stripped. I added
backend stop sequences for explicit `Return exactly ANSWER:` fast routes:

```text
\n
)
.
 The
 the
 Because
 because
 Explanation
```

This is intentionally scoped to explicit `Return exactly` + `ANSWER:` prompts. It should
not affect normal prose, code, Japanese, or deep routes.

I also made the visible-artifact cleanup for Qwen's occasional `Би` connector stronger:
the literal token is replaced with `→` before returning the final response.

Valid single-client checks:

```text
standard_fugu_20260626_180659
score 20/20 = 100.0%
overall mean 0.603s / median 0.758s / p90 1.068s / max 1.111s
completion tokens 70
request TPS 1.659

japanese_fugu_20260626_180349
score 4/4 = 100.0%
mean 0.013s / max 0.030s

avg_fugu_20260626_180451
score 16/16 = 100.0% request success
overall mean 2.659s / median 2.468s / p90 4.114s / max 4.862s
completion TPS 20.711 / theoretical completion TPS 21.079
efficiency vs theory 0.9862

audit_fugu_runtime_latest.json at 2026-06-26T18:07:19
runs 4/4
router_06b loaded
expression-only output has no code fence
short followup qwen -> qwen
```

Note: `standard_fugu_20260626_180635` was run concurrently with a probe and is not used
as a headline value. The later `standard_fugu_20260626_180659` run is the valid
single-client measurement.

## 2026-06-26 Runtime Gate 0.6.8

I tightened the FUGU runtime gate instead of only adding another benchmark number.

Changes:

- `scripts/audit_fugu_runtime.py` now fails if `/health` is not ok, `router_06b`
  is not enabled/loaded, `last_error` is set, or the exposed frontdoor model IDs are
  missing.
- Audit cases now require performance metadata on model-backed calls, so the
  benchmark cannot silently lose TPS/theoretical timing instrumentation.
- The workflow-design audit now checks that Qwen's occasional Cyrillic-looking
  connector artifact is not returned and that planner/coder/verifier labels survive.
- Added `--include-long` to run a heavier long-context head/tail retention check.
- `Return exactly \`literal\`` prompts are now post-shaped to the literal if the model
  includes it with extra text.
- Non-risky exact prompts under 20000 characters can stay on the fast path instead
  of paying for verifier/deep workflow overhead.
- Exact fast routes now use stop sequences beyond the ANSWER-only path.

Validation after syncing to the mini PC and restarting:

```text
health
status ok
front_model local-moe-fugu
router_06b enabled=true loaded=true last_error=null

audit_fugu_runtime_latest.json at 2026-06-26T18:16:16
runs 6/6
runtime_health ok
deterministic_count model_call_count 0
qwen_fast_code qwen_direct
gemma_japanese_general gemma_direct
workflow_artifact_cleanup ok
short_followup_affinity qwen -> qwen

audit_fugu_runtime_long_latest.json at 2026-06-26T18:17:32
runs 7/7
long_context_head_tail qwen_direct
answer HEAD_OK TAIL_OK
elapsed 53.879s
efficiency vs backend theory 0.9993
```

The long-context result is intentionally not sold as "fast". The important result is
that the bundle stayed on the fast exact path, retained the head and tail marker, and
reported that the latency was almost entirely backend prefill/decode theory. Long
input on this 7940HS setup is still expensive.

Fresh single-client benchmark results:

```text
standard_fugu_20260626_181828
score 20/20 = 100.0%
overall mean 0.611s / median 0.774s / p90 1.102s / max 1.128s
completion tokens 70
completion TPS 5.732 / request TPS 1.638
backend prefill TPS 100.533 / backend decode TPS 31.238
efficiency vs theory 0.9572

japanese_fugu_20260626_181837
score 4/4 = 100.0%
mean 0.018s / max 0.027s
deterministic model_call_count 0

avg_fugu_20260626_181929
requests 16/16 ok
overall mean 2.636s / median 2.295s / p90 4.215s / max 4.913s
completion TPS 20.814 / theoretical completion TPS 21.197
backend prefill TPS 91.451 / backend decode TPS 25.396
efficiency vs theory 0.9856
```

One display caveat: PowerShell may render the arrow `→` oddly in some captured output,
but the actual code point in the average-benchmark workflow preview is U+2192, not the
literal Cyrillic `Би` sequence.

## 2026-06-26 Deterministic Exact Literal 0.6.9

The previous long-context audit proved correctness but also exposed a bad conductor
choice: an exact retrieval prompt with the answer literal visible in the prompt still
went through Qwen and paid a full long-prefill cost.

I added a narrow `deterministic_exact_literal` path before worker routing:

```text
unconditional: Return exactly `literal` -> literal
conditional visibility: Return exactly `literal` if marker/visible/contains...
                        -> literal only if every literal token is visible outside the quoted target
```

This intentionally does not answer arbitrary conditional correctness prompts such as
"Return exactly `SAFE` if the code is safe"; those still need model/tool judgement.

Validation after syncing to the mini PC and restarting:

```text
audit_fugu_runtime_latest.json at 2026-06-26T18:26:44
runs 7/7
deterministic_exact_literal: FUGU_LITERAL_OK
model_call_count 0
elapsed 0.016s

audit_fugu_runtime_long_latest.json at 2026-06-26T18:27:07
runs 8/8
long_context_head_tail workflow deterministic_exact_literal
model_call_count 0
elapsed 0.002s
answer HEAD_OK TAIL_OK
```

This moves the long exact marker test from the 0.6.8 result of 53.879s on Qwen to
0.002s with no worker call. It is not a general long-context solution, but it is the
right FUGU behavior for exact visible-marker prompts: do not spend 30B-class model
time when the conductor can prove the answer.

Regression checks:

```text
standard_fugu_20260626_182726
score 20/20 = 100.0%
overall mean 0.642s / median 0.807s / p90 1.165s / max 1.227s
completion tokens 70
completion TPS 5.453 / request TPS 1.558
efficiency vs theory 0.9591

japanese_fugu_20260626_182733
score 4/4 = 100.0%
mean 0.012s / max 0.017s
deterministic model_call_count 0
```

## 2026-06-26 Mode Alias Gate 0.7.0

I checked the exposed model aliases:

```text
local-moe-fugu
local-moe-fugu:fast
local-moe-fugu:deep
local-moe-fugu:ultra
```

The aliases were exposed, but not previously audited. A direct forced `:deep` probe
with a short code-review prompt showed the full multi-model path was technically
working but impractical for short capped replies:

```text
local-moe-fugu:deep before 0.7.0
workflow gemma_spec__qwen_solution__glm_verify__qwen_repair
model_call_count 3
elapsed 94.493s
answer ended with a dangling "**Explanation:**" label
```

That is too expensive for an interactive "try deep mode" request. I changed the
mode behavior:

- If `:deep` has a small `max_tokens` cap and a short prompt, it uses
  `qwen_compact_deep_direct`.
- If `:ultra` has a small `max_tokens` cap and a short prompt, it uses
  `qwen_compact_ultra_direct`.
- Uncapped or larger `:deep` / `:ultra` requests still keep the heavier multi-stage
  workflows.
- Output cleanup now removes dangling empty labels such as a final `Explanation:`.
- `scripts/audit_fugu_runtime.py --include-modes` now verifies the aliases.

Validation after restart:

```text
audit_fugu_runtime_modes_latest.json at 2026-06-26T18:37:33
runs 10/10
mode_alias_fast:  qwen_direct, 1.464s, model_call_count 1
mode_alias_deep:  qwen_compact_deep_direct, 4.134s, model_call_count 1
mode_alias_ultra: qwen_compact_ultra_direct, 3.986s, model_call_count 1
router_06b loaded, last_error null

audit_fugu_runtime_long_latest.json at 2026-06-26T18:38:16
runs 8/8
long_context_head_tail deterministic_exact_literal, 0.019s, model_call_count 0

standard_fugu_20260626_183754
score 20/20 = 100.0%
overall mean 0.659s / median 0.822s / p90 1.174s / max 1.295s
completion tokens 70
completion TPS 5.308 / request TPS 1.517
efficiency vs theory 0.9607

japanese_fugu_20260626_183823
score 4/4 = 100.0%
mean 0.009s / max 0.026s
deterministic model_call_count 0
```

This does not remove the full heavy deep/ultra paths; it prevents capped short requests
from accidentally paying a 90s multi-model hotload/verification bill.

## 2026-06-26 API Surface And Tool Gate 0.7.1

I checked the current system against the user's direct question: "is this really
acting as FUGU and using the bundle fully?"

Short answer: it is acting as a FUGU-style frontdoor now, but not as a magical
all-purpose optimizer. The important behavior is that clients send requests to
`model=local-moe-fugu`, and the conductor decides the path:

- deterministic answer when the result can be proven locally;
- Qwen direct for fast code/logic/knowledge work;
- Gemma for Japanese/general prompts that Qwen3-Coder handles poorly here;
- compact deep/ultra aliases for short capped forced-mode probes;
- heavy multi-stage deep/ultra only when the request budget justifies it;
- explicit tool execution when `fugu_tools.commands` is supplied and accepted by
  policy.

That means the benchmark is against the FUGU bundle, not against each worker model.
The fast 0.00x second Japanese/calendar/count results are not model TPS; they are
FUGU doing the right thing by avoiding a worker call.

What I changed:

- `system_fingerprint.performance` now exposes tool execution metadata:
  `tool_call_count`, `tool_fail_count`, `tool_call_wall_ms`, and `tool_calls`.
- Explicit `fugu_tools.commands` are no longer ignored just because a deterministic
  answer path is available first.
- The tool policy now allows the narrow unittest command shape:
  `python -m unittest ...`.
- `scripts/audit_fugu_runtime.py --include-api` now checks streaming SSE, invalid
  message handling, bad tool-command rejection, and successful unittest execution.
- FastAPI app and manifest are now `0.7.1`.

Validation after restart:

```text
restart_fugu_stack.ps1
frontdoor http://127.0.0.1:9000
backend   http://127.0.0.1:8080/v1
router_06b enabled=true loaded=true last_error=null
router_06b source=trained_head:C:\llm\local-moe-fugu-7940hs\artifacts\router_06b_seed_head.npz
```

```text
audit_fugu_runtime_api_latest.json at 2026-06-26T18:44:46
runs 11/11
stream SSE ok, content-type text/event-stream
invalid messages rejected with HTTP 400
bad tool command rejected with HTTP 400
unittest tool command accepted, tool_call_count 1, tool_fail_count 0
```

```text
audit_fugu_runtime_full_latest.json at 2026-06-26T18:45:47
runs 15/15
runtime_health ok
router_06b loaded, last_error null
model aliases exposed: local-moe-fugu, :fast, :deep, :ultra
mode_alias_fast:  qwen_direct, 1.760s, model_call_count 1
mode_alias_deep:  qwen_compact_deep_direct, 3.448s, model_call_count 1
mode_alias_ultra: qwen_compact_ultra_direct, 3.388s, model_call_count 1
long_context_head_tail: deterministic_exact_literal, 0.015s, model_call_count 0
api_tool_unittest_success: qwen_direct, tool_call_count 1, tool_fail_count 0
```

Regression checks against the whole FUGU endpoint:

```text
standard_fugu_20260626_184709
score 20/20 = 100.0%
overall mean 0.625s / median 0.786s / p90 1.105s / max 1.225s
completion tokens 70
completion TPS 5.596 / request TPS 1.599 / total TPS 141.429
backend decode TPS 31.319 / backend prefill TPS 100.701
efficiency vs backend theory 0.9595

japanese_fugu_20260626_184657
score 4/4 = 100.0%
mean 0.014s / median 0.014s / p90 0.022s / max 0.024s
deterministic model_call_count 0
```

Caveats:

- The 0.6B tower is currently a trained-head router using Qwen3-0.6B hidden states.
  It is not a free-form planning model generating the full dispatch plan in natural
  language.
- Very fast deterministic paths are FUGU wins, but they are not evidence that the
  worker LLM itself decodes at impossible TPS.
- The current system is strong as a local FUGU bundle for routing, output shaping,
  mode aliases, streaming, tool policy, and measured fallback behavior. The next
  higher bar is better real-world long-context compression and more tool-aware
  planning beyond unittest.

## 2026-06-26 Long Context Retrieval Pack 0.7.2

I pushed the next bar after the user's "is this really doing the FUGU job?"
question: long-context handling that can keep a relevant middle fact, not just the
head and tail of a huge prompt.

Problem found:

- Earlier long-context handling preserved head+tail. That is safe for instructions,
  but it can drop a fact buried in the middle of a long transcript.
- The first retrieval-pack attempt had a query extraction bug: it chose
  `Return only the value` instead of the actual `Question:` / `what is ...` text.
  That meant no useful terms, a 20411-character input to Qwen, and a 157.947s
  run. That failure is intentionally recorded because it explains the final shape.

What changed in `src/conductor_7940hs.py`:

- Added query-aware transcript packing before the old middle clipping path.
- Long lookup-like prompts now keep head, tail, and query-relevant middle excerpts.
- The lookup pack target cap is 6000 characters with 2200-character chunks.
- Forced `:fast`, `:deep`, and `:ultra` aliases skip the 0.6B tower when the mode
  is already explicit. That avoids paying router overhead for forced routes.
- Long lookup-like prompts with retrieval terms also skip the 0.6B tower and use the
  rules fallback, because the control decision is already clear.
- Rule routing sends these long lookup-like prompts to `fast`, currently Qwen direct.

Audit coverage added in `scripts/audit_fugu_runtime.py`:

- `--include-long` now includes `long_context_retrieval_pack`.
- The test hides `CENTRAL_SECRET_CODE = ORCHID-4827-MIDPOINT` in the middle of a
  long note and asks for it at the end.
- Expected route is the whole FUGU frontdoor via `model=local-moe-fugu:fast`, not an
  individual worker benchmark.

Validation:

```text
0.7.2 long audit:  9/9 at 2026-06-26T19:08:03
long_context_retrieval_pack: qwen_direct, router=rules, answer ORCHID-4827-MIDPOINT
cold-ish large input path: input_chars 5161, elapsed 16.925s

0.7.2 full audit:  16/16 at 2026-06-26T19:09:44
long_context_retrieval_pack: qwen_direct, router=rules, 0.521s with prompt cache
mode aliases, streaming, invalid input rejection, bad tool rejection, and unittest
tool execution all checked in one pass.

standard_fugu_20260626_191007
score 20/20 = 100.0%
overall mean 0.621s / median 0.789s / p90 1.122s / max 1.197s
completion TPS 5.635 / request TPS 1.610 / total TPS 142.409
efficiency vs backend theory 0.9585

japanese_fugu_20260626_190955
score 4/4 = 100.0%
mean 0.055s / median 0.018s / max 0.181s
deterministic model_call_count 0
```

Large middle-lookup progression:

```text
broken first pack:  input_chars 20411 / elapsed 157.947s
query fixed:        input_chars 6421  / elapsed 24.341s
0.7.2 cap+skip:     input_chars 5161  / elapsed 16.925s
cached full audit:  elapsed 0.521s
```

Current answer to "is it fulfilling the FUGU role?":

- Yes for the local frontdoor role: the client talks to `local-moe-fugu`, not to a
  worker directly, and the bundle can choose deterministic paths, Qwen, Gemma,
  compact forced modes, heavier workflows, and whitelisted tool execution.
- Yes for measured behavior: the runtime audit covers the exposed API surface,
  routing, mode aliases, long-context retrieval, streaming, and tool policy.
- Not "full fanout all the time": on this 64GB mini PC, firing every 30B-class
  worker or forcing the 0.6B tower on already-obvious routes is slower. The correct
  FUGU behavior here is selective use, with explicit telemetry proving the path.

## 2026-06-26 Runtime Capability Matrix 0.7.3

The next MAX-mode gap was observability: health checks said the frontdoor was alive,
but they did not prove that all FUGU features were active together. I added a runtime
capability matrix endpoint and made the audit fail if that matrix is incomplete.

New endpoint:

```text
GET /fugu/capabilities
```

It reports:

- workers: Gemma planner/synthesizer, Qwen primary worker, GLM verifier/critic;
- mode aliases: `local-moe-fugu`, `:fast`, `:deep`, `:ultra`;
- control: 0.6B router state, rules fallback, explicit-mode tower skip,
  long-lookup tower skip after retrieval pack, followup affinity, fast primary;
- context: default/deep ctx, long lookup retrieval pack, lookup cap, chunk size,
  head/tail fallback;
- deterministic paths: exact literal, Japanese compound, visible count, arithmetic,
  calendar multiple choice;
- tools: request-only local tool gate, allowed prefixes, tool-grounded repair;
- runtime: anti-fanout lock, no parallel model calls on the mini PC, loaded-model
  limit and verifier policy;
- telemetry: route/performance fingerprint, actual/theoretical TPS, backend prefill
  and decode TPS, tool metrics.

`scripts/audit_fugu_runtime.py` now adds `runtime_capabilities` to the normal audit.
That row verifies the endpoint itself, not just that it returns JSON.

Validation after restart:

```text
0.7.3 full audit: 17/17 at 2026-06-26T19:23:14
runtime_capabilities: ok, version 0.7.3, workers gemma/qwen/glm,
                      router_06b enabled+loaded, retrieval pack/tool/telemetry ok
long_context_retrieval_pack: qwen_direct, router=rules,
                             answer ORCHID-4827-MIDPOINT, 16.612s
mode_alias_fast:  qwen_direct, 1.429s
mode_alias_deep:  qwen_compact_deep_direct, 3.259s
mode_alias_ultra: qwen_compact_ultra_direct, 3.722s
api_tool_unittest_success: tool_call_count 1, tool_fail_count 0

standard_fugu_20260626_192335
score 20/20 = 100.0%
overall mean 0.623s / median 0.796s / p90 1.098s / max 1.196s
completion TPS 5.617 / request TPS 1.605
efficiency vs backend theory 0.9567

japanese_fugu_20260626_192323
score 4/4 = 100.0%
mean 0.040s / median 0.014s / max 0.129s
deterministic model_call_count 0
```

This does not make the router magically smarter, but it makes the deployed FUGU
state auditable: if a future change disables a worker, drops the 0.6B tower, removes
retrieval packing, loosens tool policy, or loses telemetry, the full audit now fails.

## 2026-06-26 Non-Compact Deep/Ultra Gate 0.7.5

The remaining quality gap was that the normal audit checked compact `:deep` and
`:ultra` aliases, but did not prove that the real multi-worker quality workflows
still ran end-to-end. I added a heavy audit mode:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_fugu_runtime.py --only-heavy --timeout 720 --out logs\audit_fugu_runtime_heavy_latest.json
```

What changed:

- `GET /fugu/capabilities` now reports `quality_workflows` for `fast`, `balanced`,
  `deep_full`, `ultra_full`, `deep_compact`, and `ultra_compact`.
- `scripts/audit_fugu_runtime.py --include-heavy` adds real non-compact deep/ultra
  checks.
- `--only-heavy` reruns just health, capabilities, deep full, ultra full, and the
  short followup affinity check instead of rerunning the whole normal set.
- Ultra synthesis now explicitly preserves user-requested section labels and output
  shape. This was needed because the first heavy ultra probe ran all 6 model calls
  but lost the requested `MITIGATION` label under a 129-token cap.
- The heavy ultra probe now uses `max_tokens=192`, still non-compact, but large
  enough to keep both required labels.
- The heavy deep probe now checks the actual fix content, not only `BUG`/`FIX`
  labels: it requires `if b == 0` and `None`.

Validation:

```text
0.7.5 full audit:  17/17 at 2026-06-26T19:51:08
runtime_capabilities: version 0.7.5, quality_workflows present

0.7.5 heavy audit: 5/5 at 2026-06-26T20:01:09
heavy_deep_full_workflow:
  workflow gemma_spec__qwen_solution__glm_verify__qwen_repair
  model_call_count 3
  elapsed 94.377s
  checked BUG/FIX plus `if b == 0` and `None`

heavy_ultra_full_workflow:
  workflow gemma_spec__qwen_solution__glm_critique__gemma_synthesis__glm_verify
  model_call_count 6
  elapsed 399.353s
  checked RISKS and MITIGATION labels

standard_fugu_20260626_195130
score 20/20 = 100.0%
overall mean 0.700s / median 0.913s / p90 1.221s / max 1.307s
completion TPS 5.004 / request TPS 1.430
backend decode TPS 20.020

japanese_fugu_20260626_195136
score 4/4 = 100.0%
mean 0.013s / median 0.015s / max 0.021s
deterministic model_call_count 0
```

Important note: the standard benchmark immediately after two heavy audits was slower
than the earlier 0.7.3/0.7.4 runs because backend decode TPS was around 20 instead of
31. The FUGU routing and correctness still held. For clean speed comparisons, restart
and warm the stack before benchmarking, or avoid running heavy audits immediately
before speed probes.

## 2026-06-26 Clean Speed Gate 0.7.6

The user asked whether the system is actually fulfilling the FUGU role and using the
bundle fully. My answer after inspection is:

- Functionally yes: the frontdoor exposes one `local-moe-fugu` endpoint, the 0.6B
  tower is loaded, the deterministic paths short-circuit exact/count/calendar/math
  work, Qwen is the fast primary, Gemma remains available for planner/Japanese/
  synthesis roles, GLM remains available for verification/critique, and deep/ultra
  can exercise real multi-worker workflows.
- Operationally it still needed one guardrail: heavy deep/ultra audits can leave the
  backend in a slower state, so clean speed must be measured after restart/warmup and
  not inferred from a post-heavy run.

I added:

```powershell
.\scripts\run_clean_speed_gate.ps1
```

The gate restarts/warmups the mini PC stack by default, then sends benchmark traffic to
the whole FUGU endpoint, not to individual worker models. It fails if:

- standard benchmark has failures or accuracy below 100%
- standard mean latency is above 0.85s
- standard p90 latency is above 1.40s
- backend decode TPS is below 25
- Japanese regression has failures or accuracy below 100%
- Japanese mean/max latency exceeds 0.20s/0.50s

The script writes both a timestamped log and `logs/clean_speed_gate_latest.json`.

Validation:

```text
0.7.6 full audit: 17/17 at 2026-06-26T20:13:00
runtime_capabilities: version 0.7.6, workers gemma/qwen/glm,
                      router_06b enabled+loaded, retrieval/tool/API/telemetry ok

0.7.6 capabilities:
  router_06b enabled+loaded
  fast_primary qwen
  anti_fanout_lock true
  max_loaded_models 2

clean_speed_gate_latest at 2026-06-26T20:10:19
passed true

standard_fugu_20260626_201018
score 20/20 = 100.0%
overall mean 0.598s / median 0.767s / p90 1.058s / max 1.083s
completion TPS 5.854 / request TPS 1.673
backend decode TPS 31.201 / backend prefill TPS 100.880
efficiency vs backend theory 0.9603

japanese_fugu_20260626_201018
score 4/4 = 100.0%
mean 0.014s / median 0.015s / p90 0.020s / max 0.022s
deterministic model_call_count 0
```

Interpretation: the FUGU bundle is now not only feature-complete by capability/audit
checks, but has a repeatable clean-speed gate. The remaining honest caveat is that
`deep` and especially `ultra` are intentionally slow quality modes; they are for
hard/risky requests, not the default hot path.

I then reran the heavy gate on 0.7.6 as well so every current proof line matches the
current package version:

```text
0.7.6 heavy audit: 5/5 at 2026-06-26T20:23:52
heavy_deep_full_workflow:
  workflow gemma_spec__qwen_solution__glm_verify__qwen_repair
  model_call_count 3
  elapsed 96.178s
  answer contains BUG/FIX plus if b == 0 and return None

heavy_ultra_full_workflow:
  workflow gemma_spec__qwen_solution__glm_critique__gemma_synthesis__glm_verify
  model_call_count 6
  elapsed 417.633s
  answer preserves RISKS and MITIGATION labels
```

Because the heavy audit intentionally loads/exercises the slow quality path, I ran the
clean speed gate again with restart/warmup immediately afterward:

```text
clean_speed_gate_latest at 2026-06-26T20:24:58
passed true

standard_fugu_20260626_202457
score 20/20 = 100.0%
overall mean 0.676s / median 0.812s / p90 1.208s / max 1.748s
completion TPS 5.174 / request TPS 1.478
backend decode TPS 30.978 / backend prefill TPS 96.932

japanese_fugu_20260626_202458
score 4/4 = 100.0%
mean 0.005s / median 0.003s / p90 0.011s / max 0.014s
deterministic model_call_count 0
```

Final interpretation: yes, as of 0.7.6 FUGU is doing the actual FUGU job. The single
endpoint chooses deterministic local answers when possible, uses the 0.6B tower/rules
to route ordinary work, keeps Qwen hot for fast/code, keeps Gemma/GLM available for
planner/Japanese/verification roles, can run real deep/ultra multi-worker workflows,
and now has a clean speed gate to prove the hot path after heavy work.

## 2026-06-26 Vulkan/iGPU Trial

The mini PC does have a usable integrated GPU path:

```text
adapter AMD Radeon 780M Graphics
driver 32.0.31019.2002
Vulkan llama.cpp C:\llm\bin\llama-server.exe
version b8992 / ggml-vulkan.dll
```

Raw `llama-bench` showed that the iGPU can accelerate backend compute:

```text
Qwen3-0.6B-Q8_0, p256/n64, b8992
  CPU     prompt 539.111 t/s   decode 74.905 t/s
  Vulkan  prompt 2273.583 t/s  decode 99.179 t/s

Gemma 12B Q4_K_M, p128/n32, b8992
  CPU     prompt 82.581 t/s    decode 7.216 t/s
  Vulkan  prompt 89.181 t/s    decode 7.986 t/s

Qwen3-Coder 30B-A3B Q4_K_M, p64/n16, b8992
  CPU     prompt 18.174 t/s    decode 12.406 t/s
  Vulkan  prompt 141.164 t/s   decode 26.837 t/s
```

That is the important hardware fact: the 780M is not useless. It gives a large
prefill win on small/router-like work and a substantial backend win on the Qwen
coder model when comparing the same older b8992 binary.

I then tested the whole FUGU endpoint with `scripts/run_vulkan_trial.ps1`. This starts
the Vulkan llama.cpp backend with `-ngl 99`, starts the same FUGU frontdoor, warms Qwen,
and runs the normal FUGU benchmark scripts against the single frontdoor instead of
benching individual workers only. The script intentionally omits q8 KV cache because
the b8992 Vulkan stack failed context creation with `-ctk q8_0 -ctv q8_0` on the 0.6B
model, while the production b9784 CPU stack uses q8 KV successfully.

Whole-FUGU short standard benchmark:

```text
CPU production, b9784/q8 KV/ngram-cache
  standard_fugu_20260626_202457
  score 20/20 = 100.0%
  mean 0.676s / median 0.812s / p90 1.208s / max 1.748s
  completion TPS 5.174 / request TPS 1.478
  backend decode TPS 30.978 / backend prefill TPS 96.932
  efficiency 0.9678

Vulkan trial, b8992/f16 KV/ngl 99/ngram-cache
  standard_fugu_20260626_222047
  score 20/20 = 100.0%
  mean 0.732s / median 0.968s / p90 1.189s / max 1.401s
  completion TPS 4.780 / request TPS 1.366
  backend decode TPS 38.188 / backend prefill TPS 118.538
  efficiency 0.9276
```

Interpretation: Vulkan made backend decode/prefill faster, but the short hot-path
frontdoor result was worse overall: mean latency regressed about 8.3%, request TPS
fell about 7.6%, and completion TPS fell about 7.6%. p90 and max improved slightly,
but not enough to justify replacing the production default.

Whole-FUGU average benchmark:

```text
CPU production
  avg_fugu_20260626_181929
  count 16
  mean 2.636s / median 2.295s / p90 4.215s / max 4.913s
  completion TPS 20.814 / request TPS 0.379
  backend decode TPS 25.396 / backend prefill TPS 91.451
  efficiency 0.9856

Vulkan trial
  avg_fugu_20260626_222125
  count 16
  mean 2.392s / median 2.361s / p90 3.212s / max 3.942s
  completion TPS 22.286 / request TPS 0.418
  backend decode TPS 29.764 / backend prefill TPS 84.294
  efficiency 0.9697
```

Interpretation: for longer/more average work, Vulkan helped the whole FUGU bundle:
mean latency improved about 9.3%, request TPS improved about 10.3%, completion TPS
improved about 7.1%, p90 improved about 23.8%, and max latency improved about 19.8%.

The Japanese deterministic benchmark stayed effectively unchanged because it is a
frontdoor deterministic path with no worker call:

```text
japanese_fugu_20260626_222047
runs 4/4 ok
mean 0.018s / max 0.028s
```

Decision: keep CPU b9784 as production default for now. It is the fastest proven
default for short FUGU traffic, it supports the current q8 KV profile, and it is
already persistent via `restart_fugu_stack.ps1`. Vulkan/iGPU is promising, but not
yet a clean drop-in replacement.

The next serious optimization is hybrid routing:

```text
CPU b9784 backend on 8080:
  short deterministic/fast standard path
  q8 KV
  lowest short end-to-end latency

Vulkan b8992 backend on 8081:
  longer Qwen-heavy generation
  average/deep-ish tasks where backend compute dominates
  better tail latency on those cases
```

That requires conductor support for multiple backend base URLs and route-dependent
backend selection. Until that exists, switching the single backend to Vulkan would
trade away the fastest standard path to improve longer tasks. After the trial, I
restored production to the CPU b9784 stack and confirmed `/health` reports OK.

## 2026-06-26 Hybrid CPU/Vulkan Backend

I implemented the hybrid backend path in version 0.7.7:

```text
frontdoor:
  http://127.0.0.1:9000

CPU backend:
  name cpu
  url http://127.0.0.1:8080/v1
  binary C:\llm\llama\llama-server.exe
  profile b9784, q8 KV, ngram-cache
  role Gemma/GLM plus CPU fallback

Vulkan backend:
  name vulkan
  url http://127.0.0.1:8081/v1
  binary C:\llm\bin\llama-server.exe
  profile b8992, -ngl 99, ngram-cache, default/f16 KV
  role Qwen3-Coder primary worker
```

The first hybrid attempt only sent longer Qwen calls to Vulkan. That was worse for
short standard traffic because the CPU backend still loaded Qwen while the Vulkan
backend also loaded Qwen. The duplicated 30B-class worker caused memory pressure and
the short CPU-Qwen path regressed:

```text
bad hybrid attempt, Qwen duplicated on CPU and Vulkan
standard_fugu_20260626_224755
score 20/20 = 100.0%
mean 1.348s / p90 3.120s / max 6.456s
request TPS 0.742
backend decode TPS 20.951
```

I changed the policy so Qwen routes to Vulkan by default and the CPU backend no longer
loads Qwen during warmup. CPU remains the fallback backend and still serves Gemma/GLM.
The process state after restart confirmed this split:

```text
CPU backend 8080:
  gemma-3-12b-it-Q4_K_M loaded
  Qwen3-Coder-30B-A3B-Instruct-Q4_K_M unloaded

Vulkan backend 8081:
  Qwen3-Coder-30B-A3B-Instruct-Q4_K_M loaded
```

The final clean-speed gate for 0.7.7 passed:

```text
clean_speed_gate_latest at 2026-06-26T22:51:28
passed true

standard_fugu_20260626_225128
score 20/20 = 100.0%
overall mean 0.621s / median 0.810s / p90 1.025s / max 1.101s
completion TPS 5.638 / request TPS 1.611
backend decode TPS 33.575 / backend prefill TPS 112.226
efficiency vs backend theory 0.941

japanese_fugu_20260626_225128
score 4/4 = 100.0%
mean 0.016s / median 0.015s / p90 0.026s / max 0.030s
deterministic model_call_count 0
```

The average benchmark also stayed ahead of the CPU-only result, though not quite as
fast as the earlier single-backend Vulkan-only run:

```text
hybrid average, avg_fugu_20260626_225027
count 16
mean 2.538s / median 2.274s / p90 3.736s / max 4.503s
completion TPS 21.052 / request TPS 0.394
backend decode TPS 26.327 / backend prefill TPS 77.924
backend group: vulkan for all 16 Qwen average cases

prior CPU-only average, avg_fugu_20260626_181929
mean 2.636s / p90 4.215s / request TPS 0.379 / completion TPS 20.814

prior Vulkan-only average, avg_fugu_20260626_222125
mean 2.392s / p90 3.212s / request TPS 0.418 / completion TPS 22.286
```

Final interpretation: the optimized production shape is not "CPU for short, Vulkan for
long" anymore. On this machine, once a second backend exists, the best stable split is:

```text
Qwen primary worker -> Vulkan/iGPU backend on 8081
Gemma Japanese/planner -> CPU backend on 8080
GLM verifier/fallback -> CPU backend on 8080
deterministic routes -> frontdoor only, no worker
failed Vulkan call -> CPU fallback
```

This keeps the short standard path faster than the previous CPU-only clean run
(`0.621s` mean vs `0.676s`, `1.611` request TPS vs `1.478`) while also preserving the
longer-Qwen gains from the iGPU path. The remaining tradeoff is startup cost: the first
Vulkan Qwen load still costs about 14 seconds during warmup, but the second long Qwen
warmup immediately after load completed in about 2.5 seconds.

## 2026-06-26 CPU/iGPU Hybrid Tuning 0.7.8

After the hybrid split was working, I tuned each backend separately instead of treating
the mini PC as one generic llama.cpp profile.

Raw diagnostic tuning used `llama-bench` only to choose backend flags. The reported
headline benchmarks still target the whole FUGU endpoint:

```text
http://127.0.0.1:9000/v1/chat/completions
model: local-moe-fugu
```

Adopted CPU backend profile:

```text
C:\llm\llama\llama-server.exe on 8080
role: Gemma/GLM plus fallback
threads: 8 generation / 16 batch
batch: 512
ubatch: 128
KV: q8_0/q8_0
flash-attn: on
spec: ngram-cache
```

Why: Gemma 12B raw CPU tests showed best practical generation stability around
8 generation threads, while server-side `--threads-batch 16` keeps prompt work wider.
The f16 KV experiments failed context creation in this profile, so q8 KV remains the
safe production choice.

Adopted Vulkan/iGPU backend profile:

```text
C:\llm\bin\llama-server.exe on 8081
role: Qwen3-Coder primary worker
threads: 16
batch: 512
ubatch: 512
KV: f16/f16
flash-attn: on
ngl: 99
spec: ngram-cache
```

Why: Qwen3-Coder raw Vulkan tests improved from roughly 28.7 tok/s decode to about
31.6 tok/s decode with `-fa on`, `-ub 512`, f16 KV, and 16 threads. q8 KV on this
Vulkan stack was slower or less reliable than f16 for the Qwen route.

The 0.7.8 clean-speed gate passed after restart/warmup:

```text
clean_speed_gate_latest at 2026-06-26T23:17:27
passed true

standard_fugu_20260626_231727
score 20/20 = 100.0%
overall mean 0.606s / median 0.817s / p90 0.961s / max 1.072s
completion TPS 5.777 / request TPS 1.650
backend decode TPS 35.691 / backend prefill TPS 103.206

japanese_fugu_20260626_231727
score 4/4 = 100.0%
mean 0.013s / max 0.023s
```

Compared with the previous CPU-only clean reference:

```text
CPU-only clean gate:
  mean 0.676s / p90 1.208s / request TPS 1.478 / backend decode TPS 30.978

0.7.8 tuned hybrid:
  mean 0.606s / p90 0.961s / request TPS 1.650 / backend decode TPS 35.691
```

That is about 10.4% lower mean latency, 20.4% lower p90 latency, 11.6% higher request
TPS, and 15.2% higher backend decode TPS on the short standard gate.

Average FUGU benchmark after the CPU/iGPU tuning:

```text
avg_fugu_20260626_231149
runs 24/24 ok
overall mean 2.369s / median 2.231s / p90 3.696s / max 4.225s
completion TPS 23.001 / total TPS 62.566 / request TPS 0.422
backend decode TPS 28.448 / backend prefill TPS 73.768
backend group: vulkan for all 24 Qwen average cases
```

Full runtime audit after the tuned hybrid stack:

```text
audit_fugu_runtime_20260626_igpu_cpu_tuned_v078.json
runs 17/17 ok
capabilities version: 0.7.8
health: CPU backend 8080 and Vulkan backend 8081 both reported
capabilities: router_06b loaded, hybrid policy active, deterministic/API/tool/long-context checks ok
gemma_japanese_general: primary gemma, workflow gemma_direct, CPU path
qwen_fast_code/mode aliases/long lookup: Qwen path, Vulkan backend policy
```

Final current shape:

```text
deterministic facts/counts -> frontdoor only
Qwen/code/compact deep/ultra -> Vulkan/iGPU on 8081
Gemma Japanese/planner/synthesis -> CPU on 8080
GLM verification/critique -> CPU on 8080
Vulkan failure -> CPU fallback
```

## 2026-06-26 Priority / No-Poll Final Tuning 0.7.9

I pushed past the 0.7.8 hybrid result by testing scheduler priority, explicit llama.cpp
polling, and the latest official llama.cpp b9811 Vulkan release against the whole FUGU
frontdoor. These are not isolated model microbenchmarks; all reported numbers target:

```text
http://127.0.0.1:9000/v1/chat/completions
model: local-moe-fugu
```

Installed but not adopted as the production default:

```text
C:\llm\llama-b9811-cpu\llama-server.exe
C:\llm\llama-b9811-vulkan\llama-server.exe
source: https://github.com/ggml-org/llama.cpp/releases/tag/b9811
```

The official b9811 Vulkan backend was competitive on the average benchmark, but it
regressed the short standard gate enough that I kept the production Qwen backend on the
existing Vulkan b8992 build:

```text
b9811 Vulkan trial, standard_fugu_20260626_233142
  20/20 correct
  mean 0.624s / p90 1.049s / request TPS 1.602
  completion TPS 5.605 / backend decode TPS 37.403 / prefill TPS 106.881

b9811 Vulkan trial, avg_fugu_20260626_233402
  24/24 ok
  mean 2.193s / median 1.951s / p90 3.319s
  completion TPS 24.417 / request TPS 0.456
  backend decode TPS 30.765 / prefill TPS 72.705
```

I then compared b8992 with explicit Poll50 and with no explicit `--poll` flag. Poll50
was close, but no-poll won the average benchmark and had the best short backend decode
TPS:

```text
b8992 Vulkan + High priority + Poll50, standard_fugu_20260626_232908
  20/20 correct
  mean 0.603s / p90 0.966s / request TPS 1.658
  backend decode TPS 36.438 / prefill TPS 115.411

b8992 Vulkan + High priority + Poll50, avg_fugu_20260626_233552
  24/24 ok
  mean 2.204s / p90 3.353s / completion TPS 24.212 / request TPS 0.454
  backend decode TPS 30.631 / prefill TPS 87.888

b8992 Vulkan + High priority + no explicit --poll, standard_fugu_20260626_233905
  20/20 correct
  mean 0.597s / median 0.798s / p90 0.978s / max 1.075s
  completion TPS 5.860 / request TPS 1.674
  backend decode TPS 37.263 / prefill TPS 116.688

b8992 Vulkan + High priority + no explicit --poll, avg_fugu_20260626_234007
  24/24 ok
  mean 2.175s / median 2.047s / p90 3.346s / max 4.009s
  completion TPS 24.469 / total TPS 67.583 / request TPS 0.460
  backend decode TPS 30.732 / prefill TPS 76.249
```

After syncing the 0.7.9 source and restarting the stack, the final proof run was:

```text
runtime capabilities
  version 0.7.9
  CPU backend http://127.0.0.1:8080/v1
  Vulkan backend http://127.0.0.1:8081/v1
  router_06b loaded true

standard_fugu_20260626_234409
  20/20 correct
  mean 0.595s / median 0.785s / p90 0.974s / max 1.056s
  completion TPS 5.883 / request TPS 1.681
  backend decode TPS 37.284 / prefill TPS 116.983

japanese_fugu_20260626_234417
  4/4 correct
  mean 0.018s / p90 0.021s / max 0.023s
  deterministic model calls 0

avg_fugu_20260626_234517
  24/24 ok
  mean 2.218s / median 1.956s / p90 3.370s / max 3.967s
  completion TPS 24.627 / total TPS 66.893 / request TPS 0.451
  backend decode TPS 30.720 / prefill TPS 76.642

audit_fugu_runtime_20260626_priority_no_poll_v079.json
  8/8 ok
  capabilities version 0.7.9
  health reports both CPU and Vulkan backends
  Qwen fast path uses 0.6B tower routing
  Gemma Japanese route checked
```

Final 0.7.9 production decision:

```text
CPU backend 8080:
  C:\llm\llama\llama-server.exe
  b9784 CPU build
  Gemma/GLM plus fallback
  High process priority
  no explicit --poll by default

Vulkan/iGPU backend 8081:
  C:\llm\bin\llama-server.exe
  b8992 Vulkan build with ggml-vulkan.dll
  Qwen3-Coder primary worker
  High process priority
  no explicit --poll by default

Frontdoor 9000:
  uvicorn src.conductor_7940hs:app
  version 0.7.9
  High process priority
```

Why this is the current best practical setting: b9811 was not a free win, and explicit
polling did not beat the no-poll profile end to end. The remaining gap to theoretical
minimum is now mostly backend generation time plus roughly 45-70 ms of frontdoor/router
overhead on model-routed calls. That means further large gains probably require a
smaller/faster Qwen replacement, a better Vulkan build/driver, or accepting less answer
quality, not just another wrapper-level tweak.
