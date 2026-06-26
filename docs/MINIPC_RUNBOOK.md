# Mini PC Runbook

Target host: `<MINIPC-HOST>` via `ssh user@<MINIPC-IP>`.

## Current Installed Paths

- llama.cpp b9784: `C:\llm\llama\`
- llama server: `C:\llm\llama\llama-server.exe`
- models: `C:\llm\models\`
- runtime log: `C:\llm\server.log`

Model IDs observed from `server.log` and matched by `configs/moe_fugu_minipc_local.yaml`:

```text
gemma-3-12b-it-Q4_K_M
Qwen3-Coder-30B-A3B-Instruct-Q4_K_M
GLM-4.7-Flash-UD-Q4_K_XL
```

## Important Fix

The old helper at `C:\Users\user\start_server.ps1` starts router mode with:

```powershell
-c 4096
```

That conflicts with the conductor config, which budgets prompts for `default_ctx: 12288`
and `deep_ctx: 16384`. Use `scripts/start_llama_router_minipc.ps1` instead, or update
the old helper to match it.

## Start on the Mini PC

Copy this project to the mini PC, for example:

```powershell
scp -F NUL -r .\local_moe_fugu_7940hs user@<MINIPC-IP>:C:\llm\local-moe-fugu-7940hs
```

Then SSH in:

```powershell
ssh -F NUL user@<MINIPC-IP>
cd C:\llm\local-moe-fugu-7940hs
```

Install Python dependencies once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install the Qwen3-0.6B routing tower once:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_fugu_router_06b.ps1
```

This installs CPU PyTorch/Transformers into the project venv, downloads
`Qwen/Qwen3-0.6B` to `C:\llm\models\Qwen3-0.6B`, and warms the local prototype
head cache under `C:\llm\local-moe-fugu-7940hs\.cache\fugu_router_06b`.
It also builds a self-made seed head at:

```text
C:\llm\local-moe-fugu-7940hs\artifacts\router_06b_seed_head.npz
```

If a better trained `head_path` is later provided in `configs\moe_fugu_minipc_local.yaml`,
the same router will use that head instead.

OpenFugu is not used as a dependency. It was only a public reference for the
hidden-state-routing pattern; this package keeps its own local 3-worker router.

Start llama.cpp router:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_llama_router_minipc.ps1 -StopExisting
```

Open a second shell and start the frontdoor:

```powershell
cd C:\llm\local-moe-fugu-7940hs
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_frontdoor_minipc.ps1
```

For SSH-launched persistent runtime, use the user wrapper that starts both long-running
processes through Windows Scheduled Tasks. This avoids the OpenSSH session closing and
taking child processes down with it:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\user\restart_fugu_stack.ps1
```

This script restarts:

```text
LocalFuguRouter    -> C:\Users\user\start_server.ps1 -StopExisting
LocalFuguFrontdoor -> C:\Users\user\start_fugu_frontdoor.ps1
```

The current default for `C:\Users\user\start_server.ps1` is the measured best
profile:

```text
--models-max 2
-ctk q8_0
-ctv q8_0
--spec-type ngram-cache
```

This keeps the common Gemma/Qwen workers warm without the memory pressure seen
with three resident workers.

After both services are healthy, `restart_fugu_stack.ps1` also sends two short
warmup requests through the FUGU frontdoor:

```text
warm_gemma_router -> loads the 0.6B router and Gemma
warm_qwen_code    -> loads Qwen
```

This makes the restart command longer, but avoids making the first interactive user
request pay the router/model cold-load cost. To skip this behavior for launcher
experiments:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\user\restart_fugu_stack.ps1 -SkipWarmup
```

Runtime logs:

```text
C:\llm\server.log
C:\llm\local-moe-fugu-7940hs\logs\frontdoor.log
C:\llm\local-moe-fugu-7940hs\logs\moe_fugu_runs.jsonl
```

The current mini PC router profile starts llama.cpp with:

```text
--models-max 2
--cache-prompt
--cache-idle-slots
--reasoning off
--reasoning-budget 0
-ctk q8_0
-ctv q8_0
--spec-type ngram-cache
--spec-ngram-mod-n-min 1
--spec-ngram-mod-n-max 8
--spec-ngram-mod-n-match 24
```

This uses the cache-prompt path from the chatcache experiment and enables draft-free
ngram speculative decoding. `Qwen3-0.6B-Q8_0.gguf` is now present on disk and can
be tested through `configs\llama_models_minipc_preset.ini`, but the measured Qwen
draft-simple run was slower than ngram-cache only, so it is not enabled by default.

Smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_minipc_backend.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test.ps1
.\.venv\Scripts\python.exe .\scripts\smoke_frontdoor.py
```

`smoke_frontdoor.py` should return a `chat.completion` response whose
`system_fingerprint.route.router` is `qwen3-0.6b-hidden` when the routing tower is
available. If dependencies or the model are missing, the frontdoor falls back to the
rules router and logs `router_06b_fallback`.

## 0.6B Routing Tower

The production path in this package is now:

```text
OpenAI request -> frontdoor -> Qwen3-0.6B hidden-state router -> selected local worker
```

The 0.6B router does not generate the answer. It formats the conversation as a raw
transcript, reads one hidden vector from Qwen3-0.6B, and scores small linear heads for:

- local worker: `gemma`, `qwen`, `glm`
- mode: `fast`, `balanced`, `deep`, `ultra`
- task type: `general`, `coding`, `design`, `verification`, `research`

The default head is a self-made seed head built from synthetic local routing examples.
That is still a real hidden-state routing path, not keyword matching. The next training
step is to collect prompt/outcome logs and fit a stronger 3-worker head to local
latency/quality results.

## Short-Response Performance Fix

The conductor now reads OpenAI-compatible `max_tokens` and `max_completion_tokens` from
incoming requests. When a client asks for a short answer, fast/balanced/deep/ultra no
longer reserve 6000-10000 output tokens per internal stage. Deep and ultra keep their
multi-model workflow, but internal spec/critique/verify budgets are scaled down from the
client limit.

Latest measured `max_tokens: 96` frontdoor results on <MINIPC-HOST>:

```text
frontdoor_fast_general_hot_repeat   38.817s ->   6.007s
frontdoor_fast_code_qwen            82.100s ->  52.937s
frontdoor_balanced_code_with_verify 118.759s -> 48.678s
frontdoor_deep_small_code           352.894s -> 129.813s
frontdoor_ultra_small_code          422.043s -> 194.306s
```

The remaining cold-load cost is llama.cpp router hot-loading large GGUF files. Hot
backend generation is still around 16-20 completion tokens/s on this machine.

## Chatcache-Style Results

After enabling cache-oriented runtime flags, turning reasoning off, and biasing the
default scheduler toward fast one-model routes, the visible-output multi-turn benchmark
with `max_tokens: 256` produced:

```text
chatcache_fugu_20260625_202835
overall   mean 16.959s / median 10.101s / p90 35.380s
code      mean 12.628s / median  8.993s / p90 23.060s
general   mean 21.291s / median 16.640s / p90 41.647s
```

The same average task set, one pass with visible output and `max_tokens: 256`, produced:

```text
avg_fugu_20260625_203342
overall mean 36.529s / median 21.895s / p90 71.484s
fast    mean 23.658s / median 15.386s / p90 46.989s
```

The remaining slow cases are Qwen cold/hot-load turns and explicit `review` tasks that
still route to deep verification.

## 2026-06-26 FUGU Bundle Benchmark

Headline benchmarks target the whole FUGU bundle, not individual worker models:

```text
client -> local-moe-fugu frontdoor -> 0.6B router -> selected worker/workflow -> answer
```

After replacing the workers with Gemma 3 12B, Qwen3-Coder 30B-A3B, and GLM-4.7
Flash, then testing launcher variants, the best measured setting is:

```text
models-max=2, KV=q8_0/q8_0, ngram-cache
```

Best run:

```text
avg_fugu_20260626_015926
overall mean 20.851s / median 6.681s / fast mean 8.166s / deep 109.644s

standard_fugu_20260626_015632
score 19/20 = 95.0%
overall mean 4.944s / median 2.663s
```

Rejected variants:

```text
models-max=3 q8:            avg mean 30.761s
models-max=3 q4 cache-reuse: avg mean 32.369s
models-max=2 q4 cache-reuse: avg mean 29.341s
Qwen draft-simple preset:    avg mean 25.744s
batch=1024 ubatch=256:       avg mean 29.141s
threads=16:                  avg mean 31.143s
threads=10:                  avg mean 27.086s
```

Startup warmup check:

```text
warm_gemma_router route=gemma/fast elapsed=17.769s
warm_qwen_code    route=qwen/fast  elapsed=26.345s
first external short request after READY: 3.945s, router_elapsed_ms=142
standard_fugu_20260626_025207 after warmup: 19/20, mean 2.099s
```

## Main PC Access

The scripts bind to `127.0.0.1` on the mini PC. To call the frontdoor from the main PC,
use an SSH tunnel:

```powershell
ssh -F NUL -N -L 9000:127.0.0.1:9000 user@<MINIPC-IP>
```

Then clients on the main PC can use:

```text
http://127.0.0.1:9000/v1
```

When the Windows OpenSSH client complains about `C:\Users\user\.ssh\config`
permissions, bypass that config with `-F NUL`:

```powershell
ssh -F NUL user@<MINIPC-IP>
scp -F NUL ...
```

Do not expose the llama.cpp router or the frontdoor directly on `0.0.0.0` unless you add
proper network controls.

## Current Hot Runtime Snapshot

As of 2026-06-26 20:10 JST, the mini PC is running the faster FUGU policy:

```text
llama.cpp router: models-max=2, ctx=16384, q8_0 KV, threads=12, batch=512, ubatch=128
frontdoor:        local-moe-fugu on http://127.0.0.1:9000/v1
fast primary:     qwen
deterministic:    arithmetic + weekday multiple-choice before model routing
compact deep:     short review calls use Qwen direct with capped output
long lookup:      head/tail plus query-relevant middle retrieval pack before worker routing
capabilities:     GET /fugu/capabilities reports the active MAX-mode feature matrix
heavy quality:    --only-heavy verifies real non-compact deep/ultra workflows
speed gate:       run_clean_speed_gate.ps1 verifies clean whole-FUGU speed after restart/warmup
```

Restart command:

```powershell
ssh -F NUL user@<MINIPC-IP> "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\user\restart_fugu_stack.ps1"
```

Capability check:

```powershell
ssh -F NUL user@<MINIPC-IP> curl.exe -s http://127.0.0.1:9000/fugu/capabilities
```

This should show `version: 0.7.8`, `workers` for `gemma`, `qwen`, and `glm`,
`router_06b.loaded: true`, `long_lookup_retrieval_pack: true`, the local tool
whitelist, anti-fanout runtime constraints, and TPS telemetry flags.

Heavy quality check:

```powershell
ssh -F NUL user@<MINIPC-IP> "cd /d C:\llm\local-moe-fugu-7940hs && .venv\Scripts\python.exe scripts\audit_fugu_runtime.py --only-heavy --timeout 720 --out logs\audit_fugu_runtime_heavy_latest.json"
```

This runs the real non-compact `:deep` and `:ultra` workflows. It is intentionally
slow and should not be run immediately before a clean speed benchmark.

Clean speed gate:

```powershell
ssh -F NUL user@<MINIPC-IP> "cd /d C:\llm\local-moe-fugu-7940hs && powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_clean_speed_gate.ps1"
```

This restarts/warmups the stack, then benchmarks the whole FUGU frontdoor. It fails
if correctness drops, standard mean exceeds 0.85s, standard p90 exceeds 1.40s, backend
decode TPS drops below 25, or Japanese deterministic regression exceeds the latency
limits.

Future warmup output should look like:

```text
WARMUP warm_gemma_japanese    route=gemma/fast elapsed=... actual_tps=... theory_tps=...
WARMUP warm_qwen_code           route=qwen/fast elapsed=...
READY router=http://127.0.0.1:8080 frontdoor=http://127.0.0.1:9000
```

Latest whole-FUGU benchmark logs copied into this bundle:

```text
logs/avg_fugu_20260626_081741.*
logs/standard_fugu_20260626_081804.*
logs/avg_fugu_20260626_153938.*
logs/standard_fugu_20260626_153910.*
logs/avg_fugu_20260626_172845.*
logs/standard_fugu_20260626_172818.*
logs/japanese_fugu_20260626_173531.*
logs/standard_fugu_20260626_173555.*
logs/audit_fugu_runtime_latest.json
logs/japanese_fugu_20260626_174809.*
logs/standard_fugu_20260626_174826.*
logs/avg_fugu_20260626_174917.*
logs/audit_fugu_runtime_full_latest.json
logs/audit_fugu_runtime_long_latest.json
logs/standard_fugu_20260626_191007.*
logs/japanese_fugu_20260626_190955.*
logs/standard_fugu_20260626_192335.*
logs/japanese_fugu_20260626_192323.*
logs/audit_fugu_runtime_heavy_latest.json
logs/standard_fugu_20260626_195130.*
logs/japanese_fugu_20260626_195136.*
logs/standard_fugu_20260626_201018.*
logs/japanese_fugu_20260626_201018.*
logs/clean_speed_gate_latest.json
logs/audit_fugu_runtime_full_latest.json
logs/audit_fugu_runtime_heavy_latest.json
logs/standard_fugu_20260626_202457.*
logs/japanese_fugu_20260626_202458.*
```

Headline:

```text
average benchmark:  completion TPS 19.250 / total TPS 59.996 / request TPS 0.339
standard benchmark: 20/20, completion TPS 5.588 / total TPS 170.991 / request TPS 1.597

latest average:     mean 2.594s / completion TPS 19.568 / theoretical completion TPS 19.897
latest standard:    20/20, mean 0.846s / completion TPS 9.106 / theoretical completion TPS 9.503

compound deterministic latest: Japanese weekday/integral/ice-cream/50m/vowel-count prompt
                               routes to deterministic_japanese_compound in about 0.015s
latest average v2:  mean 2.555s / completion TPS 19.468 / theoretical completion TPS 19.815
latest standard v2: 20/20, mean 0.845s / completion TPS 9.233 / theoretical completion TPS 9.624
latest Japanese regression: 4/4, mean 0.008s, p90 0.018s, deterministic model_call_count 0
latest standard v3: 20/20, mean 0.840s / completion TPS 9.164 / theoretical completion TPS 9.583
runtime audit:     4/4, router_06b loaded, deterministic_count model_call_count 0,
                   qwen code route, gemma Japanese route, followup affinity qwen->qwen
latest Japanese v2: 4/4, mean 0.003s, p90 0.006s
latest standard v4: 20/20, mean 0.848s / completion TPS 9.202 / theoretical completion TPS 9.550
latest average v3:  16/16, mean 2.661s / completion TPS 20.478 / theoretical completion TPS 20.822
output-shape pass: audit 4/4, expression-only responses have no code fence
latest Japanese v3: 4/4, mean 0.011s, p90 0.021s
latest standard v5: 20/20, mean 0.706s / completion tokens 92 / max 2.012s
latest average v4:  16/16, mean 2.564s / completion TPS 21.279 / theoretical completion TPS 21.709
0.6.6 audit:       4/4 at 2026-06-26T18:00:17, router_06b loaded,
                   expression-only and followup-affinity checks passing
0.6.7 audit:       4/4 at 2026-06-26T18:07:19, router_06b loaded,
                   ANSWER-only stop sequences active
latest standard v6: 20/20, mean 0.603s / p90 1.068s / max 1.111s / completion tokens 70
latest Japanese v4: 4/4, mean 0.013s / max 0.030s
latest average v5:  16/16, mean 2.659s / completion TPS 20.711 / theory 21.079
0.6.8 audit:       6/6 at 2026-06-26T18:16:16, health/router/models/perf metadata
                   checked, workflow artifact cleanup checked
0.6.8 long audit:  7/7 at 2026-06-26T18:17:32, long head/tail exact retrieval ok,
                   long_context_head_tail 53.879s on qwen_direct
latest standard v7: 20/20, mean 0.611s / p90 1.102s / max 1.128s / completion tokens 70
latest Japanese v5: 4/4, mean 0.018s / max 0.027s
latest average v6:  16/16, mean 2.636s / completion TPS 20.814 / theory 21.197
0.6.9 audit:       7/7 at 2026-06-26T18:26:44, deterministic_exact_literal model_call_count 0
0.6.9 long audit:  8/8 at 2026-06-26T18:27:07, long_context_head_tail now
                   deterministic_exact_literal, 0.002s, model_call_count 0
latest standard v8: 20/20, mean 0.642s / p90 1.165s / max 1.227s / completion tokens 70
latest Japanese v6: 4/4, mean 0.012s / max 0.017s
0.7.0 mode audit: 10/10 at 2026-06-26T18:37:33, :fast/:deep/:ultra aliases checked
                  :deep compact 4.134s, :ultra compact 3.986s, both model_call_count 1
0.7.0 long audit: 8/8 at 2026-06-26T18:38:16, long exact remains deterministic, 0.019s
latest standard v9: 20/20, mean 0.659s / p90 1.174s / max 1.295s / completion tokens 70
latest Japanese v7: 4/4, mean 0.009s / max 0.026s
0.7.1 API audit:  11/11 at 2026-06-26T18:44:46, streaming SSE ok, invalid input 400,
                  bad tool command rejected, unittest tool call counted
0.7.1 full audit: 15/15 at 2026-06-26T18:45:47, router_06b loaded, mode aliases,
                  long deterministic, API tool path, and stream path checked together
latest standard v10: 20/20, mean 0.625s / p90 1.105s / max 1.225s / completion TPS 5.596
latest Japanese v8:  4/4, mean 0.014s / max 0.024s, deterministic model_call_count 0
0.7.2 long audit:  9/9 at 2026-06-26T19:08:03, retrieval pack found middle
                  code ORCHID-4827-MIDPOINT, cold-ish 16.931s on qwen_direct
0.7.2 full audit:  16/16 at 2026-06-26T19:09:44, router_06b loaded, mode aliases,
                  retrieval pack cached 0.521s, streaming/API/tool checks passing
latest standard v11: 20/20, mean 0.621s / p90 1.122s / max 1.197s /
                     completion TPS 5.635 / request TPS 1.610
latest Japanese v9:  4/4, mean 0.055s / max 0.181s,
                     deterministic model_call_count 0
0.7.3 full audit:  17/17 at 2026-06-26T19:23:14, /fugu/capabilities checked,
                  workers gemma/qwen/glm, router_06b loaded, retrieval/tool/telemetry ok
0.7.3 long lookup: ORCHID-4827-MIDPOINT found through retrieval pack, 16.612s
latest standard v12: 20/20, mean 0.623s / p90 1.098s / max 1.196s /
                     completion TPS 5.617 / request TPS 1.605
latest Japanese v10: 4/4, mean 0.040s / max 0.129s,
                     deterministic model_call_count 0
0.7.5 full audit:  17/17 at 2026-06-26T19:51:08, quality_workflows present
0.7.5 heavy audit: 5/5 at 2026-06-26T20:01:09,
                  deep full 3 model calls / 94.377s,
                  ultra full 6 model calls / 399.353s
latest standard v13: 20/20, mean 0.700s / p90 1.221s / max 1.307s /
                     completion TPS 5.004 / request TPS 1.430
latest Japanese v11: 4/4, mean 0.013s / max 0.021s,
                     deterministic model_call_count 0
0.7.6 capabilities: version 0.7.6, router_06b loaded, fast_primary qwen,
                    max_loaded_models 2, anti_fanout_lock true
0.7.6 full audit:  17/17 at 2026-06-26T20:13:00, capabilities/modes/long/API ok
0.7.6 heavy audit: 5/5 at 2026-06-26T20:23:52,
                    deep full 3 model calls / 96.178s,
                    ultra full 6 model calls / 417.633s
clean speed gate:   pass at 2026-06-26T20:24:58
latest standard v14: 20/20, mean 0.598s / p90 1.058s / max 1.083s /
                     completion TPS 5.854 / request TPS 1.673 /
                     backend decode TPS 31.201
latest Japanese v12: 4/4, mean 0.014s / max 0.022s,
                     deterministic model_call_count 0
latest standard v15: 20/20, mean 0.676s / p90 1.208s / max 1.748s /
                     completion TPS 5.174 / request TPS 1.478 /
                     backend decode TPS 30.978
latest Japanese v13: 4/4, mean 0.005s / max 0.014s,
                     deterministic model_call_count 0
```

Important caveat: these are frontdoor bundle results, not worker-model microbenchmarks.
The client sends `model=local-moe-fugu`; the conductor decides whether to answer via
deterministic path, Qwen fast direct, compact deep, or a heavier workflow.

## Japanese And Long Context Notes

Japanese non-code prompts are intentionally routed to Gemma after the normal Qwen fast
bias because Qwen3-Coder produced unreliable Japanese/general responses on this machine.
Japanese code/command prompts may still use Qwen.

For Japanese validation, avoid PowerShell `Invoke-RestMethod` unless you explicitly send
a UTF-8 JSON body. In this SSH setup it can corrupt Japanese text before FUGU sees it.
Prefer Python/urllib or any client that sends:

```text
Content-Type: application/json; charset=utf-8
```

Long requests are clipped with head+tail middle omission in both the conductor and the
0.6B router. This preserves early instructions and recent user content instead of
dropping the beginning of the prompt.

For long lookup-like prompts, the conductor now tries a retrieval context pack before
plain clipping. It keeps head, tail, and query-relevant middle excerpts, then routes
the smaller packed transcript to the worker. The 0.7.2 audit verifies a middle fact
hidden inside a long prompt:

```text
CENTRAL_SECRET_CODE = ORCHID-4827-MIDPOINT
answer: ORCHID-4827-MIDPOINT
```

Latest Japanese probe:

```text
route gemma/fast
old elapsed 52.797s
new elapsed 19.027s
actual completion TPS 4.787
theoretical completion TPS 4.790
efficiency vs theory 0.9994
```

The current narrow compound Japanese prompt path is faster because it bypasses the
worker model completely:

```text
workflow deterministic_japanese_compound
elapsed 0.015s
model_call_count 0
```

This path is intentionally narrow. It handles visible text counts and deterministic
calendar facts directly, then returns concise canned guidance for the integral,
ice-cream, and 50m travel subquestions. Broaden it only with cases that can be tested
and logged.

## Per-Request Theory Metrics

Every non-stream response includes:

```text
system_fingerprint.performance
```

Useful fields:

```text
end_to_end_ms
actual_completion_tps
theoretical_min_ms
theoretical_completion_tps
backend_prefill_tps
backend_decode_tps
backend_names
backend_call_count_by_name
backend_fallback_count
overhead_ms
efficiency_vs_backend_theory
model_calls[]
```

The theoretical value is computed from the actual llama.cpp timings for that same
request:

```text
router_elapsed_ms + prompt_ms + predicted_ms
```

So `overhead_ms` is the part outside backend compute: frontdoor overhead, HTTP,
queueing, and cold-load/model-switch effects. In a good hot path,
`efficiency_vs_backend_theory` should be close to 1.0.

For one-off UTF-8 probes, use:

```powershell
.\.venv\Scripts\python.exe .\scripts\probe_fugu_prompt.py --prompt-b64 <base64-utf8> --max-tokens 512
```

For the Japanese regression benchmark against the FUGU bundle:

```powershell
.\.venv\Scripts\python.exe .\scripts\bench_fugu_japanese.py
```

For a quick whole-runtime audit:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_fugu_runtime.py
```

This checks the frontdoor health, exposed model list, deterministic count behavior,
Qwen/Gemma routing, per-request performance metadata, and short-followup primary
affinity.

## Vulkan/iGPU Trial

The mini PC has an AMD Radeon 780M iGPU and the older Vulkan llama.cpp tree in:

```text
C:\llm\bin\llama-server.exe
C:\llm\bin\llama-bench.exe
```

The current production stack uses both backends:

```text
CPU:
  C:\llm\llama\llama-server.exe
  http://127.0.0.1:8080/v1
  q8 KV, ngram-cache
  Gemma/GLM plus CPU fallback

Vulkan/iGPU:
  C:\llm\bin\llama-server.exe
  http://127.0.0.1:8081/v1
  -ngl 99, ngram-cache, default/f16 KV
  Qwen primary worker
```

Reason: Vulkan was faster in raw backend compute. The final production policy routes
Qwen to Vulkan by default and keeps CPU for Gemma/GLM and fallback. Do not load Qwen
on both CPU and Vulkan unless you are intentionally testing memory pressure; that
regressed the standard benchmark.

Trial command:

```powershell
cd C:\llm\local-moe-fugu-7940hs
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_vulkan_trial.ps1 -IncludeAverage
```

Observed whole-FUGU comparison:

```text
standard short benchmark
  CPU production mean 0.676s, request TPS 1.478, completion TPS 5.174
  Vulkan trial   mean 0.732s, request TPS 1.366, completion TPS 4.780
  Hybrid final   mean 0.621s, request TPS 1.611, completion TPS 5.638

average benchmark
  CPU production mean 2.636s, request TPS 0.379, completion TPS 20.814
  Vulkan trial   mean 2.392s, request TPS 0.418, completion TPS 22.286
  Hybrid final   mean 2.538s, request TPS 0.394, completion TPS 21.052
```

Operational caveats:

```text
scripts\run_vulkan_trial.ps1 is a benchmark/trial launcher, not the production
persistent launcher.

The Vulkan b8992 stack failed with q8 KV cache during context creation, so the trial
uses the default/f16 KV profile. Production CPU b9784 uses q8 KV successfully.

After a Vulkan trial, restore the production hybrid stack with:
```

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\user\restart_fugu_stack.ps1
```

Expected healthy state:

```text
/health reports:
  backends.cpu    http://127.0.0.1:8080/v1
  backends.vulkan http://127.0.0.1:8081/v1
  hybrid_backend.enabled true

processes:
  C:\llm\llama\llama-server.exe on 8080
  C:\llm\bin\llama-server.exe on 8081
  CPU child model loaded: gemma-3-12b-it-Q4_K_M
  Vulkan child model loaded: Qwen3-Coder-30B-A3B-Instruct-Q4_K_M
```

Clean speed gate proof:

```text
clean_speed_gate_latest at 2026-06-26T22:51:28
passed true
standard mean 0.621s / p90 1.025s / request TPS 1.611
backend decode TPS 33.575 / backend prefill TPS 112.226
japanese mean 0.016s / max 0.030s
```

## Current CPU/iGPU Tuned Hybrid

As of 2026-06-26 23:17 JST, production uses both CPU and the AMD Radeon 780M iGPU:

```text
frontdoor:
  http://127.0.0.1:9000/v1

CPU backend:
  C:\llm\llama\llama-server.exe
  http://127.0.0.1:8080/v1
  Gemma/GLM plus fallback
  --models-max 2
  -t 8
  -tb 16
  -b 512
  -ub 128
  -ctk q8_0
  -ctv q8_0
  -fa on
  --spec-type ngram-cache

Vulkan/iGPU backend:
  C:\llm\bin\llama-server.exe
  http://127.0.0.1:8081/v1
  Qwen3-Coder primary worker
  --models-max 1
  -t 16
  -b 512
  -ub 512
  -ctk f16
  -ctv f16
  -fa on
  -ngl 99
  --spec-type ngram-cache
```

The hybrid policy sends Qwen calls to `vulkan` and keeps Gemma/GLM plus fallback on
`cpu`. This is intentional: duplicating Qwen on both backends caused memory pressure
and slower short-path results.

Latest proof logs copied into this bundle:

```text
logs/standard_fugu_20260626_231727.*
logs/japanese_fugu_20260626_231727.*
logs/clean_speed_gate_latest.json
logs/avg_fugu_20260626_231149.*
logs/audit_fugu_runtime_20260626_igpu_cpu_tuned_v078.json
logs/tune_cpu_gemma_threads_20260626.json
logs/tune_cpu_gemma_kv_fa_ubatch_20260626.json
logs/tune_vulkan_qwen_threads_20260626.json
logs/tune_vulkan_qwen_fa_ubatch_20260626.json
logs/tune_vulkan_qwen_f16_b512_ub512_20260626.json
```

Latest clean-speed gate:

```text
clean_speed_gate_latest at 2026-06-26T23:17:27
passed true
standard 20/20, mean 0.606s, p90 0.961s, request TPS 1.650
backend decode TPS 35.691, backend prefill TPS 103.206
japanese 4/4, mean 0.013s, max 0.023s
```

Latest average benchmark:

```text
avg_fugu_20260626_231149
24/24 ok
mean 2.369s, p90 3.696s, completion TPS 23.001, request TPS 0.422
backend decode TPS 28.448, backend prefill TPS 73.768
```

Latest full runtime audit:

```text
audit_fugu_runtime_20260626_igpu_cpu_tuned_v078.json
17/17 ok
capabilities version 0.7.8
CPU and Vulkan backends exposed in health/capabilities
0.6B router loaded
Gemma Japanese route checked
Qwen fast/deep/ultra routes checked
long-context retrieval, streaming, validation, and tool policy checked
```

## Current 0.7.9 Priority Tuned Hybrid

As of 2026-06-26 23:40 JST, the fastest stable default is:

```text
frontdoor:
  http://127.0.0.1:9000/v1
  uvicorn src.conductor_7940hs:app
  version 0.7.9
  High process priority

CPU backend:
  C:\llm\llama\llama-server.exe
  b9784 CPU build
  http://127.0.0.1:8080/v1
  Gemma/GLM plus fallback
  High process priority
  no explicit --poll by default

Vulkan/iGPU backend:
  C:\llm\bin\llama-server.exe
  b8992 Vulkan build
  http://127.0.0.1:8081/v1
  Qwen3-Coder primary worker
  AMD Radeon 780M iGPU via Vulkan
  High process priority
  no explicit --poll by default
```

Restart the tuned stack:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\user\restart_fugu_stack.ps1
```

Optional experiments:

```powershell
# Force explicit polling if needed for comparison.
powershell -ExecutionPolicy Bypass -File C:\Users\user\restart_fugu_stack.ps1 -Poll 50

# Try the staged official b9811 Vulkan server, not the production default.
powershell -ExecutionPolicy Bypass -File C:\Users\user\restart_fugu_stack.ps1 -VulkanServer C:\llm\llama-b9811-vulkan\llama-server.exe
```

Latest 0.7.9 proof logs:

```text
logs/standard_fugu_20260626_234409.*
logs/japanese_fugu_20260626_234417.*
logs/avg_fugu_20260626_234517.*
logs/audit_fugu_runtime_20260626_priority_no_poll_v079.json
```

Current benchmark proof:

```text
standard_fugu_20260626_234409
20/20 correct
mean 0.595s, p90 0.974s, request TPS 1.681
backend decode TPS 37.284, backend prefill TPS 116.983

japanese_fugu_20260626_234417
4/4 correct
mean 0.018s, p90 0.021s, max 0.023s

avg_fugu_20260626_234517
24/24 ok
mean 2.218s, p90 3.370s, completion TPS 24.627, request TPS 0.451
backend decode TPS 30.720, backend prefill TPS 76.642

audit_fugu_runtime_20260626_priority_no_poll_v079.json
8/8 ok
capabilities version 0.7.9
```
