# Model Selection

## Default model set

The default config uses HF tags instead of local filenames:

```yaml
gemma: bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M
qwen:  bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q4_K_M
glm:   unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL
```

Reason: llama.cpp can run `-hf repo:quant` directly, and router-mode cache discovery is simpler to explain than local filename matching.

## Why GLM uses UD-Q4_K_XL in the HF config

The Unsloth GLM-4.7-Flash GGUF page currently recommends `UD-Q4_K_XL` in its llama.cpp examples. The documentation also notes GLM-specific generation recommendations: repeat penalty 1.0, temperature/top-p settings, and min-p adjustment for llama.cpp.

For this project, GLM is used mostly as verifier/critic, so high sampling diversity is not needed. The conductor uses low temperature for verification and JSON output.

## Local filename mode

If using downloaded GGUF files under `C:\llm\models`, copy `configs/moe_fugu_local_example.yaml` and adjust model IDs to the exact IDs returned by:

```powershell
curl http://127.0.0.1:8080/v1/models
```

Do not guess IDs. llama.cpp router mode may expose HF IDs, filenames, or preset section names depending on launch mode.

## Quantization priority

Start with:

```text
Qwen Q4_K_M
Gemma Q4_K_M or IQ4_XS
GLM UD-Q4_K_XL / Q4-class
```

If memory pressure or paging appears, first reduce context size, then reduce Gemma quant, then Qwen quant. Qwen is the primary worker, so avoid degrading it first unless necessary.
