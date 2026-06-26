# Sources checked while preparing this package

These are included so another LLM can verify assumptions instead of treating this package as opaque memory.

## llama.cpp server

- llama.cpp server README: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
  - OpenAI-compatible chat completions / responses / embeddings.
  - Parallel decoding, continuous batching, schema-constrained JSON response, tool/function calling.
- Hugging Face ggml-org blog on llama.cpp Model Management: https://huggingface.co/blog/ggml-org/model-management-in-llamacpp
  - Router mode, dynamic load/unload, `--models-dir`, `--models-max`, LRU eviction.
- Debian llama-server manpage: https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html
  - KV cache options `-ctk`, `-ctv`, mmap/mlock notes.

## Hardware

- AMD Ryzen 9 7940HS specs: https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7940hs.html
  - 2 memory channels, DDR5-5600/LPDDR5x, Radeon 780M, AVX512, Ryzen AI.

## Models / GGUFs

- Gemma 4 26B A4B GGUF: https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF
  - Q4_K_M file size ~17.04GB, IQ4_XS ~14.21GB.
- Qwen3.6 35B A3B GGUF: https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF
  - Q4_K_M file size ~22.29GB, IQ4_XS ~19.70GB.
- GLM-4.7-Flash GGUF: https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF
  - 30B-A3B MoE model, llama.cpp examples currently use UD-Q4_K_XL; Q4-class file sizes around 17-19GB depending quant.
