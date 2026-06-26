# Hardware Profile

## User machine

| Item | Spec |
|---|---|
| OS | Windows 11 Pro Build 26200 |
| CPU | AMD Ryzen 9 7940HS |
| CPU cores | 8 cores / 16 threads |
| RAM | 64GB DDR5-5600 |
| SSD | 1TB NVMe |
| Form factor | Micro Computer (HK) Venus series mini PC |

## LLM-relevant interpretation

This machine should be treated as:

```text
CPU + shared-memory iGPU + 64GB RAM
```

not as:

```text
large dedicated VRAM GPU workstation
```

The Radeon 780M may help in some llama.cpp builds, but because it uses shared memory, it should be benchmarked rather than assumed to outperform CPU-only execution.

## Current llama.cpp settings

```powershell
llama-server `
  --host 127.0.0.1 `
  --port 8080 `
  --models-max 2 `
  -c 16384 `
  -np 1 `
  --cont-batching `
  --jinja `
  --cache-prompt `
  --cache-idle-slots `
  --reasoning off `
  --reasoning-budget 0 `
  -ctk q8_0 `
  -ctv q8_0 `
  -t 12 `
  -b 512 `
  -ub 128 `
  --repeat-penalty 1.0 `
  --spec-type ngram-cache `
  --spec-ngram-mod-n-min 1 `
  --spec-ngram-mod-n-max 8 `
  --spec-ngram-mod-n-match 24
```

## Tuning tiers

### Conservative

```text
ctx: 8192
KV: q8_0/q8_0
threads: 10-12
batch: 256-512
ubatch: 64-128
models-max: 1
quant:
  Gemma: IQ4_XS or Q4_K_S
  Qwen: IQ4_XS
  GLM: UD-Q4_K_XL or Q4_K_S-level
```

### Standard

```text
ctx: 12288
KV: q8_0/q8_0
threads: 12
batch: 512
ubatch: 128
models-max: 2
quant:
  Gemma: Q4_K_M
  Qwen: Q4_K_M
  GLM: UD-Q4_K_XL / Q4-class
```

### Quality priority

```text
ctx: 16384
KV: q8_0/q8_0, q4_0 only if memory pressure appears
threads: 12-14
batch: 512
ubatch: 128
models-max: 2
quant:
  Gemma: Q4_K_M or Q5_K_M if memory allows
  Qwen: Q4_K_M
  GLM: Q4-class or better
```

## Checks to run on Windows

```powershell
# RAM module / dual-channel sanity check
Get-CimInstance Win32_PhysicalMemory |
  Select-Object BankLabel, Capacity, Speed, ConfiguredClockSpeed, Manufacturer, PartNumber

# llama.cpp device visibility
llama-server --list-devices
```

If the 64GB RAM is a single DIMM rather than 2x32GB, CPU/iGPU inference can be materially worse because memory bandwidth is reduced.
