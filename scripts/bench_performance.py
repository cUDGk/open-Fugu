from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BACKEND = "http://127.0.0.1:8080/v1/chat/completions"
FRONTDOOR = "http://127.0.0.1:9000/v1/chat/completions"


@dataclass(frozen=True)
class BenchCase:
    name: str
    url: str
    model: str
    prompt: str
    max_tokens: int = 96


CASES = [
    BenchCase(
        "backend_gemma_hot_or_cold",
        BACKEND,
        "google_gemma-4-26B-A4B-it-Q4_K_M",
        "日本語で一文だけ答えて。local Fugu frontdoorの役割は何ですか。",
    ),
    BenchCase(
        "backend_qwen_cold",
        BACKEND,
        "Qwen_Qwen3.6-35B-A3B-Q4_K_M",
        "Pythonで小さな関数を実装するときの注意点を日本語で一文だけ答えて。",
    ),
    BenchCase(
        "backend_qwen_hot_repeat",
        BACKEND,
        "Qwen_Qwen3.6-35B-A3B-Q4_K_M",
        "Pythonで小さな関数を実装するときの注意点を日本語で一文だけ答えて。",
    ),
    BenchCase(
        "backend_glm_cold",
        BACKEND,
        "GLM-4.7-Flash-UD-Q4_K_XL",
        "次の矛盾を日本語で一文だけ指摘して。AはBより大きい。BはAより大きい。",
    ),
    BenchCase(
        "backend_glm_hot_repeat",
        BACKEND,
        "GLM-4.7-Flash-UD-Q4_K_XL",
        "次の矛盾を日本語で一文だけ指摘して。AはBより大きい。BはAより大きい。",
    ),
    BenchCase(
        "frontdoor_fast_general_cold_to_gemma",
        FRONTDOOR,
        "local-moe-fugu:fast",
        "日本語で一文だけ答えて。local Fugu frontdoorの役割は何ですか。",
    ),
    BenchCase(
        "frontdoor_fast_general_hot_repeat",
        FRONTDOOR,
        "local-moe-fugu:fast",
        "日本語で一文だけ答えて。local Fugu frontdoorの役割は何ですか。",
    ),
    BenchCase(
        "frontdoor_fast_code_qwen",
        FRONTDOOR,
        "local-moe-fugu:fast",
        "Pythonで小さな関数を実装するときの注意点を日本語で一文だけ答えて。",
    ),
    BenchCase(
        "frontdoor_balanced_code_with_verify",
        FRONTDOOR,
        "local-moe-fugu",
        "Pythonで小さな関数を実装するときの注意点を日本語で一文だけ答えて。",
    ),
    BenchCase(
        "frontdoor_deep_small_code",
        FRONTDOOR,
        "local-moe-fugu:deep",
        "Pythonで足し算関数を安全に実装する方針を、日本語で二文以内にまとめて。",
    ),
]


def post_json(url: str, payload: dict[str, Any], timeout: int = 1200) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_case(case: BenchCase) -> dict[str, Any]:
    payload = {
        "model": case.model,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0,
        "max_tokens": case.max_tokens,
    }
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    try:
        response = post_json(case.url, payload)
        elapsed = time.perf_counter() - t0
        msg = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage", {}) or {}
        fp = response.get("system_fingerprint", {}) or {}
        route = fp.get("route", {}) if isinstance(fp, dict) else {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return {
            "started": started,
            "name": case.name,
            "ok": True,
            "model": case.model,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or 0),
            "completion_tok_s": round(completion_tokens / elapsed, 3) if elapsed > 0 else 0,
            "route_mode": route.get("mode", ""),
            "route_primary": route.get("primary", ""),
            "route_workflow": route.get("workflow", ""),
            "preview": " ".join(str(msg).split())[:240],
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "started": started,
            "name": case.name,
            "ok": False,
            "model": case.model,
            "elapsed_s": round(elapsed, 3),
            "error": f"HTTP {exc.code}: {body[:500]}",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "started": started,
            "name": case.name,
            "ok": False,
            "model": case.model,
            "elapsed_s": round(elapsed, 3),
            "error": repr(exc),
        }


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    rows = []
    for case in CASES:
        print(f"RUN {case.name} [{case.model}]", flush=True)
        row = run_case(case)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"perf_{stamp}.jsonl"
    csv_path = out_dir / f"perf_{stamp}.csv"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"WROTE {jsonl_path}")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()
