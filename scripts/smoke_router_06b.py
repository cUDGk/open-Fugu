from __future__ import annotations

import json
import time
import urllib.request
from typing import Any


URL = "http://127.0.0.1:9000/v1/chat/completions"


PROMPTS = [
    "Answer in one sentence: what is local MoE routing useful for?",
    "Write a Python function named clamp_tokens(requested, default, minimum).",
    "Review this answer for correctness and identify any missing edge cases.",
]


def post_json(payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    for prompt in PROMPTS:
        payload = {
            "model": "local-moe-fugu",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 96,
        }
        t0 = time.perf_counter()
        response = post_json(payload)
        elapsed = time.perf_counter() - t0
        fingerprint = response.get("system_fingerprint", {}) or {}
        route = fingerprint.get("route", {}) or {}
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(
            json.dumps(
                {
                    "prompt": prompt,
                    "elapsed_s": round(elapsed, 3),
                    "router": route.get("router"),
                    "primary": route.get("primary"),
                    "mode": route.get("mode"),
                    "task_type": route.get("task_type"),
                    "confidence": route.get("confidence"),
                    "router_elapsed_ms": route.get("router_elapsed_ms"),
                    "answer_preview": " ".join(str(text).split())[:160],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
