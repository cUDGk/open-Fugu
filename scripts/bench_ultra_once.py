from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    payload = {
        "model": "local-moe-fugu:ultra",
        "messages": [
            {
                "role": "user",
                "content": "Pythonで足し算関数を安全に実装する方針を、日本語で一文だけ答えて。",
            }
        ],
        "temperature": 0,
        "max_tokens": 96,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:9000/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    usage = out.get("usage", {}) or {}
    row = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": "frontdoor_ultra_small_code",
        "ok": True,
        "model": "local-moe-fugu:ultra",
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "completion_tok_s": round((int(usage.get("completion_tokens") or 0) / elapsed), 3),
        "route": (out.get("system_fingerprint", {}) or {}).get("route", {}),
        "preview": " ".join(out["choices"][0]["message"]["content"].split())[:500],
    }
    out_path = Path("logs") / f"perf_ultra_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
