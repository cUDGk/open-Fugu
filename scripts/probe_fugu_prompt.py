from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


FRONTDOOR = "http://127.0.0.1:9000/v1/chat/completions"


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_b64:
        return base64.b64decode(args.prompt_b64).decode("utf-8")
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    return sys.stdin.read()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=FRONTDOOR)
    parser.add_argument("--model", default="local-moe-fugu")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-b64", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": read_prompt(args)}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    t0 = time.perf_counter()
    response = post_json(args.url, payload, args.timeout)
    elapsed_s = time.perf_counter() - t0

    if args.full:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return

    fp = response.get("system_fingerprint", {}) or {}
    route = fp.get("route", {}) if isinstance(fp, dict) else {}
    perf = fp.get("performance", {}) if isinstance(fp, dict) else {}
    out = {
        "elapsed_s_external": round(elapsed_s, 3),
        "answer": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "usage": response.get("usage", {}),
        "route": route,
        "performance": perf,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
