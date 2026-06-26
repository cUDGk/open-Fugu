from __future__ import annotations

import json
import urllib.request


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    payload = {
        "model": "local-moe-fugu:fast",
        "messages": [{"role": "user", "content": "OKだけ返して。"}],
    }
    out = post_json("http://127.0.0.1:9000/v1/chat/completions", payload)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
