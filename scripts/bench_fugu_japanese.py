from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRONTDOOR = "http://127.0.0.1:9000/v1/chat/completions"
MODEL = "local-moe-fugu"


@dataclass(frozen=True)
class JapaneseCase:
    name: str
    prompt: str
    expected_markers: tuple[str, ...]
    expected_workflow: str = ""
    max_tokens: int = 512
    max_elapsed_s: float = 60.0


COMPOUND_A = """今日の曜日と200年3/4の曜日を教えてください
それと不定積分の解き方を教えてください
あと成人男性に対するアイスクリームの推奨量を教えてください
それと50m先の洗車場には徒歩と車のどっちで行った方が良いですか
これらの文章中に出てくる母音aの数を教えてください"""

COMPOUND_A_JA = """今日の曜日と200年3/4の曜日を教えてください
それと不定積分の解き方を教えてください
あと成人男性に対するアイスクリームの推奨量を教えてください
それと50m先の洗車場には徒歩と車のどっちで行った方が良いですか
これらの文章中に出てくる母音"あ"の数を教えてください"""


CASES = [
    JapaneseCase(
        name="compound_latin_a",
        prompt=COMPOUND_A,
        expected_workflow="deterministic_japanese_compound",
        expected_markers=("金曜日", "火曜日", "小文字のみで1個", "A/a合算で1個"),
        max_elapsed_s=0.5,
    ),
    JapaneseCase(
        name="compound_japanese_a",
        prompt=COMPOUND_A_JA,
        expected_workflow="deterministic_japanese_compound",
        expected_markers=("金曜日", "火曜日", "ひらがな「あ」は2個", "可視かなのあ段母音は16個"),
        max_elapsed_s=0.5,
    ),
    JapaneseCase(
        name="count_japanese_a_only",
        prompt='次の文章中に出てくる母音"あ"の数を教えてください。\nあした、カサを持たないあなたは歩いた。',
        expected_workflow="deterministic_count",
        expected_markers=("ひらがな「あ」は2個", "あ/ぁ/ア/ァは2個", "可視かなのあ段母音は11個"),
        max_elapsed_s=0.5,
    ),
    JapaneseCase(
        name="count_latin_a_only",
        prompt="この文章中に出てくる母音aの数を教えてください。banana and API",
        expected_workflow="deterministic_count",
        expected_markers=("小文字のみで5個", "A/a合算で6個"),
        max_elapsed_s=0.5,
    ),
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def marker_ok(answer: str, markers: tuple[str, ...]) -> tuple[bool, str]:
    missing = [marker for marker in markers if marker not in answer]
    return not missing, ", ".join(missing)


def run_case(case: JapaneseCase, url: str, timeout: int) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0,
        "max_tokens": case.max_tokens,
    }
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    try:
        response = post_json(url, payload, timeout)
        elapsed = time.perf_counter() - t0
        answer = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        usage = response.get("usage", {}) or {}
        fp = response.get("system_fingerprint", {}) or {}
        route = fp.get("route", {}) if isinstance(fp, dict) else {}
        perf = fp.get("performance", {}) if isinstance(fp, dict) else {}
        if not isinstance(perf, dict):
            perf = {}
        markers_passed, missing = marker_ok(answer, case.expected_markers)
        workflow = str(route.get("workflow", ""))
        workflow_passed = not case.expected_workflow or workflow == case.expected_workflow
        elapsed_passed = elapsed <= case.max_elapsed_s
        correct = markers_passed and workflow_passed and elapsed_passed
        return {
            "started": started,
            "name": case.name,
            "ok": True,
            "correct": correct,
            "markers_passed": markers_passed,
            "missing_markers": missing,
            "workflow_passed": workflow_passed,
            "elapsed_passed": elapsed_passed,
            "model": MODEL,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "route_mode": route.get("mode", ""),
            "route_primary": route.get("primary", ""),
            "route_workflow": workflow,
            "router": route.get("router", ""),
            "model_call_count": perf.get("model_call_count", 0),
            "theoretical_min_ms": perf.get("theoretical_min_ms", 0),
            "overhead_ms": perf.get("overhead_ms", 0),
            "preview": re.sub(r"\s+", " ", answer)[:280],
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "started": started,
            "name": case.name,
            "ok": False,
            "correct": False,
            "elapsed_s": round(elapsed, 3),
            "error": f"HTTP {exc.code}: {body[:500]}",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "started": started,
            "name": case.name,
            "ok": False,
            "correct": False,
            "elapsed_s": round(elapsed, 3),
            "error": repr(exc),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]
    elapsed = [float(r["elapsed_s"]) for r in ok_rows]
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "fugu_japanese_regression",
        "target": {"url": FRONTDOOR, "model": MODEL},
        "runs": len(rows),
        "ok_runs": len(ok_rows),
        "failed_runs": len(rows) - len(ok_rows),
        "correct": sum(1 for r in rows if r.get("correct")),
        "accuracy": round(sum(1 for r in rows if r.get("correct")) / len(rows), 3) if rows else 0,
        "mean_elapsed_s": round(statistics.mean(elapsed), 3) if elapsed else 0,
        "median_elapsed_s": round(statistics.median(elapsed), 3) if elapsed else 0,
        "p90_elapsed_s": round(percentile(elapsed, 0.9), 3) if elapsed else 0,
        "max_elapsed_s": round(max(elapsed), 3) if elapsed else 0,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=FRONTDOOR)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parents[1])
    rows: list[dict[str, Any]] = []
    for case in CASES:
        print(f"RUN case={case.name}", flush=True)
        row = run_case(case, args.url, args.timeout)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = summarize(rows)
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"japanese_fugu_{stamp}.jsonl"
    csv_path = out_dir / f"japanese_fugu_{stamp}.csv"
    summary_path = out_dir / f"japanese_fugu_{stamp}.summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"WROTE {jsonl_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {summary_path}")

    if summary["accuracy"] < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
