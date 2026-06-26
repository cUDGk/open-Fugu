from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRONTDOOR = "http://127.0.0.1:9000/v1/chat/completions"


@dataclass(frozen=True)
class BenchCase:
    name: str
    category: str
    prompt: str
    max_tokens: int = 128


CASES = [
    BenchCase(
        "general_one_sentence",
        "general",
        "Answer in one sentence: what is local MoE routing useful for?",
    ),
    BenchCase(
        "general_tradeoffs",
        "general",
        "Give two practical tradeoffs between small local LLMs and cloud APIs.",
    ),
    BenchCase(
        "rewrite_concise",
        "general",
        "Rewrite this more concisely: The system should avoid unnecessary model calls while keeping enough verification for risky tasks.",
    ),
    BenchCase(
        "ops_steps",
        "general",
        "List three steps to restart a local API service after changing Python code.",
    ),
    BenchCase(
        "workflow_design",
        "design",
        "Design a lightweight workflow for routing user requests among planner, coder, and verifier models. Keep it short.",
    ),
    BenchCase(
        "python_small_function",
        "code",
        "Write a Python function named clamp_tokens(requested, default, minimum) that returns a safe integer token limit.",
    ),
    BenchCase(
        "powershell_command",
        "code",
        "Give a PowerShell command that checks whether localhost port 9000 is listening, and briefly explain it.",
    ),
    BenchCase(
        "deep_code_review",
        "deep",
        "Review this Python function for bugs and suggest a fixed version:\n\n"
        "def avg(xs):\n"
        "    return sum(xs) / len(xs)\n\n"
        "Consider empty input and non-numeric values.",
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
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_case(
    case: BenchCase,
    repeat_index: int,
    url: str,
    timeout: int,
    max_tokens_override: int | None,
) -> dict[str, Any]:
    max_tokens = case.max_tokens if max_tokens_override is None else max_tokens_override
    payload = {
        "model": "local-moe-fugu",
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    try:
        response = post_json(url, payload, timeout)
        elapsed = time.perf_counter() - t0
        msg = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage", {}) or {}
        fp = response.get("system_fingerprint", {}) or {}
        route = fp.get("route", {}) if isinstance(fp, dict) else {}
        perf = fp.get("performance", {}) if isinstance(fp, dict) else {}
        if not isinstance(perf, dict):
            perf = {}
        backend_names = perf.get("backend_names") or []
        if not isinstance(backend_names, list):
            backend_names = []
        backend_key = "+".join(str(x) for x in backend_names) if backend_names else ""
        completion_tokens = int(usage.get("completion_tokens") or 0)
        completion_tps = round(completion_tokens / elapsed, 3) if elapsed > 0 else 0
        return {
            "started": started,
            "repeat": repeat_index,
            "name": case.name,
            "category": case.category,
            "ok": True,
            "model": "local-moe-fugu",
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or 0),
            "completion_tok_s": completion_tps,
            "completion_tps": completion_tps,
            "frontdoor_internal_ms": perf.get("end_to_end_ms", 0),
            "actual_completion_tps": perf.get("actual_completion_tps", 0),
            "actual_total_tps": perf.get("actual_total_tps", 0),
            "backend_names": backend_key,
            "backend_call_count_by_name": json.dumps(perf.get("backend_call_count_by_name", {}), sort_keys=True),
            "backend_fallback_count": perf.get("backend_fallback_count", 0),
            "theoretical_min_ms": perf.get("theoretical_min_ms", 0),
            "theoretical_completion_tps": perf.get("theoretical_completion_tps", 0),
            "backend_decode_tps": perf.get("backend_decode_tps", 0),
            "backend_prefill_tps": perf.get("backend_prefill_tps", 0),
            "backend_prompt_ms": perf.get("backend_prompt_ms", 0),
            "backend_predicted_ms": perf.get("backend_predicted_ms", 0),
            "backend_prompt_tokens": perf.get("backend_prompt_tokens", 0),
            "backend_predicted_tokens": perf.get("backend_predicted_tokens", 0),
            "overhead_ms": perf.get("overhead_ms", 0),
            "efficiency_vs_backend_theory": perf.get("efficiency_vs_backend_theory", 0),
            "route_mode": route.get("mode", ""),
            "route_primary": route.get("primary", ""),
            "route_workflow": route.get("workflow", ""),
            "request_max_tokens": fp.get("request_max_tokens") if isinstance(fp, dict) else None,
            "answer_chars": len(str(msg)),
            "preview": " ".join(str(msg).split())[:240],
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "started": started,
            "repeat": repeat_index,
            "name": case.name,
            "category": case.category,
            "ok": False,
            "model": "local-moe-fugu",
            "elapsed_s": round(elapsed, 3),
            "error": f"HTTP {exc.code}: {body[:500]}",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "started": started,
            "repeat": repeat_index,
            "name": case.name,
            "category": case.category,
            "ok": False,
            "model": "local-moe-fugu",
            "elapsed_s": round(elapsed, 3),
            "error": repr(exc),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]

    def group_summary(label: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        elapsed = [float(r["elapsed_s"]) for r in selected]
        total_tokens = sum(int(r.get("total_tokens") or 0) for r in selected)
        completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in selected)
        elapsed_sum = sum(elapsed)
        internal_ms_sum = sum(float(r.get("frontdoor_internal_ms") or 0) for r in selected)
        theoretical_ms_sum = sum(float(r.get("theoretical_min_ms") or 0) for r in selected)
        backend_prompt_ms_sum = sum(float(r.get("backend_prompt_ms") or 0) for r in selected)
        backend_predicted_ms_sum = sum(float(r.get("backend_predicted_ms") or 0) for r in selected)
        backend_prompt_tokens = sum(int(r.get("backend_prompt_tokens") or 0) for r in selected)
        backend_predicted_tokens = sum(int(r.get("backend_predicted_tokens") or 0) for r in selected)
        completion_tps = round(completion_tokens / elapsed_sum, 3) if elapsed_sum else 0
        total_tps = round(total_tokens / elapsed_sum, 3) if elapsed_sum else 0
        request_tps = round(len(selected) / elapsed_sum, 3) if elapsed_sum else 0
        theoretical_completion_tps = (
            round(completion_tokens / (theoretical_ms_sum / 1000), 3)
            if theoretical_ms_sum and completion_tokens
            else 0
        )
        backend_decode_tps = (
            round(backend_predicted_tokens / (backend_predicted_ms_sum / 1000), 3)
            if backend_predicted_ms_sum and backend_predicted_tokens
            else 0
        )
        backend_prefill_tps = (
            round(backend_prompt_tokens / (backend_prompt_ms_sum / 1000), 3)
            if backend_prompt_ms_sum and backend_prompt_tokens
            else 0
        )
        efficiency = round(theoretical_ms_sum / internal_ms_sum, 4) if theoretical_ms_sum and internal_ms_sum else 0
        return {
            "label": label,
            "count": len(selected),
            "mean_elapsed_s": round(statistics.mean(elapsed), 3) if elapsed else 0,
            "median_elapsed_s": round(statistics.median(elapsed), 3) if elapsed else 0,
            "p90_elapsed_s": round(percentile(elapsed, 0.9), 3) if elapsed else 0,
            "min_elapsed_s": round(min(elapsed), 3) if elapsed else 0,
            "max_elapsed_s": round(max(elapsed), 3) if elapsed else 0,
            "total_tokens": total_tokens,
            "completion_tokens": completion_tokens,
            "completion_tok_s_overall": completion_tps,
            "completion_tps": completion_tps,
            "total_tps": total_tps,
            "request_tps": request_tps,
            "frontdoor_internal_s": round(internal_ms_sum / 1000, 3),
            "theoretical_min_s": round(theoretical_ms_sum / 1000, 3),
            "theoretical_completion_tps": theoretical_completion_tps,
            "backend_decode_tps": backend_decode_tps,
            "backend_prefill_tps": backend_prefill_tps,
            "efficiency_vs_backend_theory": efficiency,
        }

    groups = [group_summary("overall", ok_rows)]
    for category in sorted({r["category"] for r in ok_rows}):
        groups.append(group_summary(f"category:{category}", [r for r in ok_rows if r["category"] == category]))
    for mode in sorted({r.get("route_mode", "") for r in ok_rows}):
        groups.append(group_summary(f"route_mode:{mode}", [r for r in ok_rows if r.get("route_mode", "") == mode]))
    for primary in sorted({r.get("route_primary", "") for r in ok_rows}):
        groups.append(
            group_summary(f"route_primary:{primary}", [r for r in ok_rows if r.get("route_primary", "") == primary])
        )
    for backend in sorted({r.get("backend_names", "") for r in ok_rows}):
        groups.append(
            group_summary(f"backend:{backend or 'none'}", [r for r in ok_rows if r.get("backend_names", "") == backend])
        )
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": len(rows),
        "ok_runs": len(ok_rows),
        "failed_runs": len(rows) - len(ok_rows),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--url", default=FRONTDOOR)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parents[1])
    rows: list[dict[str, Any]] = []
    for repeat_index in range(1, args.repeat + 1):
        for case in CASES:
            print(f"RUN repeat={repeat_index} case={case.name} category={case.category}", flush=True)
            row = run_case(case, repeat_index, args.url, args.timeout, args.max_tokens)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = summarize(rows)
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"avg_fugu_{stamp}.jsonl"
    csv_path = out_dir / f"avg_fugu_{stamp}.csv"
    summary_path = out_dir / f"avg_fugu_{stamp}.summary.json"

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


if __name__ == "__main__":
    main()
