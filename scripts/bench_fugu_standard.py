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
class StandardCase:
    name: str
    domain: str
    scorer: str
    expected: str
    prompt: str
    max_tokens: int = 48


CASES = [
    StandardCase(
        "gsm_lite_apples",
        "math",
        "number",
        "26",
        "Return exactly `ANSWER: <integer>`.\nA store sells apples for 4 dollars each and oranges for 7 dollars each. If I buy 3 apples and 2 oranges, what is the total cost?",
    ),
    StandardCase(
        "gsm_lite_percentage",
        "math",
        "number",
        "36",
        "Return exactly `ANSWER: <integer>`.\nWhat is 15 percent of 240?",
    ),
    StandardCase(
        "gsm_lite_linear",
        "math",
        "number",
        "11",
        "Return exactly `ANSWER: <integer>`.\nIf x = 3 and y = 2*x + 5, what is y?",
    ),
    StandardCase(
        "gsm_lite_rate",
        "math",
        "number",
        "45",
        "Return exactly `ANSWER: <integer>`.\nA car travels 90 kilometers in 2 hours at constant speed. How many kilometers does it travel in 1 hour?",
    ),
    StandardCase(
        "gsm_lite_average",
        "math",
        "number",
        "8",
        "Return exactly `ANSWER: <integer>`.\nThe numbers are 5, 7, 8, and 12. What is their arithmetic mean?",
    ),
    StandardCase(
        "logic_syllogism",
        "logic",
        "choice",
        "A",
        "Return exactly `ANSWER: <letter>`.\nAll whales are mammals. All mammals breathe air. Therefore whales breathe air.\nA) true\nB) false\nC) cannot be determined",
    ),
    StandardCase(
        "logic_ordering",
        "logic",
        "choice",
        "A",
        "Return exactly `ANSWER: <letter>`.\nAlice is taller than Ben. Ben is taller than Cara. Who is tallest?\nA) Alice\nB) Ben\nC) Cara\nD) cannot be determined",
    ),
    StandardCase(
        "logic_calendar",
        "logic",
        "choice",
        "C",
        "Return exactly `ANSWER: <letter>`.\nIf today is Monday, what day is it 9 days from today?\nA) Monday\nB) Tuesday\nC) Wednesday\nD) Thursday",
    ),
    StandardCase(
        "logic_negation",
        "logic",
        "choice",
        "B",
        "Return exactly `ANSWER: <letter>`.\nStatement: Not every request needs the largest model. Which means the same thing?\nA) Every request needs the largest model\nB) Some requests do not need the largest model\nC) No request can use the largest model\nD) Only code requests need the largest model",
    ),
    StandardCase(
        "logic_set_membership",
        "logic",
        "choice",
        "D",
        "Return exactly `ANSWER: <letter>`.\nEvery item in Box A is red. This ball is red. What can we conclude?\nA) The ball is definitely in Box A\nB) The ball is definitely not in Box A\nC) Box A has no balls\nD) The ball may or may not be in Box A",
    ),
    StandardCase(
        "knowledge_http_404",
        "knowledge",
        "choice",
        "B",
        "Return exactly `ANSWER: <letter>`.\nIn HTTP, status code 404 usually means:\nA) OK\nB) Not Found\nC) Internal Server Error\nD) Unauthorized",
    ),
    StandardCase(
        "knowledge_dns",
        "knowledge",
        "choice",
        "A",
        "Return exactly `ANSWER: <letter>`.\nDNS is mainly used to:\nA) map domain names to IP addresses\nB) compress images\nC) encrypt disk blocks\nD) schedule CPU threads",
    ),
    StandardCase(
        "knowledge_acid_durability",
        "knowledge",
        "choice",
        "D",
        "Return exactly `ANSWER: <letter>`.\nIn database ACID properties, durability means:\nA) queries are always fast\nB) data is always normalized\nC) transactions run in parallel\nD) committed changes survive failures",
    ),
    StandardCase(
        "knowledge_tls",
        "knowledge",
        "choice",
        "C",
        "Return exactly `ANSWER: <letter>`.\nTLS is commonly used to provide:\nA) image rendering\nB) file deduplication\nC) encrypted and authenticated network communication\nD) CPU branch prediction",
    ),
    StandardCase(
        "knowledge_cache",
        "knowledge",
        "choice",
        "A",
        "Return exactly `ANSWER: <letter>`.\nA CPU cache primarily helps by:\nA) keeping frequently used data close to the processor\nB) replacing the operating system\nC) increasing monitor resolution\nD) changing source code syntax",
    ),
    StandardCase(
        "code_python_slice",
        "code",
        "choice",
        "B",
        "Return exactly `ANSWER: <letter>`.\nIn Python, what is the value of [1, 2, 3, 4][1:3]?\nA) [1, 2]\nB) [2, 3]\nC) [3, 4]\nD) [1, 2, 3]",
    ),
    StandardCase(
        "code_big_o_binary_search",
        "code",
        "choice",
        "C",
        "Return exactly `ANSWER: <letter>`.\nWhat is the usual time complexity of binary search on a sorted array?\nA) O(1)\nB) O(n)\nC) O(log n)\nD) O(n^2)",
    ),
    StandardCase(
        "code_python_truthy",
        "code",
        "choice",
        "B",
        "Return exactly `ANSWER: <letter>`.\nIn Python, bool([]) evaluates to:\nA) True\nB) False\nC) None\nD) []",
    ),
    StandardCase(
        "code_exception_kind",
        "code",
        "choice",
        "A",
        "Return exactly `ANSWER: <letter>`.\nIn Python, int('abc') raises which common exception?\nA) ValueError\nB) KeyError\nC) IndexError\nD) StopIteration",
    ),
    StandardCase(
        "code_sql_filter",
        "code",
        "choice",
        "D",
        "Return exactly `ANSWER: <letter>`.\nIn SQL, which clause filters rows before grouping?\nA) ORDER BY\nB) HAVING\nC) LIMIT\nD) WHERE",
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


def extract_choice(text: str) -> str | None:
    matches = re.findall(r"ANSWER\s*:\s*([A-D])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    stripped = text.strip()
    if re.fullmatch(r"[A-Da-d]", stripped):
        return stripped.upper()
    first_line = stripped.splitlines()[0] if stripped else ""
    match = re.search(r"\b([A-D])\b", first_line, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_number(text: str) -> str | None:
    matches = re.findall(r"ANSWER\s*:\s*(-?\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    if matches:
        value = matches[-1]
        if re.fullmatch(r"-?\d+\.0+", value):
            return str(int(float(value)))
        return value
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    value = numbers[-1]
    if re.fullmatch(r"-?\d+\.0+", value):
        return str(int(float(value)))
    return value


def score_answer(case: StandardCase, text: str) -> tuple[bool, str]:
    if case.scorer == "choice":
        observed = extract_choice(text) or ""
    elif case.scorer == "number":
        observed = extract_number(text) or ""
    else:
        raise ValueError(f"unknown scorer: {case.scorer}")
    return observed == case.expected, observed


def run_case(case: StandardCase, url: str, timeout: int, max_tokens_override: int | None) -> dict[str, Any]:
    max_tokens = case.max_tokens if max_tokens_override is None else max_tokens_override
    payload = {
        "model": MODEL,
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
        correct, observed = score_answer(case, str(msg))
        return {
            "started": started,
            "name": case.name,
            "domain": case.domain,
            "ok": True,
            "correct": correct,
            "expected": case.expected,
            "observed": observed,
            "model": MODEL,
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
            "name": case.name,
            "domain": case.domain,
            "ok": False,
            "correct": False,
            "model": MODEL,
            "elapsed_s": round(elapsed, 3),
            "error": f"HTTP {exc.code}: {body[:500]}",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "started": started,
            "name": case.name,
            "domain": case.domain,
            "ok": False,
            "correct": False,
            "model": MODEL,
            "elapsed_s": round(elapsed, 3),
            "error": repr(exc),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]

    def group_summary(label: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        elapsed = [float(r["elapsed_s"]) for r in selected]
        correct = sum(1 for r in selected if r.get("correct"))
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
            "correct": correct,
            "accuracy": round(correct / len(selected), 3) if selected else 0,
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
    for domain in sorted({r["domain"] for r in ok_rows}):
        groups.append(group_summary(f"domain:{domain}", [r for r in ok_rows if r["domain"] == domain]))
    for primary in sorted({r.get("route_primary", "") for r in ok_rows}):
        groups.append(group_summary(f"route_primary:{primary}", [r for r in ok_rows if r.get("route_primary") == primary]))
    for backend in sorted({r.get("backend_names", "") for r in ok_rows}):
        groups.append(group_summary(f"backend:{backend or 'none'}", [r for r in ok_rows if r.get("backend_names", "") == backend]))
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "fugu_standard_lite",
        "target": {"url": FRONTDOOR, "model": MODEL},
        "runs": len(rows),
        "ok_runs": len(ok_rows),
        "failed_runs": len(rows) - len(ok_rows),
        "groups": groups,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=FRONTDOOR)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parents[1])
    selected = CASES[: args.limit] if args.limit else CASES
    rows: list[dict[str, Any]] = []

    for case in selected:
        print(f"RUN case={case.name} domain={case.domain}", flush=True)
        row = run_case(case, args.url, args.timeout, args.max_tokens)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = summarize(rows)
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"standard_fugu_{stamp}.jsonl"
    csv_path = out_dir / f"standard_fugu_{stamp}.csv"
    summary_path = out_dir / f"standard_fugu_{stamp}.summary.json"

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
