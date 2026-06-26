from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditCase:
    name: str
    messages: list[dict[str, str]]
    model: str | None = None
    max_tokens: int = 128
    expect_mode: str | None = None
    expect_workflow: str | None = None
    expect_primary: str | None = None
    expect_model_calls: int | None = None
    min_model_calls: int | None = None
    max_model_calls: int | None = None
    expect_performance: bool = True
    max_elapsed_s: float | None = None
    markers: tuple[str, ...] = ()
    forbidden_markers: tuple[str, ...] = ()


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_bytes(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, str], str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), dict(resp.headers.items()), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), dict(exc.headers.items()), body


def get_json(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def route_of(resp: dict[str, Any]) -> dict[str, Any]:
    return resp.get("system_fingerprint", {}).get("route", {}) or {}


def perf_of(resp: dict[str, Any]) -> dict[str, Any]:
    return resp.get("system_fingerprint", {}).get("performance", {}) or {}


def answer_of(resp: dict[str, Any]) -> str:
    return resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


def capability_row(capabilities: dict[str, Any], model: str, error: str | None = None) -> dict[str, Any]:
    checks: list[str] = []
    if error:
        checks.append(f"capability endpoint failed: {error}")

    aliases = set(capabilities.get("mode_aliases") or [])
    expected_aliases = {model, f"{model}:fast", f"{model}:deep", f"{model}:ultra"}
    missing_aliases = sorted(expected_aliases - aliases)
    if missing_aliases:
        checks.append(f"missing mode aliases: {', '.join(missing_aliases)}")

    workers = capabilities.get("workers") or {}
    missing_workers = sorted({"gemma", "qwen", "glm"} - set(workers))
    if missing_workers:
        checks.append(f"missing workers: {', '.join(missing_workers)}")

    control = capabilities.get("control") or {}
    router_06b = control.get("router_06b") or {}
    if not router_06b.get("enabled"):
        checks.append("capability router_06b.enabled is false")
    if not router_06b.get("loaded"):
        checks.append("capability router_06b.loaded is false")
    for key in ("rules_fallback", "explicit_mode_skips_tower", "long_lookup_skips_tower_after_pack"):
        if control.get(key) is not True:
            checks.append(f"control.{key} expected true, got {control.get(key)!r}")

    context = capabilities.get("context") or {}
    if context.get("long_lookup_retrieval_pack") is not True:
        checks.append("context.long_lookup_retrieval_pack expected true")
    if int(context.get("retrieval_pack_lookup_cap_chars") or 0) < 3500:
        checks.append("retrieval_pack_lookup_cap_chars too small")

    deterministic = set(capabilities.get("deterministic_paths") or [])
    required_deterministic = {
        "exact_literal",
        "japanese_compound",
        "visible_count",
        "arithmetic",
        "calendar_multiple_choice",
    }
    missing_det = sorted(required_deterministic - deterministic)
    if missing_det:
        checks.append(f"missing deterministic paths: {', '.join(missing_det)}")

    quality = capabilities.get("quality_workflows") or {}
    required_quality = {
        "fast",
        "balanced",
        "deep_full",
        "ultra_full",
        "deep_compact",
        "ultra_compact",
    }
    missing_quality = sorted(required_quality - set(quality))
    if missing_quality:
        checks.append(f"missing quality workflows: {', '.join(missing_quality)}")

    tools = capabilities.get("tools") or {}
    allowed_prefixes = tools.get("allowed_prefixes") or []
    if ["python", "-m", "unittest"] not in allowed_prefixes:
        checks.append("tool whitelist missing python -m unittest")
    if tools.get("tool_grounded_repair") is not True:
        checks.append("tools.tool_grounded_repair expected true")

    runtime = capabilities.get("runtime") or {}
    if runtime.get("anti_fanout_lock") is not True:
        checks.append("runtime.anti_fanout_lock expected true")
    if runtime.get("parallel_model_calls") is not False:
        checks.append("runtime.parallel_model_calls expected false for this mini PC profile")

    telemetry = capabilities.get("telemetry") or {}
    for key in (
        "route_in_system_fingerprint",
        "performance_in_system_fingerprint",
        "actual_tps",
        "theoretical_tps",
        "backend_prefill_decode_tps",
        "tool_call_metrics",
    ):
        if telemetry.get(key) is not True:
            checks.append(f"telemetry.{key} expected true")

    return {
        "name": "runtime_capabilities",
        "ok": not checks,
        "checks": checks,
        "version": capabilities.get("version"),
        "aliases": sorted(aliases),
        "workers": sorted(workers),
        "router_06b": router_06b,
        "context": context,
        "quality_workflows": quality,
        "tool_allowed_prefixes": allowed_prefixes,
    }


def run_api_checks(chat_url: str, model: str, timeout: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    status, headers, body = post_bytes(
        chat_url,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Return exactly `FUGU_STREAM_OK`."}],
            "max_completion_tokens": 16,
            "temperature": 0.0,
            "stream": True,
        },
        timeout,
    )
    stream_checks: list[str] = []
    if status != 200:
        stream_checks.append(f"HTTP status expected 200, got {status}")
    content_type = headers.get("Content-Type", headers.get("content-type", ""))
    if "text/event-stream" not in content_type:
        stream_checks.append(f"Content-Type expected text/event-stream, got {content_type!r}")
    if "FUGU_STREAM_OK" not in body:
        stream_checks.append("missing streamed content marker")
    if "system_fingerprint" not in body:
        stream_checks.append("missing final stream system_fingerprint")
    if "data: [DONE]" not in body:
        stream_checks.append("missing [DONE] event")
    rows.append(
        {
            "name": "api_stream_sse",
            "ok": not stream_checks,
            "checks": stream_checks,
            "status": status,
            "content_type": content_type,
            "body_preview": body[:400],
        }
    )

    status, _headers, body = post_bytes(
        chat_url,
        {"model": model, "messages": "not-a-list", "max_tokens": 16, "temperature": 0.0},
        timeout,
    )
    rows.append(
        {
            "name": "api_invalid_messages_400",
            "ok": status == 400 and "messages must be a list" in body,
            "checks": []
            if status == 400 and "messages must be a list" in body
            else [f"expected HTTP 400 messages error, got status={status} body={body[:200]!r}"],
            "status": status,
            "body_preview": body[:240],
        }
    )

    status, _headers, body = post_bytes(
        chat_url,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Return exactly `BAD_TOOL_SHOULD_NOT_RUN`."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "fugu_tools": {"cwd": ".", "commands": [["cmd", "/c", "echo", "bad"]], "timeout_s": 5},
        },
        timeout,
    )
    rows.append(
        {
            "name": "api_tool_policy_rejects_bad_command",
            "ok": status == 400 and "command not allowed" in body,
            "checks": []
            if status == 400 and "command not allowed" in body
            else [f"expected HTTP 400 command-not-allowed, got status={status} body={body[:200]!r}"],
            "status": status,
            "body_preview": body[:240],
        }
    )

    with tempfile.TemporaryDirectory(prefix="fugu_audit_unittest_") as tmp:
        root = Path(tmp)
        (root / "test_ok.py").write_text(
            "import unittest\n\n"
            "class FuguAuditTest(unittest.TestCase):\n"
            "    def test_truth(self):\n"
            "        self.assertEqual(2 + 2, 4)\n",
            encoding="utf-8",
        )
        tool_resp = post_json(
            chat_url,
            {
                "model": f"{model}:fast",
                "messages": [{"role": "user", "content": "In one sentence, say what a unit test is."}],
                "max_completion_tokens": 48,
                "temperature": 0.0,
                "fugu_tools": {
                    "cwd": str(root),
                    "commands": [["python", "-m", "unittest", "discover", "-q"]],
                    "timeout_s": 30,
                },
            },
            timeout,
        )
    fp = tool_resp.get("system_fingerprint", {}) or {}
    perf = fp.get("performance", {}) or {}
    tool_checks: list[str] = []
    if fp.get("request_max_tokens") != 48:
        tool_checks.append(f"request_max_tokens expected 48, got {fp.get('request_max_tokens')}")
    if int(perf.get("tool_call_count") or 0) != 1:
        tool_checks.append(f"tool_call_count expected 1, got {perf.get('tool_call_count')}")
    if int(perf.get("tool_fail_count") or 0) != 0:
        tool_checks.append(f"tool_fail_count expected 0, got {perf.get('tool_fail_count')}")
    if not answer_of(tool_resp).strip():
        tool_checks.append("empty tool-backed answer")
    rows.append(
        {
            "name": "api_tool_unittest_success",
            "ok": not tool_checks,
            "checks": tool_checks,
            "workflow": route_of(tool_resp).get("workflow"),
            "mode": route_of(tool_resp).get("mode"),
            "tool_call_count": perf.get("tool_call_count"),
            "tool_fail_count": perf.get("tool_fail_count"),
            "tool_call_wall_ms": perf.get("tool_call_wall_ms"),
            "request_max_tokens": fp.get("request_max_tokens"),
            "answer_preview": answer_of(tool_resp)[:240],
        }
    )
    return rows


def run_case(case: AuditCase, chat_url: str, model: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    request_model = case.model or model
    resp = post_json(
        chat_url,
        {
            "model": request_model,
            "messages": case.messages,
            "max_tokens": case.max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    route = route_of(resp)
    perf = perf_of(resp)
    answer = answer_of(resp)
    answer_lower = answer.lower()

    checks: list[str] = []
    model_calls = int(perf.get("model_call_count") or 0)
    if case.expect_mode and route.get("mode") != case.expect_mode:
        checks.append(f"mode expected {case.expect_mode}, got {route.get('mode')}")
    if case.expect_workflow and route.get("workflow") != case.expect_workflow:
        checks.append(f"workflow expected {case.expect_workflow}, got {route.get('workflow')}")
    if case.expect_primary and route.get("primary") != case.expect_primary:
        checks.append(f"primary expected {case.expect_primary}, got {route.get('primary')}")
    if case.expect_model_calls is not None and model_calls != case.expect_model_calls:
        checks.append(
            f"model_call_count expected {case.expect_model_calls}, got {perf.get('model_call_count')}"
        )
    if case.min_model_calls is not None and model_calls < case.min_model_calls:
        checks.append(f"model_call_count expected >= {case.min_model_calls}, got {model_calls}")
    if case.max_model_calls is not None and model_calls > case.max_model_calls:
        checks.append(f"model_call_count expected <= {case.max_model_calls}, got {model_calls}")
    if case.max_elapsed_s is not None and elapsed > case.max_elapsed_s:
        checks.append(f"elapsed expected <= {case.max_elapsed_s:.3f}s, got {elapsed:.3f}s")
    if case.expect_performance:
        required_perf = ("model_call_count", "end_to_end_ms", "efficiency_vs_backend_theory")
        missing_perf = [key for key in required_perf if key not in perf]
        if missing_perf:
            checks.append(f"missing performance keys: {', '.join(missing_perf)}")
        if int(perf.get("model_call_count") or 0) > 0:
            model_perf = ("actual_completion_tps", "theoretical_completion_tps", "backend_decode_tps")
            missing_model_perf = [key for key in model_perf if key not in perf]
            if missing_model_perf:
                checks.append(f"missing model performance keys: {', '.join(missing_model_perf)}")
    for marker in case.markers:
        if marker not in answer and marker.lower() not in answer_lower:
            checks.append(f"missing marker {marker!r}")
    for marker in case.forbidden_markers:
        if marker in answer:
            checks.append(f"forbidden marker {marker!r}")

    return {
        "name": case.name,
        "ok": not checks,
        "checks": checks,
        "elapsed_s_external": round(elapsed, 3),
        "request_model": request_model,
        "workflow": route.get("workflow"),
        "mode": route.get("mode"),
        "primary": route.get("primary"),
        "router": route.get("router"),
        "reason": route.get("reason"),
        "model_call_count": perf.get("model_call_count"),
        "actual_completion_tps": perf.get("actual_completion_tps"),
        "theoretical_completion_tps": perf.get("theoretical_completion_tps"),
        "efficiency_vs_backend_theory": perf.get("efficiency_vs_backend_theory"),
        "answer_preview": answer[:240],
        "messages": case.messages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the whole local-moe-fugu runtime.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="local-moe-fugu")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--include-long",
        action="store_true",
        help="Also run a long-context head/tail retention check. This is slower than the normal audit.",
    )
    parser.add_argument(
        "--include-modes",
        action="store_true",
        help="Also run fast/deep/ultra model-alias checks. Deep/ultra are compact when max_tokens is small.",
    )
    parser.add_argument(
        "--include-api",
        action="store_true",
        help="Also run API-surface checks for streaming, max_completion_tokens, validation, and fugu_tools.",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Also run the real non-compact deep/ultra workflows. This can be several minutes on the mini PC.",
    )
    parser.add_argument(
        "--only-heavy",
        action="store_true",
        help="Run only health, capabilities, and the real non-compact deep/ultra workflows.",
    )
    args = parser.parse_args()
    if args.only_heavy:
        args.include_heavy = True

    base = args.base_url.rstrip("/")
    chat_url = f"{base}/v1/chat/completions"
    health = get_json(f"{base}/health", args.timeout)
    models = get_json(f"{base}/v1/models", args.timeout)
    capability_error: str | None = None
    try:
        capabilities = get_json(f"{base}/fugu/capabilities", args.timeout)
    except Exception as exc:
        capabilities = {}
        capability_error = repr(exc)
    model_ids = [item.get("id") for item in models.get("data", [])]

    health_checks: list[str] = []
    if health.get("status") != "ok":
        health_checks.append(f"health status expected ok, got {health.get('status')}")
    if health.get("front_model") != args.model:
        health_checks.append(f"front_model expected {args.model}, got {health.get('front_model')}")
    router_06b = health.get("router_06b") or {}
    if not router_06b.get("enabled"):
        health_checks.append("router_06b is not enabled")
    if not router_06b.get("loaded"):
        health_checks.append("router_06b is not loaded")
    if router_06b.get("last_error"):
        health_checks.append(f"router_06b last_error is set: {router_06b.get('last_error')}")
    required_models = {args.model, f"{args.model}:fast", f"{args.model}:deep", f"{args.model}:ultra"}
    missing_models = sorted(required_models - set(model_ids))
    if missing_models:
        health_checks.append(f"missing exposed models: {', '.join(missing_models)}")

    cases = [
        AuditCase(
            name="deterministic_japanese_count",
            messages=[
                {
                    "role": "user",
                    "content": '次の文章中に出てくる母音"あ"の数を教えてください。\nあした、カサを持たないあなたは歩いた。',
                }
            ],
            expect_workflow="deterministic_count",
            expect_model_calls=0,
            expect_mode="fast",
            markers=("ひらがな「あ」は2個", "可視かなのあ段母音は11個"),
        ),
        AuditCase(
            name="deterministic_exact_literal",
            messages=[
                {
                    "role": "user",
                    "content": "Return exactly `FUGU_LITERAL_OK`.",
                }
            ],
            max_tokens=16,
            expect_workflow="deterministic_exact_literal",
            expect_model_calls=0,
            expect_mode="fast",
            max_elapsed_s=1.0,
            markers=("FUGU_LITERAL_OK",),
        ),
        AuditCase(
            name="qwen_fast_code",
            messages=[
                {
                    "role": "user",
                    "content": "Write a Python expression for the sum of squares from 1 to 5. Return only the expression.",
                }
            ],
            expect_mode="fast",
            expect_primary="qwen",
            forbidden_markers=("```",),
        ),
        AuditCase(
            name="gemma_japanese_general",
            messages=[{"role": "user", "content": "不定積分の考え方を日本語で2文だけで説明して。"}],
            expect_mode="fast",
            expect_primary="gemma",
        ),
        AuditCase(
            name="workflow_artifact_cleanup",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Design a lightweight workflow for routing user requests among "
                        "planner, coder, and verifier models. Keep it short."
                    ),
                }
            ],
            max_tokens=96,
            expect_mode="fast",
            markers=("planner", "coder", "verifier"),
            forbidden_markers=("Би", "```"),
        ),
    ]
    if args.only_heavy:
        cases = []

    if args.include_modes:
        cases.extend(
            [
                AuditCase(
                    name="mode_alias_fast",
                    model=f"{args.model}:fast",
                    messages=[{"role": "user", "content": "Answer in one sentence: what is a cache?"}],
                    max_tokens=64,
                    expect_mode="fast",
                    expect_primary="qwen",
                    min_model_calls=1,
                    max_model_calls=1,
                    max_elapsed_s=15.0,
                    markers=("cache",),
                ),
                AuditCase(
                    name="mode_alias_deep_compact",
                    model=f"{args.model}:deep",
                    messages=[
                        {
                            "role": "user",
                            "content": "Review this function for one bug and give a tiny fixed version: def avg(xs): return sum(xs)/len(xs)",
                        }
                    ],
                    max_tokens=64,
                    expect_mode="deep",
                    expect_workflow="qwen_compact_deep_direct",
                    expect_primary="qwen",
                    min_model_calls=1,
                    max_model_calls=1,
                    max_elapsed_s=20.0,
                    markers=("bug", "def avg"),
                    forbidden_markers=("Explanation:",),
                ),
                AuditCase(
                    name="mode_alias_ultra_compact",
                    model=f"{args.model}:ultra",
                    messages=[
                        {
                            "role": "user",
                            "content": "Give two risks of deploying untested code and one mitigation. Keep it short.",
                        }
                    ],
                    max_tokens=80,
                    expect_mode="ultra",
                    expect_workflow="qwen_compact_ultra_direct",
                    expect_primary="qwen",
                    min_model_calls=1,
                    max_model_calls=1,
                    max_elapsed_s=20.0,
                    markers=("risk", "mitigation"),
                    forbidden_markers=("Explanation:",),
                ),
            ]
        )

    if args.include_heavy:
        cases.extend(
            [
                AuditCase(
                    name="heavy_deep_full_workflow",
                    model=f"{args.model}:deep",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Review this Python function for one correctness bug. "
                                "Requirement: when b == 0, the function must return None instead of raising. "
                                "Use the labels BUG and FIX in the final answer. "
                                "Function: def divide(a, b): return a / b"
                            ),
                        }
                    ],
                    # Must be >256 to avoid the compact deep shortcut.
                    max_tokens=257,
                    expect_mode="deep",
                    expect_workflow="gemma_spec__qwen_solution__glm_verify__qwen_repair",
                    expect_primary="qwen",
                    min_model_calls=3,
                    max_model_calls=4,
                    max_elapsed_s=300.0,
                    markers=("BUG", "FIX", "if b == 0", "None"),
                    forbidden_markers=("internal routing",),
                ),
                AuditCase(
                    name="heavy_ultra_full_workflow",
                    model=f"{args.model}:ultra",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Evaluate this deployment plan for a small web service: "
                                "merge directly to production without tests or rollback. "
                                "Use the labels RISKS and MITIGATION in the final answer."
                            ),
                        }
                    ],
                    # Must be >128 to avoid the compact ultra shortcut; 192 is
                    # enough to retain both required labels in the final answer.
                    max_tokens=192,
                    expect_mode="ultra",
                    expect_workflow="gemma_spec__qwen_solution__glm_critique__gemma_synthesis__glm_verify",
                    expect_primary="qwen",
                    min_model_calls=5,
                    max_model_calls=6,
                    max_elapsed_s=540.0,
                    markers=("RISKS", "MITIGATION"),
                    forbidden_markers=("internal routing",),
                ),
            ]
        )

    if args.include_long:
        filler = " alpha beta gamma delta epsilon zeta eta theta iota kappa" * 220
        cases.append(
            AuditCase(
                name="long_context_head_tail",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Return exactly `HEAD_OK TAIL_OK` if you can see both marker tokens. "
                            "The head marker is HEAD_OK.\n\n"
                            f"{filler}\n\n"
                            "The tail marker is TAIL_OK."
                        ),
                    }
                ],
                max_tokens=32,
                expect_mode="fast",
                expect_workflow="deterministic_exact_literal",
                expect_model_calls=0,
                max_elapsed_s=1.0,
                markers=("HEAD_OK", "TAIL_OK"),
                forbidden_markers=("```",),
            )
        )
        filler_before = " alpha beta gamma delta epsilon zeta eta theta iota kappa" * 450
        filler_after = " lambda mu nu xi omicron pi rho sigma tau upsilon" * 450
        cases.append(
            AuditCase(
                name="long_context_retrieval_pack",
                model=f"{args.model}:fast",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You are given a long operations note. Answer the question at the end.\n\n"
                            f"{filler_before}\n\n"
                            "CENTRAL_SECRET_CODE = ORCHID-4827-MIDPOINT\n"
                            "Use this exact code when asked for CENTRAL_SECRET_CODE.\n\n"
                            f"{filler_after}\n\n"
                            "Question: What is the value of CENTRAL_SECRET_CODE? Return only the value."
                        ),
                    }
                ],
                max_tokens=32,
                expect_mode="fast",
                expect_primary="qwen",
                min_model_calls=1,
                max_model_calls=1,
                max_elapsed_s=40.0,
                markers=("ORCHID-4827-MIDPOINT",),
                forbidden_markers=("```",),
            )
        )

    rows = [
        {
            "name": "runtime_health",
            "ok": not health_checks,
            "checks": health_checks,
            "status": health.get("status"),
            "front_model": health.get("front_model"),
            "backend": health.get("backend"),
            "router_06b": router_06b,
            "models": model_ids,
        }
    ]
    rows.append(capability_row(capabilities, args.model, capability_error))
    rows.extend(run_case(case, chat_url, args.model, args.timeout) for case in cases)
    if args.include_api:
        rows.extend(run_api_checks(chat_url, args.model, args.timeout))

    first_messages = [
        {"role": "user", "content": "Write a Python expression for the sum of numbers from 1 to 10. Return only the expression."}
    ]
    first = post_json(
        chat_url,
        {"model": args.model, "messages": first_messages, "max_tokens": 64, "temperature": 0.0},
        timeout=args.timeout,
    )
    first_answer = answer_of(first)
    followup_messages = [
        first_messages[0],
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": "もっと短く。"},
    ]
    followup = post_json(
        chat_url,
        {"model": args.model, "messages": followup_messages, "max_tokens": 64, "temperature": 0.0},
        timeout=args.timeout,
    )
    first_route = route_of(first)
    followup_route = route_of(followup)
    followup_ok = first_route.get("primary") == followup_route.get("primary")
    followup_answer = answer_of(followup)
    if "```" in followup_answer:
        followup_ok = False
    rows.append(
        {
            "name": "short_followup_affinity",
            "ok": followup_ok,
            "checks": []
            if followup_ok
            else [
                check
                for check in [
                    (
                        f"primary changed {first_route.get('primary')} -> {followup_route.get('primary')}"
                        if first_route.get("primary") != followup_route.get("primary")
                        else ""
                    ),
                    "forbidden marker '```'" if "```" in followup_answer else "",
                ]
                if check
            ],
            "elapsed_s_external": None,
            "workflow": followup_route.get("workflow"),
            "primary": followup_route.get("primary"),
            "router": followup_route.get("router"),
            "reason": followup_route.get("reason"),
            "model_call_count": perf_of(followup).get("model_call_count"),
            "actual_completion_tps": perf_of(followup).get("actual_completion_tps"),
            "theoretical_completion_tps": perf_of(followup).get("theoretical_completion_tps"),
            "efficiency_vs_backend_theory": perf_of(followup).get("efficiency_vs_backend_theory"),
            "answer_preview": followup_answer[:240],
            "first_primary": first_route.get("primary"),
            "first_workflow": first_route.get("workflow"),
        }
    )

    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target": {"base_url": base, "model": args.model},
        "health": health,
        "capabilities": capabilities,
        "models": model_ids,
        "runs": len(rows),
        "ok_runs": sum(1 for row in rows if row["ok"]),
        "failed_runs": sum(1 for row in rows if not row["ok"]),
        "rows": rows,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if summary["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
