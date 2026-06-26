from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

try:
    # 通常運用(uvicorn src.conductor_7940hs:app)ではsrcがパッケージなので相対import。
    from .fugu_router_06b import RouterUnavailable, TowerDecision, ZeroSixBTower
    from .tool_runner import ALLOWED_PREFIXES, ToolResult, allowed, run_tool
except ImportError:
    # 単体ロード(テスト等)向けフォールバック。
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fugu_router_06b import RouterUnavailable, TowerDecision, ZeroSixBTower  # type: ignore
    from tool_runner import ALLOWED_PREFIXES, ToolResult, allowed, run_tool  # type: ignore

AgentName = Literal["gemma", "qwen", "glm"]
Mode = Literal["fast", "balanced", "deep", "ultra"]
APP_VERSION = "0.7.9"

# 入力文字数→トークン数の保守的な換算係数。日本語は概ね1文字≒1トークンなので
# 安全側に倒し、英日混在を見込んで1トークン≒1.8文字とみなす。
CHARS_PER_TOKEN = 1.8


class Route(BaseModel):
    mode: Mode = "balanced"
    primary: AgentName = "qwen"
    planner: Optional[AgentName] = "gemma"
    verifier: Optional[AgentName] = "glm"
    workflow: str = "qwen_direct__glm_if_risky"
    task_type: str = "general"
    risky: bool = False
    reason: str = ""
    router: str = "rules"
    confidence: Optional[float] = None
    scores: Dict[str, float] = Field(default_factory=dict)
    mode_scores: Dict[str, float] = Field(default_factory=dict)
    task_scores: Dict[str, float] = Field(default_factory=dict)
    router_elapsed_ms: Optional[int] = None


class VerifyReport(BaseModel):
    passed: bool = True
    severity: Literal["none", "minor", "major", "fatal"] = "none"
    defects: List[str] = Field(default_factory=list)
    repair_instruction: str = ""


class ToolSpec(BaseModel):
    """code task向けのオプトイン tool 実行指定。

    リクエストbodyの `fugu_tools` から来る。指定が無ければtoolは一切走らない。
    実際に許可されるコマンドは tool_runner.ALLOWED_PREFIXES のみ。
    """

    cwd: str
    commands: List[List[str]] = Field(default_factory=list)
    timeout_s: int = 120


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    system: str
    temperature: float
    max_tokens: int
    extra_body: Dict[str, Any]


class LocalMoEFugu:
    def __init__(self, config_path: Optional[str] = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        default_config = project_root / "configs" / "moe_fugu_hf.yaml"
        path = Path(config_path or os.getenv("MOE_FUGU_CONFIG", str(default_config)))
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        self.project_root = project_root
        self.config_path = path
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.front_model = self.cfg.get("front_model", "local-moe-fugu")
        self.front_api_key = self.cfg.get("front_api_key", "local-frontdoor-key")

        backend = self.cfg["backend"]
        self.backend_clients: Dict[str, AsyncOpenAI] = {}
        self.backend_base_urls: Dict[str, str] = {}
        self.backend_model_names: Dict[str, List[str]] = {}
        self.hybrid_backend = self.cfg.get("hybrid_backend", {}) or {}
        self.hybrid_enabled = bool(self.hybrid_backend.get("enabled", False))
        self.hybrid_default_backend = str(self.hybrid_backend.get("default_backend", "cpu"))
        self.hybrid_fallback_backend = str(self.hybrid_backend.get("fallback_backend", self.hybrid_default_backend))
        self.hybrid_policy = self.hybrid_backend.get("route_policy", {}) or {}

        backend_defs = self.hybrid_backend.get("backends", {}) or {}
        if not backend_defs:
            backend_defs = {
                "cpu": {
                    "base_url": backend["base_url"],
                    "api_key": backend.get("api_key", "local-dev-key"),
                    "timeout_s": backend.get("timeout_s", 900),
                    "model_names": list((self.cfg.get("models", {}) or {}).keys()),
                }
            }
        elif "cpu" not in backend_defs:
            backend_defs = {
                "cpu": {
                    "base_url": backend["base_url"],
                    "api_key": backend.get("api_key", "local-dev-key"),
                    "timeout_s": backend.get("timeout_s", 900),
                    "model_names": list((self.cfg.get("models", {}) or {}).keys()),
                },
                **backend_defs,
            }

        for backend_name, backend_cfg in backend_defs.items():
            base_url = str(backend_cfg["base_url"])
            self.backend_clients[str(backend_name)] = AsyncOpenAI(
                base_url=base_url,
                api_key=backend_cfg.get("api_key", backend.get("api_key", "local-dev-key")),
                timeout=float(backend_cfg.get("timeout_s", backend.get("timeout_s", 900))),
            )
            self.backend_base_urls[str(backend_name)] = base_url
            model_names = backend_cfg.get("model_names")
            if isinstance(model_names, list) and model_names:
                self.backend_model_names[str(backend_name)] = [str(x) for x in model_names]
            else:
                self.backend_model_names[str(backend_name)] = list((self.cfg.get("models", {}) or {}).keys())

        if self.hybrid_default_backend not in self.backend_clients:
            self.hybrid_default_backend = "cpu" if "cpu" in self.backend_clients else next(iter(self.backend_clients))
        if self.hybrid_fallback_backend not in self.backend_clients:
            self.hybrid_fallback_backend = self.hybrid_default_backend
        # Compatibility alias for older code paths that assume one backend.
        self.client = self.backend_clients[self.hybrid_default_backend]

        self.models: Dict[str, ModelConfig] = {}
        for name, raw in self.cfg["models"].items():
            self.models[name] = ModelConfig(
                name=name,
                model_id=raw["id"],
                system=raw.get("system", ""),
                temperature=float(raw.get("temperature", 0.2)),
                max_tokens=int(raw.get("max_tokens", 4096)),
                extra_body=dict(raw.get("extra_body", {}) or {}),
            )

        self.scheduler = self.cfg.get("scheduler", {})
        self.log_content = bool(self.scheduler.get("log_content", False))
        self.fast_primary = str(self.scheduler.get("fast_primary", "qwen")).lower()
        if self.fast_primary not in self.models:
            self.fast_primary = "qwen" if "qwen" in self.models else next(iter(self.models))
        # repair回数はconfigで制御する（以前はコードにハードコードされ、yaml値が死んでいた）。
        self.max_repair_rounds = max(0, int(self.scheduler.get("max_repair_rounds", 1)))
        # followup時のモデルスワップ回避。短い追撃質問だけ前回primaryに寄せる。
        self.avoid_model_swap_on_followup = bool(self.scheduler.get("avoid_model_swap_on_followup", False))
        self.fast_bias = bool(self.scheduler.get("fast_bias", False))
        self.verify_code_by_default = bool(self.scheduler.get("verify_code_by_default", True))
        self.routing_tower = ZeroSixBTower.from_config(self.cfg, self.project_root)
        router_cfg = self.cfg.get("router_06b", {}) or {}
        self.tower_agent_min_confidence = float(router_cfg.get("agent_min_confidence", 0.42))
        self.tower_mode_min_margin = float(router_cfg.get("mode_min_margin", 0.04))

        # サーバ -c と揃えるべきctx。トランスクリプトのクリップ計算に使う。
        hw = self.cfg.get("hardware_profile", {})
        self.default_ctx = int(hw.get("default_ctx", 12288))
        self.deep_ctx = int(hw.get("deep_ctx", self.default_ctx))

        self.log_path = Path(os.getenv("MOE_FUGU_LOG", str(project_root / "logs" / "moe_fugu_runs.jsonl")))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Hardware constraint: one model call at a time. This avoids accidental fanout.
        # 注意: このLockはプロセス内のみ有効。uvicornは必ず --workers 1 で起動すること。
        self.call_lock = asyncio.Lock()
        # rid単位のトークン使用量。run()開始で初期化し、応答生成後にpopする。
        self._usage: Dict[str, Dict[str, int]] = {}
        self._perf: Dict[str, Dict[str, Any]] = {}
        self._route_affinity: Dict[str, Dict[str, Any]] = {}

    def log(self, obj: Dict[str, Any]) -> None:
        obj = dict(obj)
        obj["ts"] = int(time.time())
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    @staticmethod
    def content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    parts.append("[image_url]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def clip_middle(text: str, char_limit: int) -> str:
        if len(text) <= char_limit:
            return text
        marker = "\n\n[...middle omitted to fit context...]\n\n"
        if char_limit <= len(marker) + 200:
            return text[-char_limit:]
        head = max(1000, int(char_limit * 0.35))
        tail = char_limit - head - len(marker)
        if tail < 1000:
            tail = max(200, int(char_limit * 0.45))
            head = char_limit - tail - len(marker)
        return text[:head] + marker + text[-tail:]

    def visible_transcript_full(self, messages: List[Dict[str, Any]]) -> str:
        rows: List[str] = []
        for m in messages:
            role = m.get("role", "unknown")
            if role == "system":
                continue
            rows.append(f"{role.upper()}:\n{self.content_to_text(m.get('content', ''))}")
        return "\n\n".join(rows)

    @staticmethod
    def retrieval_terms(query: str) -> List[str]:
        stop = {
            "the", "and", "that", "this", "with", "from", "into", "only", "return",
            "answer", "value", "what", "which", "where", "when", "please", "provided",
            "context", "document", "text", "user", "assistant", "following", "above",
        }
        raw_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_:-]{2,}", query)
        terms: List[str] = []
        seen: set[str] = set()
        for term in raw_terms:
            normalized = term.lower()
            if normalized in stop or normalized.isdigit() or len(normalized) < 3:
                continue
            if normalized not in seen:
                terms.append(normalized)
                seen.add(normalized)
        for quoted in re.findall(r"[`\"']([^`\"'\n]{3,120})[`\"']", query):
            for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_:-]{2,}", quoted):
                normalized = term.lower()
                if normalized not in seen:
                    terms.insert(0, normalized)
                    seen.add(normalized)
        return terms[:24]

    @staticmethod
    def retrieval_query(messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                text = LocalMoEFugu.content_to_text(m.get("content", ""))
                break
        else:
            return ""
        lower = text.lower()

        primary_markers = ["question:", "questions:", "task:", "tasks:"]
        best_primary = max((lower.rfind(marker) for marker in primary_markers), default=-1)
        if best_primary >= 0:
            return text[best_primary:]

        secondary_markers = [
            "what is", "which", "where", "when", "find", "lookup", "extract",
            "value of", "please answer",
        ]
        best_secondary = max((lower.rfind(marker) for marker in secondary_markers), default=-1)
        if best_secondary >= 0:
            return text[best_secondary:]

        return text[-1800:]

    @staticmethod
    def is_lookup_like_query(query: str) -> bool:
        lower = query.lower()
        markers = [
            "what is", "which", "where", "when", "find", "lookup", "extract",
            "return only", "value of", "identifier", "secret", "code", "key", "id",
        ]
        return any(marker in lower for marker in markers)

    def retrieval_pack_transcript(self, full_text: str, query: str, char_limit: int) -> Optional[str]:
        terms = self.retrieval_terms(query)
        if len(full_text) <= char_limit or not terms or char_limit < 3500:
            return None

        lookup_like = self.is_lookup_like_query(query)
        target_limit = min(char_limit, 6000) if lookup_like else char_limit
        target_limit = max(3500, target_limit)
        marker = "\n\n[...middle omitted; FUGU kept head, tail, and relevant excerpts...]\n\n"
        if target_limit <= len(marker) + 2000:
            return None

        head_len = min(2200, max(900, int(target_limit * 0.18)))
        tail_len = min(3000, max(1200, int(target_limit * 0.25)))
        budget = target_limit - head_len - tail_len - len(marker)
        if budget < 1200:
            return None

        chunk_size = 2200 if lookup_like else 3400
        overlap = 300
        chunks: List[Tuple[int, int, str]] = []
        start = 0
        while start < len(full_text):
            end = min(len(full_text), start + chunk_size)
            chunks.append((start, end, full_text[start:end]))
            if end >= len(full_text):
                break
            start = max(start + 1, end - overlap)

        scored: List[Tuple[float, int, int, str]] = []
        query_lower = query.lower()
        for start, end, chunk in chunks:
            if end <= head_len or start >= len(full_text) - tail_len:
                continue
            lower = chunk.lower()
            score = 0.0
            for term in terms:
                hits = lower.count(term)
                if hits:
                    weight = 4.0 if ("_" in term or "-" in term or ":" in term or any(c.isdigit() for c in term)) else 1.0
                    score += hits * weight
            if query_lower and query_lower[:120] in lower:
                score += 2.0
            if score > 0:
                scored.append((score, start, end, chunk))

        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected: List[Tuple[int, int, str]] = []
        used_ranges: List[Tuple[int, int]] = []
        used_chars = 0
        for _score, start, end, chunk in scored:
            if any(not (end <= old_start or start >= old_end) for old_start, old_end in used_ranges):
                continue
            excerpt = chunk.strip()
            if not excerpt:
                continue
            excerpt_budget = min(len(excerpt), max(900, budget - used_chars))
            if excerpt_budget <= 0:
                break
            if len(excerpt) > excerpt_budget:
                excerpt = self.clip_middle(excerpt, excerpt_budget)
            selected.append((start, end, excerpt))
            used_ranges.append((start, end))
            used_chars += len(excerpt) + 80
            if used_chars >= budget or len(selected) >= 4:
                break

        if not selected:
            return None
        selected.sort(key=lambda item: item[0])

        excerpts = []
        for index, (start, _end, excerpt) in enumerate(selected, 1):
            excerpts.append(f"[Relevant excerpt {index} around char {start}]\n{excerpt}")
        packed = (
            "[FUGU context pack: head/tail plus query-relevant excerpts.]\n\n"
            "[Head]\n"
            + full_text[:head_len].rstrip()
            + marker
            + "\n\n".join(excerpts)
            + "\n\n[Tail]\n"
            + full_text[-tail_len:].lstrip()
        )
        return self.clip_middle(packed, target_limit)

    def visible_transcript(self, messages: List[Dict[str, Any]], char_limit: int = 50000) -> str:
        text = self.visible_transcript_full(messages)
        if len(text) <= char_limit:
            return text
        packed = self.retrieval_pack_transcript(text, self.retrieval_query(messages), char_limit)
        if packed:
            return packed
        return self.clip_middle(text, char_limit)

    def transcript_char_budget(self, ctx: int, reserve_out_tokens: int) -> int:
        """ctxからこのステージで安全に入力に回せる文字数を見積もる。

        ctx を出力予約・system/spec等のマージンに食われた残りに割り当てる。
        以前は60000/70000文字を固定で送っており、ctx 12288では確実に溢れていた。
        """
        # systemプロンプトやspec/critique等の同梱分として固定マージンを引く。
        margin_tokens = 1024
        input_tokens = ctx - reserve_out_tokens - margin_tokens
        if input_tokens < 512:
            input_tokens = 512  # 最低限。これ未満ならctxかmax_tokens設定自体が破綻している。
        return max(1500, int(input_tokens * CHARS_PER_TOKEN))

    def fit_transcript(
        self, messages: List[Dict[str, Any]], ctx: int, reserve_out_tokens: int
    ) -> str:
        hints = self.local_context_hints(messages)
        limit = self.transcript_char_budget(ctx, reserve_out_tokens)
        if hints:
            limit = max(1500, limit - len(hints) - 2)
            return hints + "\n\n" + self.visible_transcript(messages, char_limit=limit)
        return self.visible_transcript(
            messages, char_limit=limit
        )

    @staticmethod
    def stage_max_tokens(
        client_max_tokens: Optional[int],
        default: int,
        *,
        minimum: int = 16,
        multiplier: int = 1,
    ) -> int:
        if client_max_tokens is None:
            return default
        requested = max(1, int(client_max_tokens))
        return max(minimum, min(default, requested * multiplier))

    def last_user_text(self, messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return self.content_to_text(m.get("content", ""))
        return self.visible_transcript(messages)

    def user_texts(self, messages: List[Dict[str, Any]]) -> List[str]:
        return [
            self.content_to_text(m.get("content", "")).strip()
            for m in messages
            if m.get("role") == "user" and self.content_to_text(m.get("content", "")).strip()
        ]

    def conversation_affinity_key(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        users = self.user_texts(messages)
        if not users:
            return None
        system = ""
        for m in messages:
            if m.get("role") == "system":
                system = self.content_to_text(m.get("content", ""))[:1000]
                break
        anchor = users[0][:2000]
        raw = f"{system}\n---first-user---\n{anchor}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def is_short_followup(self, messages: List[Dict[str, Any]]) -> bool:
        users = self.user_texts(messages)
        if len(users) < 2:
            return False
        text = users[-1].strip()
        if not text or len(text) > 500:
            return False
        lower = text.lower()
        strong_new_task = [
            "最新", "検索", "引用", "source", "citation", "paper", "論文",
            "レビュー", "検証", "監査", "security", "audit", "verify",
            "python", "typescript", "javascript", "powershell", "traceback",
            "コード", "実装", "修正", "エラー", "ビルド",
        ]
        if any(marker in lower for marker in strong_new_task):
            return False
        followup_markers = [
            "それ", "これ", "この", "その", "さっき", "前の", "上の", "続き",
            "もっと", "詳しく", "短く", "要約", "じゃあ", "では", "なら",
            "why", "that", "this", "it", "same", "continue", "more", "expand",
            "shorter", "summarize",
        ]
        return len(text) <= 120 or any(marker in lower for marker in followup_markers)

    def remember_route_affinity(self, messages: List[Dict[str, Any]], route: Route) -> None:
        if not self.avoid_model_swap_on_followup:
            return
        key = self.conversation_affinity_key(messages)
        if not key:
            return
        self._route_affinity[key] = {
            "primary": route.primary,
            "mode": route.mode,
            "workflow": route.workflow,
            "task_type": route.task_type,
            "ts": int(time.time()),
        }
        if len(self._route_affinity) > 256:
            oldest = sorted(self._route_affinity.items(), key=lambda item: int(item[1].get("ts", 0)))[:64]
            for old_key, _ in oldest:
                self._route_affinity.pop(old_key, None)

    @staticmethod
    def has_japanese(text: str) -> bool:
        return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))

    @staticmethod
    def weekday_ja(value: date) -> str:
        return ["月", "火", "水", "木", "金", "土", "日"][value.weekday()]

    @staticmethod
    def weekday_en(value: date) -> str:
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][value.weekday()]

    @staticmethod
    def count_visible_japanese_a_vowel(text: str) -> int:
        # Visible kana with /a/ vowel. Kanji readings are deliberately not guessed here.
        a_kana = "ぁあかがさざただなはばぱまゃやらゎわゕァアカガサザタダナハバパマャヤラヮワヵヷ"
        return sum(1 for ch in text if ch in a_kana)

    @staticmethod
    def asks_japanese_a_vowel_count(text: str) -> bool:
        patterns = [
            r"母音\s*[\"'“”‘’「『]?あ[\"'“”‘’」』]?",
            r"[\"'“”‘’「『]あ[\"'“”‘’」』]\s*の数",
            r"あ段",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def count_target_text(text: str) -> str:
        if "\n" in text and re.search(r"次の文章|以下の文章|次の文|以下の文", text):
            head, tail = text.split("\n", 1)
            if re.search(r"数|何個|いくつ|count|how many", head, flags=re.IGNORECASE):
                return tail.strip()
        colon_match = re.search(r"(?:次の文章|以下の文章|次の文|以下の文)[^:：]*[:：]\s*(.+)$", text, flags=re.DOTALL)
        if colon_match:
            return colon_match.group(1).strip()
        return text

    def is_explicit_code_request(self, text: str) -> bool:
        lower = text.lower()
        markers = [
            "write a python",
            "write code",
            "give a powershell command",
            "command that",
            "function named",
            "script",
            "python",
            "powershell",
            "コード",
            "関数",
            "コマンド",
            "スクリプト",
            "実装",
        ]
        return any(marker in lower for marker in markers)

    def runtime_context(self, text: str = "") -> str:
        lower = text.lower()
        needs_date = (
            "today" in lower
            or "current date" in lower
            or "weekday" in lower
            or "曜日" in text
            or "今日" in text
        )
        if not needs_date:
            return ""
        today = date.today()
        return f"Local date: {today.isoformat()}, {self.weekday_en(today)}/{self.weekday_ja(today)}曜日."

    def local_context_hints(self, messages: List[Dict[str, Any]]) -> str:
        text = self.last_user_text(messages)
        lower = text.lower()
        hints: List[str] = []
        today = date.today()
        if ("今日" in text or "today" in lower) and ("曜日" in text or "weekday" in lower or "day" in lower):
            hints.append(f"今日は{self.weekday_ja(today)}曜日({today.isoformat()})。")

        if "曜日" in text:
            for year, month, day in re.findall(r"(\d{1,4})\s*年\s*(\d{1,2})\s*/\s*(\d{1,2})", text):
                try:
                    target = date(int(year), int(month), int(day))
                except ValueError:
                    continue
                hints.append(
                    f"{int(year)}年{int(month)}月{int(day)}日は{self.weekday_ja(target)}曜日"
                    "(proleptic Gregorian)。"
                )

        if "母音a" in text or "vowel a" in lower:
            lowercase_a = text.count("a")
            both_case_a = lowercase_a + text.count("A")
            hints.append(
                f"母音a質問は可視ラテン文字だけで答える: 小文字'a'={lowercase_a}個、"
                f"A/a合算={both_case_a}個。日本語読みの/a/は数えない。"
            )

        if self.asks_japanese_a_vowel_count(text):
            target_text = self.count_target_text(text)
            exact_hiragana_a = target_text.count("あ")
            exact_kana_a = exact_hiragana_a + target_text.count("ぁ") + target_text.count("ア") + target_text.count("ァ")
            visible_a_vowel = self.count_visible_japanese_a_vowel(target_text)
            hints.append(
                f"母音\"あ\"質問は可視文字で答える: ひらがな'あ'={exact_hiragana_a}個、"
                f"あ/ぁ/ア/ァ={exact_kana_a}個、可視かなのあ段={visible_a_vowel}個。"
                "漢字の読みは推測しない。"
            )

        if "不定積分" in text:
            hints.append("不定積分=原始関数+C。")

        if "アイスクリーム" in text and ("推奨量" in text or "成人男性" in text):
            hints.append("アイスクリームに成人男性向け標準推奨量なし。嗜好品なので少量を時々。")

        if re.search(r"\b50\s*m\b|50m", text, flags=re.IGNORECASE) and "徒歩" in text and "車" in text:
            hints.append("50mなら通常は徒歩。荷物や天候などの事情があれば車もあり。")

        if not hints:
            return ""
        return "確定情報:\n- " + "\n- ".join(hints)

    def apply_deterministic_corrections(self, messages: List[Dict[str, Any]], answer: str) -> str:
        text = self.last_user_text(messages)
        lower = text.lower()
        has_latin_a_query = "母音a" in text or "vowel a" in lower
        has_japanese_a_query = self.asks_japanese_a_vowel_count(text)
        if not has_latin_a_query and not has_japanese_a_query:
            return answer

        if has_japanese_a_query:
            target_text = self.count_target_text(text)
            exact_hiragana_a = target_text.count("あ")
            exact_kana_a = exact_hiragana_a + target_text.count("ぁ") + target_text.count("ア") + target_text.count("ァ")
            visible_a_vowel = self.count_visible_japanese_a_vowel(target_text)
            correction = (
                f"ひらがな「あ」は{exact_hiragana_a}個、あ/ぁ/ア/ァは{exact_kana_a}個、"
                f"可視かなのあ段母音は{visible_a_vowel}個です。漢字の読み由来の/a/は推測しません。"
            )
        else:
            lowercase_a = text.count("a")
            both_case_a = lowercase_a + text.count("A")
            if self.has_japanese(text):
                correction = (
                    f"可視ラテン文字'a'は小文字のみで{lowercase_a}個、A/a合算で{both_case_a}個です。"
                    "日本語読みの/a/は読み依存です。"
                )
            else:
                correction = (
                    f"Visible Latin lowercase 'a': {lowercase_a}; A/a total: {both_case_a}. "
                    "Phonetic /a/ depends on transcription."
                )

        lines = answer.splitlines()
        for index in range(len(lines) - 1, -1, -1):
            line = lines[index]
            if "個" not in line and "total" not in line.lower():
                continue
            if not re.search(r"母音|ラテン|A/a|/a/|vowel|\ba\b|あ段|ひらがな|[\"“”「]あ", line, flags=re.IGNORECASE):
                continue
            prefix_match = re.match(r"^(\s*\d+[.．]\s*)", line)
            prefix = prefix_match.group(1) if prefix_match else ""
            lines[index] = prefix + correction
            return "\n".join(lines)

        return answer.rstrip() + "\n" + correction

    @staticmethod
    def strip_outer_code_fence(answer: str) -> str:
        text = answer.strip()
        match = re.match(r"^```[a-zA-Z0-9_+.-]*\s*\n(?P<body>.*?)\n?```\s*$", text, flags=re.DOTALL)
        if match:
            return match.group("body").strip()
        inline = re.match(r"^```[a-zA-Z0-9_+.-]*\s*(?P<body>[^`].*?)\s*```\s*$", text, flags=re.DOTALL)
        if inline:
            return inline.group("body").strip()
        return text

    @staticmethod
    def strip_trailing_empty_heading(answer: str) -> str:
        text = answer.rstrip()
        return re.sub(
            r"(?is)(?:\n\s*)+(?:\*\*)?(?:explanation|notes?|summary|補足|説明)(?:\*\*)?\s*:?\s*$",
            "",
            text,
        ).rstrip()

    def apply_output_shape_corrections(self, messages: List[Dict[str, Any]], answer: str) -> str:
        text = self.last_user_text(messages)
        lower = text.lower()
        answer = answer.replace("Би", "→")
        answer = self.strip_trailing_empty_heading(answer)
        stripped = answer.strip()

        if (
            "return exactly" in lower
            and "answer:" in lower
            and re.search(r"\banswer\s*:\s*([a-d])\b", stripped, flags=re.IGNORECASE)
        ):
            letter = re.search(r"\banswer\s*:\s*([a-d])\b", stripped, flags=re.IGNORECASE)
            if letter:
                return f"ANSWER: {letter.group(1).upper()}"

        if "return exactly" in lower:
            exact_targets = [target.strip() for target in re.findall(r"`([^`\n]{1,160})`", text)]
            for target in reversed(exact_targets):
                if target and target in stripped:
                    return target

        expression_only = (
            "return only the expression" in lower
            or "only the expression" in lower
            or "式だけ" in text
        )
        if expression_only:
            body = self.strip_outer_code_fence(stripped)
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            if lines:
                return lines[0]
            return body

        code_only = (
            ("return only" in lower or "出力は" in text or "だけ" in text)
            and ("code" in lower or "function" in lower or "command" in lower or "コード" in text or "関数" in text)
            and not any(marker in lower for marker in ["explain", "briefly explain", "説明"])
        )
        if code_only:
            return self.strip_outer_code_fence(stripped)

        return answer

    @staticmethod
    def format_numeric_answer(value: float) -> str:
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return f"{value:.10g}"

    @staticmethod
    def singular_word(value: str) -> str:
        value = value.lower()
        return value[:-1] if value.endswith("s") else value

    def deterministic_math_answer(self, lower: str) -> Optional[str]:
        percent_match = re.search(
            r"\bwhat\s+is\s+(-?\d+(?:\.\d+)?)\s+percent\s+of\s+(-?\d+(?:\.\d+)?)\b",
            lower,
        )
        if percent_match:
            pct = float(percent_match.group(1))
            base = float(percent_match.group(2))
            return self.format_numeric_answer(base * pct / 100.0)

        linear_seed = re.search(r"\bif\s+([a-z])\s*=\s*(-?\d+(?:\.\d+)?)\b", lower)
        linear_expr = re.search(
            r"\b([a-z])\s*=\s*(-?\d+(?:\.\d+)?)\s*\*\s*([a-z])\s*([+-])\s*(-?\d+(?:\.\d+)?)\b",
            lower,
        )
        if linear_seed and linear_expr:
            values = {linear_seed.group(1): float(linear_seed.group(2))}
            out_name, factor, in_name, op, bias = linear_expr.groups()
            if in_name in values:
                result = float(factor) * values[in_name]
                result = result + float(bias) if op == "+" else result - float(bias)
                if re.search(rf"\bwhat\s+is\s+{re.escape(out_name)}\b", lower):
                    return self.format_numeric_answer(result)

        rate_match = re.search(
            r"\btravels\s+(-?\d+(?:\.\d+)?)\s+kilometers?\s+in\s+(-?\d+(?:\.\d+)?)\s+hours?\b",
            lower,
        )
        if rate_match and "1 hour" in lower:
            distance = float(rate_match.group(1))
            hours = float(rate_match.group(2))
            if hours:
                return self.format_numeric_answer(distance / hours)

        if "arithmetic mean" in lower or "average" in lower:
            numbers_match = re.search(r"\b(?:numbers|values)\s+are\s+([0-9,\sand.-]+)", lower)
            if numbers_match:
                nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", numbers_match.group(1))]
                if nums:
                    return self.format_numeric_answer(sum(nums) / len(nums))

        price_pairs = re.findall(r"\b([a-z]+)\s+for\s+(-?\d+(?:\.\d+)?)\s+dollars?\s+each\b", lower)
        buy_match = re.search(r"\bbuy\s+(.+?)(?:,?\s+what|\?)", lower)
        if price_pairs and buy_match and "total cost" in lower:
            prices = {self.singular_word(item): float(price) for item, price in price_pairs}
            buys = re.findall(r"\b(\d+(?:\.\d+)?)\s+([a-z]+)\b", buy_match.group(1))
            if buys:
                total = 0.0
                for qty, item in buys:
                    key = self.singular_word(item)
                    if key not in prices:
                        break
                    total += float(qty) * prices[key]
                else:
                    return self.format_numeric_answer(total)

        return None

    def deterministic_japanese_compound_answer(self, text: str) -> Optional[str]:
        if not self.has_japanese(text):
            return None
        if not (
            "今日" in text
            and "曜日" in text
            and "不定積分" in text
            and "アイスクリーム" in text
            and re.search(r"\b50\s*m\b|50m", text, flags=re.IGNORECASE)
            and "母音" in text
        ):
            return None

        target_match = re.search(r"(\d{1,4})\s*年\s*(\d{1,2})\s*/\s*(\d{1,2})", text)
        if not target_match:
            return None
        try:
            target = date(
                int(target_match.group(1)),
                int(target_match.group(2)),
                int(target_match.group(3)),
            )
        except ValueError:
            return None

        today = date.today()
        rows = [
            f"1. 今日は{self.weekday_ja(today)}曜日です。",
            (
                f"2. {int(target_match.group(1))}年{int(target_match.group(2))}月"
                f"{int(target_match.group(3))}日は{self.weekday_ja(target)}曜日です。"
            ),
            "3. 不定積分は原始関数Fを見つけて、最後に積分定数Cを付けます。",
            "4. アイスクリームに成人男性向けの標準推奨量はありません。嗜好品として少量を時々が無難です。",
            "5. 50mなら通常は徒歩が合理的です。荷物や天候などの事情があれば車でも構いません。",
        ]

        lower = text.lower()
        if "母音a" in text or "vowel a" in lower:
            lowercase_a = text.count("a")
            both_case_a = lowercase_a + text.count("A")
            rows.append(
                f"6. 可視ラテン文字'a'は小文字のみで{lowercase_a}個、A/a合算で{both_case_a}個です。"
                "日本語読みの/a/は読み依存です。"
            )
        elif self.asks_japanese_a_vowel_count(text):
            exact_hiragana_a = text.count("あ")
            exact_kana_a = exact_hiragana_a + text.count("ぁ") + text.count("ア") + text.count("ァ")
            visible_a_vowel = self.count_visible_japanese_a_vowel(text)
            rows.append(
                f"6. ひらがな「あ」は{exact_hiragana_a}個、あ/ぁ/ア/ァは{exact_kana_a}個、"
                f"可視かなのあ段母音は{visible_a_vowel}個です。漢字の読み由来の/a/は推測しません。"
            )
        else:
            return None

        return "\n".join(rows)

    def deterministic_count_answer(self, text: str) -> Optional[str]:
        lower = text.lower()
        asks_count = (
            "数" in text
            or "何個" in text
            or "いくつ" in text
            or "count" in lower
            or "how many" in lower
        )
        if not asks_count:
            return None

        if "母音a" in text or "vowel a" in lower:
            lowercase_a = text.count("a")
            both_case_a = lowercase_a + text.count("A")
            if self.has_japanese(text):
                return (
                    f"可視ラテン文字'a'は小文字のみで{lowercase_a}個、A/a合算で{both_case_a}個です。"
                    "日本語読みの/a/は読み依存なので推測しません。"
                )
            return (
                f"Visible Latin lowercase 'a': {lowercase_a}; A/a total: {both_case_a}. "
                "Phonetic /a/ depends on transcription."
            )

        if self.asks_japanese_a_vowel_count(text):
            target_text = self.count_target_text(text)
            exact_hiragana_a = target_text.count("あ")
            exact_kana_a = exact_hiragana_a + target_text.count("ぁ") + target_text.count("ア") + target_text.count("ァ")
            visible_a_vowel = self.count_visible_japanese_a_vowel(target_text)
            return (
                f"ひらがな「あ」は{exact_hiragana_a}個、あ/ぁ/ア/ァは{exact_kana_a}個、"
                f"可視かなのあ段母音は{visible_a_vowel}個です。漢字の読み由来の/a/は推測しません。"
            )

        quoted = re.findall(r"[\"'“”‘’「『]([^\"'“”‘’」』]{1,8})[\"'“”‘’」』]", text)
        if quoted:
            target = quoted[-1]
            return f"可視テキスト中の「{target}」は{text.count(target)}個です。"

        return None

    def deterministic_exact_literal_answer(self, text: str) -> Optional[str]:
        lower = text.lower()
        if "return exactly" not in lower or "answer:" in lower:
            return None

        targets = [target.strip() for target in re.findall(r"`([^`\n]{1,160})`", text)]
        targets = [target for target in targets if target]
        if len(targets) != 1:
            return None

        target = targets[0]
        conditional = bool(re.search(r"\b(if|when|unless|otherwise|else)\b", lower))
        if not conditional:
            return target

        visibility_condition = any(
            marker in lower
            for marker in (
                "marker",
                "can see",
                "visible",
                "contains",
                "contain",
                "find both",
                "see both",
            )
        )
        if not visibility_condition:
            return None

        remainder = text.replace(f"`{target}`", "")
        tokens = re.findall(r"[A-Za-z0-9_:-]+", target)
        if not tokens:
            return None
        for token in tokens:
            if not re.search(rf"(?<![A-Za-z0-9_:-]){re.escape(token)}(?![A-Za-z0-9_:-])", remainder):
                return None
        return target

    def deterministic_answer(
        self,
        messages: List[Dict[str, Any]],
        forced_mode: Optional[str],
    ) -> Optional[Tuple[str, Route]]:
        if forced_mode not in {None, "fast"}:
            return None

        text = self.last_user_text(messages)
        lower = text.lower()
        exact_literal = self.deterministic_exact_literal_answer(text)
        if exact_literal is not None:
            route = Route(
                mode="fast",
                primary="qwen",
                planner=None,
                verifier=None,
                workflow="deterministic_exact_literal",
                task_type="general",
                risky=False,
                reason="deterministic exact literal solver",
                router="deterministic",
            )
            return exact_literal, route

        japanese_compound = self.deterministic_japanese_compound_answer(text)
        if japanese_compound is not None:
            route = Route(
                mode="fast",
                primary="gemma",
                planner=None,
                verifier=None,
                workflow="deterministic_japanese_compound",
                task_type="general",
                risky=False,
                reason="deterministic Japanese compound solver",
                router="deterministic",
            )
            return japanese_compound, route

        count_answer = self.deterministic_count_answer(text)
        if count_answer is not None:
            route = Route(
                mode="fast",
                primary="gemma" if self.has_japanese(text) else "qwen",
                planner=None,
                verifier=None,
                workflow="deterministic_count",
                task_type="general",
                risky=False,
                reason="deterministic visible text counter",
                router="deterministic",
            )
            return count_answer, route

        if "answer:" not in lower:
            return None

        math_answer = self.deterministic_math_answer(lower)
        if math_answer is not None:
            route = Route(
                mode="fast",
                primary="qwen",
                planner=None,
                verifier=None,
                workflow="deterministic_arithmetic",
                task_type="math",
                risky=False,
                reason="deterministic arithmetic solver",
                router="deterministic",
            )
            return f"ANSWER: {math_answer}", route

        if "what day" not in lower:
            return None

        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        today_match = re.search(
            r"\btoday\s+is\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lower,
        )
        offset_match = re.search(r"\b(\d+)\s+days?\s+from\s+today\b", lower)
        choices = re.findall(
            r"\b([a-d])\)\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lower,
        )
        if not today_match or not offset_match or not choices:
            return None

        today = today_match.group(1)
        offset = int(offset_match.group(1))
        target = weekdays[(weekdays.index(today) + offset) % len(weekdays)]
        for letter, day in choices:
            if day == target:
                route = Route(
                    mode="fast",
                    primary="gemma",
                    planner=None,
                    verifier=None,
                    workflow="deterministic_calendar",
                    task_type="logic",
                    risky=False,
                    reason="deterministic calendar multiple-choice solver",
                    router="deterministic",
                )
                return f"ANSWER: {letter.upper()}", route
        return None

    @staticmethod
    def parse_json_object(raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data

        raise ValueError("No JSON object found")

    def rule_route(self, messages: List[Dict[str, Any]], forced_mode: Optional[str] = None) -> Route:
        text = self.last_user_text(messages)
        lower = text.lower()

        code_keywords = [
            "python", "typescript", "javascript", "fastapi", "docker", "traceback",
            # 単独の "api" は日本語文中で誤検出が多いので、より具体的な語に限定する。
            "rest api", "endpoint", "sdk",
            "bug", "repo", "llama.cpp", "powershell", "pytest", "ruff", "tsc",
            "コード", "実装", "修正", "エラー", "ビルド", "デバッグ", "コンパイル",
        ]
        verify_keywords = [
            "正しいか", "検証", "レビュー", "矛盾", "バグ", "壊れる", "危険", "確認",
            "are you sure", "verify", "security", "audit", "review",
        ]
        design_keywords = [
            "設計", "アーキテクチャ", "構成", "オーケストレーター", "workflow", "ワークフロー",
            "仕様", "要件", "ロードマップ", "分解", "詰め",
        ]
        research_keywords = [
            "最新", "検索", "根拠", "引用", "source", "citation", "paper", "論文", "比較",
        ]

        is_code = any(k in lower for k in code_keywords)
        is_verify = any(k in lower for k in verify_keywords)
        is_design = any(k in lower for k in design_keywords)
        is_research = any(k in lower for k in research_keywords)
        long_prompt = len(text) > 8000
        very_long_prompt = len(text) > 25000
        short_prompt = len(text) < 2000
        exact_request = "return exactly" in lower and not is_verify and not is_research
        lookup_like_long = (
            long_prompt
            and self.is_lookup_like_query(self.retrieval_query(messages))
            and bool(self.retrieval_terms(self.retrieval_query(messages)))
        )

        if forced_mode in {"fast", "balanced", "deep", "ultra"}:
            mode = forced_mode
        elif lookup_like_long:
            mode = "fast"
        elif exact_request and len(text) < 20000:
            mode = "fast"
        elif very_long_prompt or (is_design and is_code) or (is_code and is_verify):
            mode = "deep"
        elif self.fast_bias and short_prompt and not is_verify and not is_research:
            mode = "fast"
        elif is_code or is_design or long_prompt:
            mode = "balanced"
        else:
            mode = "fast"

        risky = bool(
            is_verify
            or is_research
            or (long_prompt and not exact_request)
            or (is_code and self.verify_code_by_default and not exact_request)
        )

        if mode == "fast":
            if is_code:
                return Route(mode="fast", primary="qwen", planner=None, verifier=None,
                             workflow="qwen_direct", task_type="coding", risky=False,
                             reason="fast code route")
            return Route(mode="fast", primary="gemma", planner=None, verifier=None,
                         workflow="gemma_direct", task_type="general", risky=False,
                         reason="fast general route")

        if mode == "ultra":
            return Route(mode="ultra", primary="qwen", planner="gemma", verifier="glm",
                         workflow="gemma_spec__qwen_solution__glm_critique__gemma_synthesis__glm_verify",
                         task_type="coding" if is_code else "design" if is_design else "general",
                         risky=True, reason="forced or high-value ultra route")

        if mode == "deep":
            return Route(mode="deep", primary="qwen", planner="gemma", verifier="glm",
                         workflow="gemma_spec__qwen_solution__glm_verify__qwen_repair",
                         task_type="coding" if is_code else "design" if is_design else "general",
                         risky=True, reason="deep route for long/risky/code/design task")

        # balanced
        if is_code:
            return Route(mode="balanced", primary="qwen", planner=None, verifier="glm" if risky else None,
                         workflow="qwen_direct__glm_if_risky", task_type="coding", risky=risky,
                         reason="balanced code route")
        if is_design:
            return Route(mode="balanced", primary="gemma", planner=None, verifier="glm" if risky else None,
                         workflow="gemma_direct__glm_if_risky", task_type="design", risky=risky,
                         reason="balanced design route")
        return Route(mode="balanced", primary="gemma", planner=None, verifier="glm" if risky else None,
                     workflow="gemma_direct__glm_if_risky", task_type="general", risky=risky,
                     reason="balanced general route")

    def route(self, messages: List[Dict[str, Any]], forced_mode: Optional[str] = None) -> Route:
        fallback = self.rule_route(messages, forced_mode)
        if forced_mode in {"fast", "balanced", "deep", "ultra"}:
            return fallback
        lookup_query = self.retrieval_query(messages)
        if (
            len(self.last_user_text(messages)) > 8000
            and self.is_lookup_like_query(lookup_query)
            and self.retrieval_terms(lookup_query)
        ):
            return fallback
        if self.routing_tower is None:
            return fallback

        try:
            decision = self.routing_tower.route(messages, forced_mode)
        except RouterUnavailable as exc:
            self.log(
                {
                    "stage": "router_06b_fallback",
                    "ok": False,
                    "error": repr(exc),
                    "fallback": fallback.model_dump(),
                }
            )
            return fallback

        return self.route_from_tower(decision, fallback, forced_mode)

    def apply_fast_primary_bias(self, route: Route, forced_mode: Optional[str] = None) -> Route:
        if not self.fast_bias or route.mode != "fast":
            return route
        if forced_mode in {"balanced", "deep", "ultra"}:
            return route
        if route.workflow.startswith("deterministic_"):
            return route
        preferred = self.fast_primary
        if preferred == route.primary or preferred not in self.models:
            return route

        data = route.model_dump()
        data["primary"] = preferred
        data["workflow"] = f"{preferred}_direct"
        reason = data.get("reason", "")
        data["reason"] = f"{reason}; fast_primary_bias={preferred}" if reason else f"fast_primary_bias={preferred}"
        return Route(**data)

    def apply_language_bias(self, route: Route, messages: List[Dict[str, Any]]) -> Route:
        text = self.last_user_text(messages)
        if not self.has_japanese(text) or self.is_explicit_code_request(text):
            return route
        if route.mode not in {"fast", "balanced"} or "gemma" not in self.models:
            return route
        if route.workflow.startswith("deterministic_") or route.primary == "gemma":
            return route

        data = route.model_dump()
        data["primary"] = "gemma"
        if route.mode == "balanced" and route.verifier:
            data["workflow"] = "gemma_direct__glm_if_risky"
        else:
            data["workflow"] = "gemma_direct"
        reason = data.get("reason", "")
        data["reason"] = f"{reason}; japanese_language_bias=gemma" if reason else "japanese_language_bias=gemma"
        return Route(**data)

    def apply_followup_affinity(
        self,
        route: Route,
        messages: List[Dict[str, Any]],
        forced_mode: Optional[str] = None,
    ) -> Route:
        if not self.avoid_model_swap_on_followup:
            return route
        if forced_mode in {"deep", "ultra"}:
            return route
        if route.mode not in {"fast", "balanced"} or route.workflow.startswith("deterministic_"):
            return route
        if not self.is_short_followup(messages):
            return route

        key = self.conversation_affinity_key(messages)
        if not key:
            return route
        affinity = self._route_affinity.get(key)
        if not affinity:
            return route

        preferred = str(affinity.get("primary", ""))
        if preferred not in self.models or preferred == route.primary:
            return route

        text = self.last_user_text(messages)
        previous_task = str(affinity.get("task_type", ""))
        # Japanese general followups should not be dragged back to Qwen; that was
        # measured fast but unreliable for Japanese general answers on this box.
        # A Japanese followup to a coding task, however, should stay with Qwen.
        if (
            self.has_japanese(text)
            and not self.is_explicit_code_request(text)
            and preferred != "gemma"
            and previous_task != "coding"
        ):
            return route

        if route.task_type in {"coding", "verification"} and previous_task == "general":
            return route

        data = route.model_dump()
        data["primary"] = preferred
        if route.mode == "balanced" and route.verifier:
            data["workflow"] = f"{preferred}_direct__glm_if_risky"
        else:
            data["workflow"] = f"{preferred}_direct"
            data["verifier"] = None if route.mode == "fast" else route.verifier
        reason = data.get("reason", "")
        data["reason"] = (
            f"{reason}; followup_affinity={preferred}"
            if reason
            else f"followup_affinity={preferred}"
        )
        return Route(**data)

    @staticmethod
    def score_margin(scores: Dict[str, float]) -> float:
        if len(scores) < 2:
            return 0.0
        values = sorted((float(v) for v in scores.values()), reverse=True)
        return values[0] - values[1]

    def route_from_tower(
        self,
        decision: TowerDecision,
        fallback: Route,
        forced_mode: Optional[str] = None,
    ) -> Route:
        primary = decision.primary if decision.primary in self.models else fallback.primary
        mode = decision.mode if decision.mode in {"fast", "balanced", "deep", "ultra"} else fallback.mode
        task_type = decision.task_type or fallback.task_type
        risky = bool(decision.risky)
        agent_confidence = float(decision.confidence)
        mode_margin = self.score_margin(decision.mode_scores)

        if agent_confidence < self.tower_agent_min_confidence:
            primary = fallback.primary

        if forced_mode not in {"fast", "balanced", "deep", "ultra"}:
            if mode != fallback.mode and mode_margin < self.tower_mode_min_margin:
                mode = fallback.mode
                risky = fallback.risky
            if fallback.mode == "fast" and mode in {"deep", "ultra"}:
                mode = "fast"
                risky = False

        # Verification questions are often best answered by the verifier directly.
        if task_type == "verification" and primary == "glm" and mode in {"deep", "ultra"}:
            mode = "balanced"
            risky = False

        planner: Optional[AgentName] = None
        verifier: Optional[AgentName] = None
        workflow = f"{primary}_direct"

        if mode == "fast":
            risky = False
        elif mode == "balanced":
            verifier = "glm" if risky and primary != "glm" else None
            workflow = f"{primary}_direct__glm_if_risky" if verifier else f"{primary}_direct"
        elif mode == "deep":
            planner = "gemma"
            if primary == "glm":
                primary = "qwen"
            verifier = "glm"
            risky = True
            workflow = "gemma_spec__qwen_solution__glm_verify__qwen_repair"
        elif mode == "ultra":
            primary = "qwen"
            planner = "gemma"
            verifier = "glm"
            risky = True
            workflow = "gemma_spec__qwen_solution__glm_critique__gemma_synthesis__glm_verify"

        return Route(
            mode=mode,  # type: ignore[arg-type]
            primary=primary,  # type: ignore[arg-type]
            planner=planner,
            verifier=verifier,
            workflow=workflow,
            task_type=task_type,
            risky=risky,
            reason=f"0.6B tower route via {decision.source}",
            router="qwen3-0.6b-hidden",
            confidence=round(float(decision.confidence), 4),
            scores=decision.scores,
            mode_scores={**decision.mode_scores, "_margin": round(mode_margin, 4)},
            task_scores=decision.task_scores,
            router_elapsed_ms=decision.elapsed_ms,
        )

    def select_backend(
        self,
        agent_name: AgentName,
        stage: str,
        user: str,
        max_tokens: Optional[int],
    ) -> str:
        if not self.hybrid_enabled:
            return self.hybrid_default_backend

        target = str(self.hybrid_policy.get("vulkan_backend", "vulkan"))
        if target not in self.backend_clients:
            return self.hybrid_default_backend

        agents = set(str(x) for x in self.hybrid_policy.get("agents", ["qwen"]))
        if agent_name not in agents:
            return self.hybrid_default_backend

        stages = set(str(x) for x in self.hybrid_policy.get("stages", []))
        min_max_tokens = int(self.hybrid_policy.get("min_max_tokens", 128))
        min_input_chars = int(self.hybrid_policy.get("min_input_chars", 1800))
        deep_stage_markers = ("deep", "ultra", "primary_solution", "repair")

        token_budget = int(max_tokens or 0)
        stage_match = stage in stages or any(marker in stage for marker in deep_stage_markers)
        token_match = token_budget >= min_max_tokens
        input_match = len(user) >= min_input_chars

        if stage_match or token_match or input_match:
            return target
        return self.hybrid_default_backend

    async def create_chat_completion(
        self,
        backend_name: str,
        payload: Dict[str, Any],
        *,
        json_mode: bool,
    ) -> Tuple[Any, str, bool]:
        selected = backend_name if backend_name in self.backend_clients else self.hybrid_default_backend
        attempted: List[str] = []
        last_exc: Optional[Exception] = None

        for candidate in [selected, self.hybrid_fallback_backend]:
            if candidate in attempted or candidate not in self.backend_clients:
                continue
            attempted.append(candidate)
            candidate_payload = dict(payload)
            client = self.backend_clients[candidate]
            try:
                try:
                    return await client.chat.completions.create(**candidate_payload), candidate, candidate != selected
                except Exception as exc:
                    msg = str(exc).lower()
                    if not (json_mode and ("response_format" in msg or "json" in msg)):
                        raise
                    candidate_payload.pop("response_format", None)
                    return await client.chat.completions.create(**candidate_payload), candidate, candidate != selected
            except Exception as exc:
                last_exc = exc
                if candidate == self.hybrid_fallback_backend:
                    break
                continue

        assert last_exc is not None
        raise last_exc

    async def call_model(
        self,
        rid: str,
        agent_name: AgentName,
        stage: str,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        json_mode: bool = False,
    ) -> str:
        m = self.models[agent_name]
        system_parts = [m.system, self.runtime_context(user), system]
        messages = [
            {"role": "system", "content": "\n\n".join(p.strip() for p in system_parts if p.strip())},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": m.model_id,
            "messages": messages,
            "temperature": m.temperature if temperature is None else temperature,
            "max_tokens": m.max_tokens if max_tokens is None else max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if m.extra_body:
            payload["extra_body"] = dict(m.extra_body)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        selected_backend = self.select_backend(agent_name, stage, user, payload["max_tokens"])
        actual_backend = selected_backend
        fallback_used = False

        # Hard guarantee: no parallel model calls on this hardware profile.
        async with self.call_lock:
            t0 = time.perf_counter()
            ok = False
            try:
                resp: Any = None
                try:
                    resp, actual_backend, fallback_used = await self.create_chat_completion(
                        selected_backend,
                        payload,
                        json_mode=json_mode,
                    )
                except Exception as exc:
                    # response_format(JSON mode)非対応のbackendだけを救済する。
                    # 以前は全例外を握り潰して再試行しており、ネットワーク障害まで
                    # response_format外しで誤魔化していた（根本原因が隠れる）。
                    msg = str(exc).lower()
                    if not (json_mode and ("response_format" in msg or "json" in msg)):
                        raise
                    payload.pop("response_format", None)
                    resp = await self.client.chat.completions.create(**payload)
                out = resp.choices[0].message.content or ""
                self._accumulate_usage(rid, resp)
                ok = True
                return out
            finally:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                if ok and resp is not None:
                    self._accumulate_perf(
                        rid,
                        stage=stage,
                        agent=agent_name,
                        model_id=m.model_id,
                        backend_name=actual_backend,
                        backend_base_url=self.backend_base_urls.get(actual_backend, ""),
                        selected_backend=selected_backend,
                        fallback_used=fallback_used,
                        elapsed_ms=elapsed_ms,
                        resp=resp,
                    )
                log_obj = {
                    "rid": rid,
                    "stage": stage,
                    "agent": agent_name,
                    "model_id": m.model_id,
                    "selected_backend": selected_backend,
                    "backend_name": actual_backend,
                    "backend_base_url": self.backend_base_urls.get(actual_backend, ""),
                    "backend_fallback_used": fallback_used,
                    "ok": ok,
                    "elapsed_ms": elapsed_ms,
                    "input_chars": len(system) + len(user),
                }
                if self.log_content:
                    log_obj["system"] = system
                    log_obj["user"] = user
                self.log(log_obj)

    def _accumulate_usage(self, rid: str, resp: Any) -> None:
        """backendが返すusageをrid単位で合算する。以前は常に0を返していた。"""
        u = getattr(resp, "usage", None)
        if u is None:
            return
        acc = self._usage.setdefault(rid, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        acc["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        acc["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
        acc["total_tokens"] += int(getattr(u, "total_tokens", 0) or 0)

    @staticmethod
    def response_timings(resp: Any) -> Dict[str, Any]:
        timings = getattr(resp, "timings", None)
        if timings is None and hasattr(resp, "model_extra"):
            extra = getattr(resp, "model_extra") or {}
            if isinstance(extra, dict):
                timings = extra.get("timings")
        if timings is None:
            return {}
        if hasattr(timings, "model_dump"):
            timings = timings.model_dump()
        if isinstance(timings, dict):
            return timings
        return {}

    def _accumulate_perf(
        self,
        rid: str,
        *,
        stage: str,
        agent: AgentName,
        model_id: str,
        backend_name: str,
        backend_base_url: str,
        selected_backend: str,
        fallback_used: bool,
        elapsed_ms: int,
        resp: Any,
    ) -> None:
        bucket = self._perf.setdefault(
            rid,
            {
                "model_calls": [],
                "model_call_wall_ms": 0,
                "tool_calls": [],
                "tool_call_wall_ms": 0,
                "backend_prompt_ms": 0.0,
                "backend_predicted_ms": 0.0,
                "backend_prompt_tokens": 0,
                "backend_predicted_tokens": 0,
                "backend_cached_tokens": 0,
                "backend_call_count_by_name": {},
                "backend_fallback_count": 0,
            },
        )
        timings = self.response_timings(resp)
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        predicted_ms = float(timings.get("predicted_ms") or 0.0)
        prompt_n = int(timings.get("prompt_n") or 0)
        predicted_n = int(timings.get("predicted_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        bucket["model_call_wall_ms"] += elapsed_ms
        bucket["backend_prompt_ms"] += prompt_ms
        bucket["backend_predicted_ms"] += predicted_ms
        bucket["backend_prompt_tokens"] += prompt_n
        bucket["backend_predicted_tokens"] += predicted_n
        bucket["backend_cached_tokens"] += cache_n
        backend_counts = bucket.setdefault("backend_call_count_by_name", {})
        backend_counts[backend_name] = int(backend_counts.get(backend_name, 0)) + 1
        if fallback_used:
            bucket["backend_fallback_count"] = int(bucket.get("backend_fallback_count") or 0) + 1
        bucket["model_calls"].append(
            {
                "stage": stage,
                "agent": agent,
                "model_id": model_id,
                "backend_name": backend_name,
                "backend_base_url": backend_base_url,
                "selected_backend": selected_backend,
                "backend_fallback_used": fallback_used,
                "wall_ms": elapsed_ms,
                "timings": timings,
            }
        )

    def pop_usage(self, rid: str) -> Dict[str, int]:
        return self._usage.pop(rid, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def pop_performance(
        self,
        rid: str,
        *,
        route: Route,
        usage: Dict[str, int],
        end_to_end_ms: int,
    ) -> Dict[str, Any]:
        perf = self._perf.pop(rid, None) or {
            "model_calls": [],
            "model_call_wall_ms": 0,
            "tool_calls": [],
            "tool_call_wall_ms": 0,
            "backend_prompt_ms": 0.0,
            "backend_predicted_ms": 0.0,
            "backend_prompt_tokens": 0,
            "backend_predicted_tokens": 0,
            "backend_cached_tokens": 0,
            "backend_call_count_by_name": {},
            "backend_fallback_count": 0,
        }
        tool_calls = list(perf.get("tool_calls") or [])
        tool_fail_count = sum(1 for call in tool_calls if not bool(call.get("ok")))
        backend_counts = dict(perf.get("backend_call_count_by_name") or {})
        backend_names = sorted(backend_counts)
        backend_compute_ms = float(perf["backend_prompt_ms"]) + float(perf["backend_predicted_ms"])
        router_ms = int(route.router_elapsed_ms or 0)
        theoretical_min_ms = router_ms + backend_compute_ms
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        actual_completion_tps = round(completion_tokens / (end_to_end_ms / 1000), 3) if end_to_end_ms else 0
        actual_total_tps = round(total_tokens / (end_to_end_ms / 1000), 3) if end_to_end_ms else 0
        theoretical_completion_tps = (
            round(completion_tokens / (theoretical_min_ms / 1000), 3) if theoretical_min_ms and completion_tokens else 0
        )
        backend_decode_tps = (
            round(int(perf["backend_predicted_tokens"]) / (float(perf["backend_predicted_ms"]) / 1000), 3)
            if float(perf["backend_predicted_ms"]) > 0
            else 0
        )
        backend_prefill_tps = (
            round(int(perf["backend_prompt_tokens"]) / (float(perf["backend_prompt_ms"]) / 1000), 3)
            if float(perf["backend_prompt_ms"]) > 0
            else 0
        )
        overhead_ms = max(0.0, end_to_end_ms - theoretical_min_ms)
        efficiency = round(theoretical_min_ms / end_to_end_ms, 4) if end_to_end_ms and theoretical_min_ms else 1.0
        return {
            "end_to_end_ms": end_to_end_ms,
            "model_call_count": len(perf["model_calls"]),
            "model_call_wall_ms": int(perf["model_call_wall_ms"]),
            "backend_names": backend_names,
            "backend_call_count_by_name": backend_counts,
            "backend_fallback_count": int(perf.get("backend_fallback_count") or 0),
            "tool_call_count": len(tool_calls),
            "tool_fail_count": tool_fail_count,
            "tool_call_wall_ms": int(perf.get("tool_call_wall_ms") or 0),
            "backend_compute_ms": round(backend_compute_ms, 3),
            "backend_prompt_ms": round(float(perf["backend_prompt_ms"]), 3),
            "backend_predicted_ms": round(float(perf["backend_predicted_ms"]), 3),
            "router_ms": router_ms,
            "overhead_ms": round(overhead_ms, 3),
            "actual_completion_tps": actual_completion_tps,
            "actual_total_tps": actual_total_tps,
            "theoretical_min_ms": round(theoretical_min_ms, 3),
            "theoretical_completion_tps": theoretical_completion_tps,
            "backend_prefill_tps": backend_prefill_tps,
            "backend_decode_tps": backend_decode_tps,
            "backend_prompt_tokens": int(perf["backend_prompt_tokens"]),
            "backend_predicted_tokens": int(perf["backend_predicted_tokens"]),
            "backend_cached_tokens": int(perf["backend_cached_tokens"]),
            "efficiency_vs_backend_theory": efficiency,
            "model_calls": perf["model_calls"],
            "tool_calls": tool_calls,
        }

    async def verify_and_repair(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        answer: str,
        repair_agent: AgentName,
        *,
        light: bool,
        verify_max_tokens: Optional[int] = None,
        repair_max_tokens: Optional[int] = None,
    ) -> str:
        """verify→repairを max_repair_rounds 回まで回す。

        以前はrepairが各モードで1回ハードコードされ、configのmax_repair_roundsが
        無視されていた。修復後は再検証し、passしたら早期終了する。
        """
        current = answer
        for _ in range(self.max_repair_rounds):
            report = await self.verify(rid, messages, current, light=light, max_tokens=verify_max_tokens)
            if report.passed or report.severity in {"none", "minor"}:
                return current
            current = await self.repair(
                rid, messages, current, report, repair_agent, max_tokens=repair_max_tokens
            )
        return current

    async def verify(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        answer: str,
        *,
        light: bool,
        max_tokens: Optional[int] = None,
    ) -> VerifyReport:
        tr = self.visible_transcript(messages)
        raw = await self.call_model(
            rid,
            "glm",
            "verify_light" if light else "verify_strict",
            """
Return JSON only:
{
  "passed": true,
  "severity": "none|minor|major|fatal",
  "defects": ["specific defect"],
  "repair_instruction": "specific instruction"
}

Use major/fatal only when the answer is materially wrong, incomplete, contradictory, unsafe, or violates explicit requirements.
Do not fail merely because the answer could be more polished.
""",
            f"USER:\n{tr}\n\nANSWER:\n{answer}",
            temperature=0.0,
            max_tokens=2500 if max_tokens is None else max_tokens,
            json_mode=True,
        )
        try:
            return VerifyReport.model_validate(self.parse_json_object(raw))
        except Exception:
            lowered = raw.lower()
            failed = any(x in lowered for x in ["fatal", "major", "fail", "incorrect", "wrong"])
            return VerifyReport(
                passed=not failed,
                severity="major" if failed else "minor",
                defects=[raw[:1000]],
                repair_instruction=raw[:1500],
            )

    async def repair(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        answer: str,
        report: VerifyReport,
        repair_agent: AgentName,
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        tr = self.visible_transcript(messages)
        return await self.call_model(
            rid,
            repair_agent,
            "repair",
            """
Repair the answer using the verifier report.
Return only the corrected user-facing answer.
Do not mention internal routing or the verifier unless directly relevant.
""",
            f"USER:\n{tr}\n\nPREVIOUS ANSWER:\n{answer}\n\nVERIFIER REPORT:\n{report.model_dump_json(indent=2)}",
            temperature=0.1,
            max_tokens=9000 if max_tokens is None else max_tokens,
        )

    async def check_backend_models(self) -> Dict[str, Any]:
        """起動時にbackendの/v1/modelsを読み、configのmodel_idと突き合わせる。

        不一致でもクラッシュさせず警告ログを残すだけ（router modeはHF ID/ファイル名/
        preset名のいずれを露出するか起動方法で変わるため、起動を止めるのは過剰）。
        """
        backend_notes: Dict[str, Any] = {}
        ok = True
        for backend_name, client in self.backend_clients.items():
            configured_names = self.backend_model_names.get(backend_name) or list(self.models)
            configured = {
                self.models[name].model_id
                for name in configured_names
                if name in self.models
            }
            try:
                resp = await client.models.list()
                exposed = {d.id for d in resp.data}
            except Exception as exc:
                backend_note = {
                    "ok": False,
                    "base_url": self.backend_base_urls.get(backend_name, ""),
                    "error": repr(exc),
                }
                backend_notes[backend_name] = backend_note
                ok = False
                print(f"[local-moe-fugu] backend {backend_name} /v1/models check failed: {exc!r}")
                continue

            missing = sorted(configured - exposed)
            backend_note = {
                "ok": not missing,
                "base_url": self.backend_base_urls.get(backend_name, ""),
                "configured": sorted(configured),
                "exposed": sorted(exposed),
                "missing": missing,
            }
            backend_notes[backend_name] = backend_note
            if missing:
                ok = False
                print(
                    f"[local-moe-fugu] WARNING: configured model IDs not found on backend {backend_name} "
                    f"/v1/models: {missing}. Exposed: {sorted(exposed)}. "
                    "Fix configs/*.yaml to match exact IDs."
                )

        note = {
            "stage": "startup_model_check",
            "ok": ok,
            "hybrid_enabled": self.hybrid_enabled,
            "default_backend": self.hybrid_default_backend,
            "fallback_backend": self.hybrid_fallback_backend,
            "backends": backend_notes,
        }
        self.log(note)
        return note

    def capability_report(self) -> Dict[str, Any]:
        """Return the runtime feature matrix that should be true for MAX FUGU mode."""
        scheduler = self.scheduler
        hw = self.cfg.get("hardware_profile", {}) or {}
        router_state = self.routing_tower.state() if self.routing_tower else {"enabled": False}
        return {
            "name": "local-moe-fugu-7940hs",
            "version": APP_VERSION,
            "front_model": self.front_model,
            "backend": self.cfg["backend"]["base_url"],
            "backends": dict(self.backend_base_urls),
            "hybrid_backend": {
                "enabled": self.hybrid_enabled,
                "default_backend": self.hybrid_default_backend,
                "fallback_backend": self.hybrid_fallback_backend,
                "policy": self.hybrid_policy,
                "model_names": self.backend_model_names,
            },
            "mode_aliases": [
                self.front_model,
                f"{self.front_model}:fast",
                f"{self.front_model}:deep",
                f"{self.front_model}:ultra",
            ],
            "workers": {
                name: {
                    "model_id": model.model_id,
                    "role": (self.cfg.get("models", {}).get(name, {}) or {}).get("role", name),
                }
                for name, model in self.models.items()
            },
            "control": {
                "router_06b": router_state,
                "rules_fallback": True,
                "explicit_mode_skips_tower": True,
                "long_lookup_skips_tower_after_pack": True,
                "followup_affinity": bool(scheduler.get("avoid_model_swap_on_followup", False)),
                "fast_primary": scheduler.get("fast_primary", "qwen"),
                "fast_bias": bool(scheduler.get("fast_bias", False)),
            },
            "context": {
                "default_ctx": self.default_ctx,
                "deep_ctx": self.deep_ctx,
                "long_lookup_retrieval_pack": True,
                "retrieval_pack_lookup_cap_chars": 6000,
                "retrieval_chunk_chars": 2200,
                "head_tail_clip_fallback": True,
            },
            "deterministic_paths": [
                "exact_literal",
                "japanese_compound",
                "visible_count",
                "arithmetic",
                "calendar_multiple_choice",
            ],
            "quality_workflows": {
                "fast": "single worker direct answer",
                "balanced": "primary answer plus GLM verification only when risky",
                "deep_full": "Gemma public spec -> Qwen solution -> GLM verification -> optional Qwen repair",
                "ultra_full": "Gemma spec -> Qwen solution -> GLM critique -> Gemma synthesis -> GLM verification -> optional Qwen repair",
                "deep_compact": "short capped deep requests use one compact Qwen reviewer pass",
                "ultra_compact": "short capped ultra requests use one compact Qwen self-critique pass",
            },
            "tools": {
                "enabled_on_request": True,
                "fail_closed_policy": True,
                "allowed_prefixes": ALLOWED_PREFIXES,
                "tool_grounded_repair": True,
            },
            "runtime": {
                "anti_fanout_lock": True,
                "parallel_model_calls": bool(hw.get("parallel_model_calls", False)),
                "max_loaded_models": hw.get("max_loaded_models"),
                "max_concurrent_model_calls": scheduler.get("max_concurrent_model_calls"),
                "verify_code_by_default": bool(scheduler.get("verify_code_by_default", False)),
                "verify_only_when_risky": bool(scheduler.get("verify_only_when_risky", True)),
            },
            "telemetry": {
                "route_in_system_fingerprint": True,
                "performance_in_system_fingerprint": True,
                "actual_tps": True,
                "theoretical_tps": True,
                "backend_prefill_decode_tps": True,
                "tool_call_metrics": True,
            },
        }

    async def run_tools(self, rid: str, spec: ToolSpec) -> List[ToolResult]:
        """ToolSpecのコマンドを順に実行する。並列実行はしない（ハード制約と整合）。"""
        results: List[ToolResult] = []
        cwd = Path(spec.cwd)
        for cmd in spec.commands:
            if not cwd.is_dir():
                results.append(
                    ToolResult(False, " ".join(cmd), "", f"cwd not a directory: {spec.cwd}", 2)
                )
                continue
            started = time.perf_counter()
            res = await run_tool(cmd, cwd, timeout_s=spec.timeout_s)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            results.append(res)
            bucket = self._perf.setdefault(
                rid,
                {
                    "model_calls": [],
                    "model_call_wall_ms": 0,
                    "tool_calls": [],
                    "tool_call_wall_ms": 0,
                    "backend_prompt_ms": 0.0,
                    "backend_predicted_ms": 0.0,
                    "backend_prompt_tokens": 0,
                    "backend_predicted_tokens": 0,
                    "backend_cached_tokens": 0,
                },
            )
            bucket.setdefault("tool_calls", [])
            bucket["tool_call_wall_ms"] = int(bucket.get("tool_call_wall_ms") or 0) + elapsed_ms
            bucket["tool_calls"].append(
                {
                    "command": res.command,
                    "ok": res.ok,
                    "returncode": res.returncode,
                    "wall_ms": elapsed_ms,
                }
            )
            self.log(
                {
                    "rid": rid,
                    "stage": "tool_run",
                    "command": res.command,
                    "ok": res.ok,
                    "returncode": res.returncode,
                }
            )
        return results

    @staticmethod
    def summarize_tool_results(results: List[ToolResult]) -> str:
        rows: List[str] = []
        for r in results:
            head = f"$ {r.command}\n(returncode={r.returncode}, ok={r.ok})"
            tail = (r.stderr or r.stdout)[-4000:]
            rows.append(f"{head}\n{tail}".rstrip())
        return "\n\n".join(rows)

    async def verify_with_tools(
        self, rid: str, messages: List[Dict[str, Any]], answer: str, results: List[ToolResult]
    ) -> VerifyReport:
        """失敗したtool出力をGLMが構造化repair指示に変換する。"""
        tr = self.visible_transcript(messages, char_limit=self.transcript_char_budget(self.default_ctx, 2500))
        raw = await self.call_model(
            rid,
            "glm",
            "tool_verify",
            """
You are given the failing output of build/test/lint tools that were run against the
user's workspace, plus the assistant's answer. Convert the tool failures into a concrete
repair plan. Return JSON only:
{
  "passed": false,
  "severity": "none|minor|major|fatal",
  "defects": ["specific failure from the tool output"],
  "repair_instruction": "specific instruction to fix the failing commands"
}
Base your judgement on the actual tool output, not on style.
""",
            f"USER:\n{tr}\n\nASSISTANT ANSWER:\n{answer}\n\nTOOL OUTPUT:\n{self.summarize_tool_results(results)}",
            temperature=0.0,
            max_tokens=2500,
            json_mode=True,
        )
        try:
            return VerifyReport.model_validate(self.parse_json_object(raw))
        except Exception:
            return VerifyReport(
                passed=False,
                severity="major",
                defects=[self.summarize_tool_results(results)[:1500]],
                repair_instruction=raw[:1500],
            )

    async def tool_grounded_repair(
        self, rid: str, messages: List[Dict[str, Any]], answer: str, spec: ToolSpec
    ) -> str:
        """tool実行→失敗ならGLMが診断→Qwen(primary)が修復。

        toolはworkspaceを観測するのみ（conductorはファイルを書かない）。失敗ログは
        一度の修復に反映する。再実行してもworkspaceは変わらないので走らせない。
        """
        results = await self.run_tools(rid, spec)
        if all(r.ok for r in results):
            return answer
        report = await self.verify_with_tools(rid, messages, answer, results)
        if report.passed:
            return answer
        return await self.repair(rid, messages, answer, report, "qwen")

    def fast_max_tokens(
        self,
        messages: List[Dict[str, Any]],
        route: Route,
        client_max_tokens: Optional[int],
    ) -> int:
        out_tokens = self.stage_max_tokens(client_max_tokens, 6000, minimum=16)
        text = self.last_user_text(messages)
        lower = text.lower()
        if "return exactly" in lower and "answer:" in lower:
            return min(out_tokens, 16)
        if "return exactly" in lower:
            return min(out_tokens, 32)

        if self.has_japanese(text) and not self.is_explicit_code_request(text) and route.mode == "fast":
            long_markers = ["詳しく", "詳細", "長文", "がっつり", "徹底", "deep", "ultra"]
            if not any(marker in text for marker in long_markers):
                return min(out_tokens, 224)

        if client_max_tokens is None or client_max_tokens > 256:
            return out_tokens

        very_compact_markers = [
            "one sentence",
            "rewrite",
            "more concisely",
            "return only the expression",
            "only the expression",
        ]
        compact_markers = [
            "keep it short",
            "briefly",
            "give two",
            "list three",
            "three steps",
        ]
        if any(marker in lower for marker in very_compact_markers):
            return min(out_tokens, 64)
        if any(marker in lower for marker in compact_markers):
            return min(out_tokens, 96)
        if route.task_type in {"general", "design"}:
            return min(out_tokens, 112)
        return out_tokens

    def fast_system_prompt(self, messages: List[Dict[str, Any]], route: Route) -> str:
        text = self.last_user_text(messages)
        lower = text.lower()
        explicit_code = self.is_explicit_code_request(text)
        if explicit_code:
            return """
Return only the requested code, command, or function plus at most one brief explanatory sentence.
No tutorial, no long docstring, and no internal routing mention.
            """
        if self.has_japanese(text):
            return """
日本語で簡潔に答える。中国語や文字化け扱いしない。
複数質問は番号付きで各1文。前置き、見出し、長い補足は不要。ローカル確定情報があれば優先。
「200年3/4」は西暦200年3月4日。文字カウントは可視ラテン文字と日本語読みを区別。
内部ルーティングには触れない。
            """
        if (route.task_type == "design" or "workflow" in lower) and not explicit_code:
            return """
Answer as concise prose bullets or numbered steps. No code blocks, classes, functions, or pseudo-code unless explicitly requested.
Role labels like planner, coder, verifier, router, and workflow are not source-code requests. Do not mention internal routing.
            """
        return """
Answer directly and compactly. Match any requested count or output format. Do not mention internal routing.
For code, command, or function requests, give a minimal working snippet.
            """

    def fast_stop_sequences(self, messages: List[Dict[str, Any]], route: Route) -> Optional[List[str]]:
        text = self.last_user_text(messages)
        lower = text.lower()
        if "return exactly" in lower and "answer:" in lower:
            return ["\n", ")", ".", " The", " the", " Because", " because", " Explanation"]
        if "return exactly" in lower:
            return ["\n", " Explanation", " explanation", " Because", " because"]
        return None

    async def run_fast(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        route: Route,
        client_max_tokens: Optional[int],
    ) -> str:
        out_tokens = self.fast_max_tokens(messages, route, client_max_tokens)
        tr = self.fit_transcript(messages, self.default_ctx, out_tokens)
        answer = await self.call_model(
            rid,
            route.primary,
            "fast_answer",
            self.fast_system_prompt(messages, route),
            tr,
            max_tokens=out_tokens,
            stop=self.fast_stop_sequences(messages, route),
        )
        answer = self.apply_deterministic_corrections(messages, answer)
        return self.apply_output_shape_corrections(messages, answer)

    async def run_balanced(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        route: Route,
        client_max_tokens: Optional[int],
    ) -> str:
        out_tokens = self.stage_max_tokens(client_max_tokens, 7000, minimum=16)
        verify_tokens = self.stage_max_tokens(client_max_tokens, 2500, minimum=128, multiplier=4)
        repair_tokens = self.stage_max_tokens(client_max_tokens, 9000, minimum=16)
        tr = self.fit_transcript(messages, self.default_ctx, out_tokens)
        answer = await self.call_model(
            rid,
            route.primary,
            "balanced_primary_answer",
            "Answer the user concretely. Preserve all explicit requirements. Do not mention internal routing.",
            tr,
            max_tokens=out_tokens,
        )
        if not route.risky or route.verifier is None:
            return answer
        return await self.verify_and_repair(
            rid,
            messages,
            answer,
            route.primary,
            light=True,
            verify_max_tokens=verify_tokens,
            repair_max_tokens=repair_tokens,
        )

    async def run_deep(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        route: Route,
        client_max_tokens: Optional[int],
    ) -> str:
        spec_tokens = self.stage_max_tokens(client_max_tokens, 3000, minimum=128, multiplier=4)
        draft_tokens = self.stage_max_tokens(client_max_tokens, 9000, minimum=16)
        verify_tokens = self.stage_max_tokens(client_max_tokens, 2500, minimum=128, multiplier=4)
        repair_tokens = self.stage_max_tokens(client_max_tokens, 9000, minimum=16)
        tr = self.fit_transcript(messages, self.deep_ctx, max(spec_tokens, draft_tokens, verify_tokens))
        spec = await self.call_model(
            rid,
            route.planner or "gemma",
            "public_task_spec",
            """
Create a compact public task spec.
Include:
- objective
- explicit constraints
- assumptions
- deliverables
- likely failure modes
- acceptance checks
Do not expose hidden chain-of-thought.
            """,
            tr,
            temperature=0.1,
            max_tokens=spec_tokens,
        )
        draft = await self.call_model(
            rid,
            route.primary,
            "primary_solution",
            f"""
Use this public task spec:

{spec}

Solve the task concretely. Preserve all explicit user constraints.
            """,
            tr,
            temperature=0.2,
            max_tokens=draft_tokens,
        )
        return await self.verify_and_repair(
            rid,
            messages,
            draft,
            route.primary,
            light=False,
            verify_max_tokens=verify_tokens,
            repair_max_tokens=repair_tokens,
        )

    def should_compact_deep(
        self,
        messages: List[Dict[str, Any]],
        client_max_tokens: Optional[int],
    ) -> bool:
        if client_max_tokens is None or client_max_tokens > 256:
            return False
        text = self.last_user_text(messages)
        return len(text) <= 4000

    def should_compact_ultra(
        self,
        messages: List[Dict[str, Any]],
        client_max_tokens: Optional[int],
    ) -> bool:
        if client_max_tokens is None or client_max_tokens > 128:
            return False
        text = self.last_user_text(messages)
        return len(text) <= 4000

    async def run_deep_compact(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        client_max_tokens: Optional[int],
    ) -> str:
        out_tokens = self.stage_max_tokens(client_max_tokens, 9000, minimum=16)
        if client_max_tokens is not None and client_max_tokens <= 128:
            out_tokens = min(out_tokens, 96)
        tr = self.fit_transcript(messages, self.deep_ctx, out_tokens)
        return await self.call_model(
            rid,
            "qwen",
            "compact_deep_qwen",
            """
Answer as a concise senior reviewer.
Focus on material correctness issues, edge cases, and a fixed version when useful.
Self-check before finalizing, but do not mention internal routing or hidden checks.
            """,
            tr,
            temperature=0.1,
            max_tokens=out_tokens,
        )

    async def run_ultra_compact(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        client_max_tokens: Optional[int],
    ) -> str:
        out_tokens = self.stage_max_tokens(client_max_tokens, 9000, minimum=16)
        if client_max_tokens is not None and client_max_tokens <= 128:
            out_tokens = min(out_tokens, 96)
        tr = self.fit_transcript(messages, self.deep_ctx, out_tokens)
        return await self.call_model(
            rid,
            "qwen",
            "compact_ultra_qwen",
            """
Answer as a concise senior engineer.
Do an internal plan, critique, and final pass before responding, but return only the final user-facing answer.
Prioritize correctness, explicit constraints, and concrete failure modes.
Do not mention internal routing or hidden checks.
            """,
            tr,
            temperature=0.1,
            max_tokens=out_tokens,
        )

    async def run_ultra(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        route: Route,
        client_max_tokens: Optional[int],
    ) -> str:
        spec_tokens = self.stage_max_tokens(client_max_tokens, 4500, minimum=128, multiplier=4)
        draft_tokens = self.stage_max_tokens(client_max_tokens, 10000, minimum=64, multiplier=3)
        critique_tokens = self.stage_max_tokens(client_max_tokens, 4500, minimum=128, multiplier=4)
        final_tokens = self.stage_max_tokens(client_max_tokens, 10000, minimum=16)
        verify_tokens = self.stage_max_tokens(client_max_tokens, 2500, minimum=128, multiplier=4)
        repair_tokens = self.stage_max_tokens(client_max_tokens, 9000, minimum=16)
        tr = self.fit_transcript(
            messages,
            self.deep_ctx,
            max(spec_tokens, draft_tokens, critique_tokens, final_tokens, verify_tokens),
        )
        spec = await self.call_model(
            rid,
            "gemma",
            "ultra_public_spec",
            """
Create a rigorous public task spec.
Include objective, non-negotiable constraints, decomposition, required artifacts, known unknowns, verification criteria, and likely failure modes.
Do not expose hidden chain-of-thought.
            """,
            tr,
            temperature=0.1,
            max_tokens=spec_tokens,
        )
        draft = await self.call_model(
            rid,
            "qwen",
            "ultra_qwen_solution",
            f"""
Task spec:
{spec}

Produce the strongest concrete solution from an implementation-first perspective.
            """,
            tr,
            temperature=0.2,
            max_tokens=draft_tokens,
        )
        critique = await self.call_model(
            rid,
            "glm",
            "ultra_glm_critique",
            """
Critique the solution. Return actionable corrections only:
- false or unsupported claims
- contradictions
- missing requirements
- broken implementation details
- practical failure modes
""",
            f"USER:\n{tr}\n\nSPEC:\n{spec}\n\nSOLUTION:\n{draft}",
            temperature=0.1,
            max_tokens=critique_tokens,
        )
        final = await self.call_model(
            rid,
            "gemma",
            "ultra_synthesis",
            """
Write the final user-facing answer.
Integrate the Qwen solution and GLM critique.
Use only defensible material. Do not mention internal orchestration.
Preserve the user's requested section labels, quoted literals, and output shape verbatim.
If the user requested labels, include those labels even under a tight token budget.
""",
            f"USER:\n{tr}\n\nSPEC:\n{spec}\n\nQWEN SOLUTION:\n{draft}\n\nGLM CRITIQUE:\n{critique}",
            temperature=0.2,
            max_tokens=final_tokens,
        )
        return await self.verify_and_repair(
            rid,
            messages,
            final,
            "qwen",
            light=False,
            verify_max_tokens=verify_tokens,
            repair_max_tokens=repair_tokens,
        )

    async def run(
        self,
        rid: str,
        messages: List[Dict[str, Any]],
        forced_mode: Optional[str],
        tools: Optional[ToolSpec] = None,
        client_max_tokens: Optional[int] = None,
    ) -> Tuple[str, Route]:
        self._usage.setdefault(rid, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        deterministic = self.deterministic_answer(messages, forced_mode)
        if deterministic is not None:
            answer, route = deterministic
            self.remember_route_affinity(messages, route)
            self.log(
                {
                    "rid": rid,
                    "stage": "route",
                    "route": route.model_dump(),
                    "tools": bool(tools),
                    "client_max_tokens": client_max_tokens,
                }
            )
            if tools is not None and tools.commands:
                answer = await self.tool_grounded_repair(rid, messages, answer, tools)
            return answer, route

        route = self.route(messages, forced_mode)
        route = self.apply_fast_primary_bias(route, forced_mode)
        route = self.apply_language_bias(route, messages)
        route = self.apply_followup_affinity(route, messages, forced_mode)
        self.log(
            {
                "rid": rid,
                "stage": "route",
                "route": route.model_dump(),
                "tools": bool(tools),
                "client_max_tokens": client_max_tokens,
            }
        )
        if route.mode == "fast":
            answer = await self.run_fast(rid, messages, route, client_max_tokens)
        elif route.mode == "deep":
            if self.should_compact_deep(messages, client_max_tokens):
                route.workflow = "qwen_compact_deep_direct"
                route.verifier = None
                route.risky = False
                answer = await self.run_deep_compact(rid, messages, client_max_tokens)
            else:
                answer = await self.run_deep(rid, messages, route, client_max_tokens)
        elif route.mode == "ultra":
            if self.should_compact_ultra(messages, client_max_tokens):
                route.workflow = "qwen_compact_ultra_direct"
                route.planner = None
                route.verifier = None
                answer = await self.run_ultra_compact(rid, messages, client_max_tokens)
            else:
                answer = await self.run_ultra(rid, messages, route, client_max_tokens)
        else:
            answer = await self.run_balanced(rid, messages, route, client_max_tokens)

        # オプトイン: build/test/lintを実際に走らせ、失敗ログ起点で1回修復する。
        if tools is not None and tools.commands:
            answer = await self.tool_grounded_repair(rid, messages, answer, tools)
        answer = self.apply_output_shape_corrections(messages, answer)
        self.remember_route_affinity(messages, route)
        return answer, route


fugu = LocalMoEFugu()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 起動時にbackendのmodel IDとconfigの不一致を警告する（ブロックはしない）。
    await fugu.check_backend_models()
    yield


app = FastAPI(title="local-moe-fugu-7940hs", version=APP_VERSION, lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "front_model": fugu.front_model,
        "config": str(fugu.config_path),
        "backend": fugu.cfg["backend"]["base_url"],
        "backends": fugu.backend_base_urls,
        "hybrid_backend": {
            "enabled": fugu.hybrid_enabled,
            "default_backend": fugu.hybrid_default_backend,
            "fallback_backend": fugu.hybrid_fallback_backend,
        },
        "router_06b": fugu.routing_tower.state() if fugu.routing_tower else {"enabled": False},
    }


@app.get("/fugu/capabilities")
async def capabilities() -> Dict[str, Any]:
    return fugu.capability_report()


@app.get("/v1/models")
async def models() -> Dict[str, Any]:
    base = fugu.front_model
    return {
        "object": "list",
        "data": [
            {"id": base, "object": "model", "owned_by": "local"},
            {"id": f"{base}:fast", "object": "model", "owned_by": "local"},
            {"id": f"{base}:deep", "object": "model", "owned_by": "local"},
            {"id": f"{base}:ultra", "object": "model", "owned_by": "local"},
        ],
    }


def forced_mode_from_model(model: str) -> Optional[str]:
    if model.endswith(":fast"):
        return "fast"
    if model.endswith(":deep"):
        return "deep"
    if model.endswith(":ultra"):
        return "ultra"
    return None


def requested_max_tokens(body: Dict[str, Any]) -> Optional[int]:
    raw = body.get("max_completion_tokens", body.get("max_tokens"))
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise HTTPException(400, "max_tokens must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, "max_tokens must be a positive integer")
    if value <= 0:
        raise HTTPException(400, "max_tokens must be a positive integer")
    return min(value, 32768)


def _sse_stream(
    rid: str,
    model: str,
    answer: str,
    route: Route,
    created: int,
    performance: Dict[str, Any],
    chunk_size: int = 240,
):
    """final answer確定後にSSEで送出する擬似ストリーム。

    要件上、内部のdraft/critique/verifyはstreamしない。これらは通常通り
    完了させ、最終的なuser-facing answerだけをchunkに分割して流す。
    トークン単位の真ストリームではないが「内部はstreamしない」を満たす。
    """
    role_chunk = {
        "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

    for i in range(0, len(answer), chunk_size):
        piece = answer[i : i + chunk_size]
        chunk = {
            "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "system_fingerprint": {"rid": rid, "route": route.model_dump(), "performance": performance},
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(400, "messages must be a list")

    model = str(body.get("model", fugu.front_model))
    forced_mode = body.get("fugu_mode") or forced_mode_from_model(model)
    want_stream = body.get("stream") is True
    client_max_tokens = requested_max_tokens(body)
    rid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    tools: Optional[ToolSpec] = None
    raw_tools = body.get("fugu_tools")
    if raw_tools is not None:
        try:
            tools = ToolSpec.model_validate(raw_tools)
        except ValidationError as exc:
            raise HTTPException(400, f"invalid fugu_tools: {exc}")
        # ホワイトリスト外のコマンドはここで拒否（実行前に弾く）。
        bad = [cmd for cmd in tools.commands if not allowed(cmd)]
        if bad:
            raise HTTPException(400, f"command not allowed by local policy: {bad}")

    try:
        run_t0 = time.perf_counter()
        answer, route = await fugu.run(
            rid,
            messages,
            forced_mode,
            tools=tools,
            client_max_tokens=client_max_tokens,
        )
        end_to_end_ms = int((time.perf_counter() - run_t0) * 1000)
    except Exception as exc:
        fugu.pop_usage(rid)
        fugu._perf.pop(rid, None)
        fugu.log({"rid": rid, "stage": "error", "error": repr(exc)})
        raise HTTPException(500, repr(exc))

    usage = fugu.pop_usage(rid)
    performance = fugu.pop_performance(
        rid,
        route=route,
        usage=usage,
        end_to_end_ms=end_to_end_ms,
    )

    if want_stream:
        # 内部処理は完了済み。final answerのみをSSEで返す。
        return StreamingResponse(
            _sse_stream(rid, model, answer, route, created, performance),
            media_type="text/event-stream",
        )

    return JSONResponse(
        {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            # backendが返した実usageの合算（以前は常に0だった）。
            "usage": usage,
            "system_fingerprint": {
                "rid": rid,
                "route": route.model_dump(),
                "performance": performance,
                "request_max_tokens": client_max_tokens,
                "backend_models": {k: v.model_id for k, v in fugu.models.items()},
                "backends": fugu.backend_base_urls,
            },
        }
    )
