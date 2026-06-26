from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


AGENTS = ("gemma", "qwen", "glm")
MODES = ("fast", "balanced", "deep", "ultra")
TASK_TYPES = ("general", "coding", "design", "verification", "research")

DEFAULT_AGENT_PROFILES: Dict[str, List[str]] = {
    "gemma": [
        "general conversation, synthesis, explanation, planning, rewrite, summary",
        "compose a clear answer, organize ideas, make tradeoffs concise",
        "product thinking, architecture outline, non-code reasoning",
    ],
    "qwen": [
        "software engineering, code, debugging, terminal commands, repository edits",
        "algorithms, Python, PowerShell, API design, implementation details",
        "turn a requirement into working code and tests",
    ],
    "glm": [
        "verification, critique, audit, risk detection, factual checking",
        "find contradictions, missing requirements, unsafe assumptions",
        "review an answer and decide whether it is correct enough",
    ],
}

DEFAULT_MODE_PROFILES: Dict[str, List[str]] = {
    "fast": [
        "short direct answer, ordinary question, no verification required",
        "simple instruction that should be answered by one model",
    ],
    "balanced": [
        "normal implementation or design task, one primary worker is enough",
        "moderate complexity, answer concretely without a long workflow",
    ],
    "deep": [
        "risky, long, complex, or correctness-sensitive task needing verification",
        "multi-step engineering work with possible failure modes",
    ],
    "ultra": [
        "explicit request for strongest possible multi-agent workflow",
        "high-value task requiring planner, worker, critic, and final verification",
    ],
}

DEFAULT_TASK_PROFILES: Dict[str, List[str]] = {
    "general": ["general question, explanation, summary, rewrite"],
    "coding": ["code, debugging, scripts, tests, stack traces, APIs"],
    "design": ["architecture, workflow, system design, product design"],
    "verification": ["review, verify, audit, check correctness, risk"],
    "research": ["latest information, citations, papers, sources, comparison"],
}


class RouterUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class TowerDecision:
    primary: str
    mode: str
    task_type: str
    risky: bool
    confidence: float
    source: str
    elapsed_ms: int
    scores: Dict[str, float]
    mode_scores: Dict[str, float]
    task_scores: Dict[str, float]


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


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def normalize_vec(vector: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vector))
    if denom == 0:
        return vector
    return vector / denom


def stable_path_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def clip_middle(text: str, char_limit: int) -> str:
    if len(text) <= char_limit:
        return text
    marker = "\n\n[...middle omitted to fit router context...]\n\n"
    if char_limit <= len(marker) + 200:
        return text[-char_limit:]
    head = max(1000, int(char_limit * 0.35))
    tail = char_limit - head - len(marker)
    if tail < 1000:
        tail = max(200, int(char_limit * 0.45))
        head = char_limit - tail - len(marker)
    return text[:head] + marker + text[-tail:]


class ZeroSixBTower:
    """Qwen3-0.6B hidden-state router.

    This is the local Fugu control tower path: the 0.6B model does not generate
    the final answer. It produces one hidden vector, then a small linear head
    scores local workers. If a trained head is absent, a prototype head is built
    from semantic worker descriptions so the system can run immediately.
    """

    def __init__(self, cfg: Dict[str, Any], project_root: Path) -> None:
        self.cfg = dict(cfg or {})
        self.project_root = project_root
        self.enabled = bool(self.cfg.get("enabled", False))
        self.fail_closed = bool(self.cfg.get("fail_closed", False))
        self.retry_after_s = int(self.cfg.get("retry_after_s", 300))
        self.model_path = str(
            self.cfg.get("model_path")
            or os.getenv("FUGU_ROUTER_MODEL")
            or "Qwen/Qwen3-0.6B"
        )
        self.head_path = self._resolve_optional_path(self.cfg.get("head_path"))
        self.cache_dir = self._resolve_path(
            self.cfg.get("cache_dir") or (project_root / ".cache" / "fugu_router_06b")
        )
        self.max_input_tokens = int(self.cfg.get("max_input_tokens", 1536))
        self.max_input_chars = int(self.cfg.get("max_input_chars", 12000))
        self.hidden_position = int(self.cfg.get("hidden_position", -2))
        self.device = str(self.cfg.get("device") or os.getenv("FUGU_ROUTER_DEVICE") or "cpu")
        self.dtype_name = str(self.cfg.get("dtype") or "float32")
        self.torch_threads = int(self.cfg.get("torch_threads", max(1, min(6, (os.cpu_count() or 4) // 2))))
        self.mode_margin_deep = float(self.cfg.get("mode_margin_deep", 0.04))
        self.risk_threshold = float(self.cfg.get("risk_threshold", 0.36))
        self.confidence_temperature = float(self.cfg.get("confidence_temperature", 4.0))

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._agent_names = list(AGENTS)
        self._mode_names = list(MODES)
        self._task_names = list(TASK_TYPES)
        self._agent_head: Optional[np.ndarray] = None
        self._mode_head: Optional[np.ndarray] = None
        self._task_head: Optional[np.ndarray] = None
        self._source = "unloaded"
        self._last_error: Optional[str] = None
        self._next_retry_at = 0.0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], project_root: Path) -> Optional["ZeroSixBTower"]:
        tower_cfg = cfg.get("router_06b") or {}
        if not bool(tower_cfg.get("enabled", False)):
            return None
        return cls(tower_cfg, project_root)

    def state(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loaded": self._model is not None,
            "source": self._source,
            "model_path": self.model_path,
            "head_path": str(self.head_path) if self.head_path else None,
            "last_error": self._last_error,
        }

    def route(self, messages: List[Dict[str, Any]], forced_mode: Optional[str] = None) -> TowerDecision:
        if not self.enabled:
            raise RouterUnavailable("0.6B router disabled")

        now = time.time()
        if self._last_error and now < self._next_retry_at:
            raise RouterUnavailable(self._last_error)

        started = time.perf_counter()
        try:
            self._ensure_loaded()
            assert self._agent_head is not None
            assert self._mode_head is not None
            assert self._task_head is not None

            transcript = self.format_transcript(messages)
            h = self._hidden(transcript)
            h = normalize_vec(h)

            agent_logits = self._agent_head @ h
            mode_logits = self._mode_head @ h
            task_logits = self._task_head @ h
            agent_probs = softmax((agent_logits * self.confidence_temperature).tolist())
            mode_probs = softmax((mode_logits * self.confidence_temperature).tolist())
            task_probs = softmax((task_logits * self.confidence_temperature).tolist())

            primary = self._agent_names[int(np.argmax(agent_logits))]
            mode = forced_mode if forced_mode in MODES else self._mode_names[int(np.argmax(mode_logits))]
            task_type = self._task_names[int(np.argmax(task_logits))]

            risk_score = 0.0
            for name in ("deep", "ultra"):
                if name in self._mode_names:
                    risk_score = max(risk_score, mode_probs[self._mode_names.index(name)])
            for name in ("verification", "research"):
                if name in self._task_names:
                    risk_score = max(risk_score, task_probs[self._task_names.index(name)])
            risky = bool(mode in {"deep", "ultra"} or risk_score >= self.risk_threshold)
            if mode == "balanced" and risk_score >= (self.risk_threshold + self.mode_margin_deep):
                mode = "deep"

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self._last_error = None
            return TowerDecision(
                primary=primary,
                mode=mode,
                task_type=task_type,
                risky=risky,
                confidence=float(max(agent_probs) if agent_probs else 0.0),
                source=self._source,
                elapsed_ms=elapsed_ms,
                scores={k: round(float(v), 4) for k, v in zip(self._agent_names, agent_logits.tolist())},
                mode_scores={k: round(float(v), 4) for k, v in zip(self._mode_names, mode_logits.tolist())},
                task_scores={k: round(float(v), 4) for k, v in zip(self._task_names, task_logits.tolist())},
            )
        except Exception as exc:
            self._last_error = repr(exc)
            self._next_retry_at = time.time() + self.retry_after_s
            if self.fail_closed:
                raise
            raise RouterUnavailable(self._last_error) from exc

    def format_transcript(self, messages: List[Dict[str, Any]]) -> str:
        rows: List[str] = []
        for msg in messages:
            role = str(msg.get("role", "user")).lower()
            text = content_to_text(msg.get("content", ""))
            if role == "system":
                text = text[:2000]
            rows.append(f"{role}: {text}")
        text = "\n".join(rows)
        if len(text) > self.max_input_chars:
            return clip_middle(text, self.max_input_chars)
        return text

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RouterUnavailable(
                "missing router dependencies; run scripts/setup_fugu_router_06b.ps1"
            ) from exc

        self._torch = torch
        if self.torch_threads > 0:
            torch.set_num_threads(self.torch_threads)

        dtype = getattr(torch, self.dtype_name, torch.float32)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).eval()
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                trust_remote_code=True,
            ).eval()
        self._model.to(self.device)

        if self.head_path and self.head_path.exists():
            self._load_head(self.head_path)
            self._source = f"trained_head:{self.head_path}"
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"prototype_head_{stable_path_id(self.model_path)}.npz"
        if cache_path.exists():
            self._load_head(cache_path)
            self._source = f"prototype_cache:{cache_path}"
            return

        self._build_prototype_head()
        np.savez(
            cache_path,
            agent_names=np.array(self._agent_names),
            mode_names=np.array(self._mode_names),
            task_names=np.array(self._task_names),
            agent_head=self._agent_head,
            mode_head=self._mode_head,
            task_head=self._task_head,
            normalize=np.array([1], dtype=np.int32),
        )
        self._source = f"prototype_built:{cache_path}"

    def _load_head(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        agent_names = data["agent_names"] if "agent_names" in data.files else np.array(AGENTS)
        mode_names = data["mode_names"] if "mode_names" in data.files else np.array(MODES)
        task_names = data["task_names"] if "task_names" in data.files else np.array(TASK_TYPES)
        self._agent_names = [str(x) for x in agent_names.tolist()]
        self._mode_names = [str(x) for x in mode_names.tolist()]
        self._task_names = [str(x) for x in task_names.tolist()]
        self._agent_head = np.asarray(data["agent_head"], dtype=np.float32)
        self._mode_head = np.asarray(data["mode_head"], dtype=np.float32)
        self._task_head = np.asarray(data["task_head"], dtype=np.float32)
        normalize = data["normalize"] if "normalize" in data.files else np.array([0])
        if int(np.asarray(normalize)[0]) == 1:
            self._agent_head = normalize_rows(self._agent_head)
            self._mode_head = normalize_rows(self._mode_head)
            self._task_head = normalize_rows(self._task_head)

        for name in self._agent_names:
            if name not in AGENTS:
                raise ValueError(f"unknown local agent in router head: {name}")
        for name in self._mode_names:
            if name not in MODES:
                raise ValueError(f"unknown mode in router head: {name}")

    def _build_prototype_head(self) -> None:
        self._agent_names = list(AGENTS)
        self._mode_names = list(MODES)
        self._task_names = list(TASK_TYPES)
        self._agent_head = self._profile_matrix(DEFAULT_AGENT_PROFILES, self._agent_names)
        self._mode_head = self._profile_matrix(DEFAULT_MODE_PROFILES, self._mode_names)
        self._task_head = self._profile_matrix(DEFAULT_TASK_PROFILES, self._task_names)

    def _profile_matrix(self, profiles: Dict[str, List[str]], names: Iterable[str]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for name in names:
            vectors = [normalize_vec(self._hidden(f"user: {text}")) for text in profiles[name]]
            rows.append(normalize_vec(np.mean(np.stack(vectors, axis=0), axis=0)))
        return np.stack(rows, axis=0).astype(np.float32)

    def _hidden(self, text: str) -> np.ndarray:
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        torch = self._torch
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            base = getattr(self._model, "model", None)
            if base is not None:
                out = base(**inputs, use_cache=False)
                hidden = out.last_hidden_state
            else:
                out = self._model(**inputs, output_hidden_states=True, use_cache=False)
                hidden = out.hidden_states[-1]
        pos = self.hidden_position
        if hidden.shape[1] < abs(pos):
            pos = -1
        return hidden[0, pos, :].detach().float().cpu().numpy().astype(np.float32)

    def _resolve_path(self, value: Any) -> Path:
        p = Path(str(value))
        if p.is_absolute():
            return p
        return self.project_root / p

    def _resolve_optional_path(self, value: Any) -> Optional[Path]:
        if not value:
            return None
        return self._resolve_path(value)
