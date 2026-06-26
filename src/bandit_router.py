from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

ACTIONS = [
    "gemma_direct",
    "qwen_direct",
    "qwen_direct__glm_if_risky",
    "gemma_direct__glm_if_risky",
    "gemma_spec__qwen_solution__glm_verify",
    "gemma_spec__qwen_solution__glm_critique__gemma_synthesis__glm_verify",
]


@dataclass
class BanditChoice:
    action: str
    score: float
    n: int
    mean_reward: float


class BanditRouter:
    """Small UCB router for later route correction.

    This is intentionally not integrated into the main conductor by default.
    Start deterministic, collect logs, then use this for route overrides.
    """

    def __init__(self, path: str = "logs/moe_fugu_bandit.sqlite3") -> None:
        self.db = sqlite3.connect(path)
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                bucket TEXT NOT NULL,
                action TEXT NOT NULL,
                reward REAL NOT NULL,
                meta_json TEXT NOT NULL
            )
            """
        )
        self.db.commit()

    @staticmethod
    def bucket(task_type: str, risk: Iterable[str], prompt_chars: int) -> str:
        risk_key = ",".join(sorted(set(risk))) or "none"
        if prompt_chars > 40000:
            length = "long"
        elif prompt_chars > 12000:
            length = "medium"
        else:
            length = "short"
        return f"{task_type}|{risk_key}|{length}"

    def choose(self, bucket: str, allowed_actions: Iterable[str], exploration: float = 1.2) -> BanditChoice:
        allowed: List[str] = [a for a in allowed_actions if a in ACTIONS]
        if not allowed:
            allowed = list(ACTIONS)

        total_n = self.db.execute(
            "SELECT COUNT(*) FROM outcomes WHERE bucket = ?",
            (bucket,),
        ).fetchone()[0]

        best: BanditChoice | None = None
        for action in allowed:
            row = self.db.execute(
                "SELECT COUNT(*), AVG(reward) FROM outcomes WHERE bucket = ? AND action = ?",
                (bucket, action),
            ).fetchone()
            n = int(row[0] or 0)
            mean = float(row[1] if row[1] is not None else 0.0)
            if n == 0:
                score = 999.0
            else:
                score = mean + exploration * math.sqrt(math.log(total_n + 1) / n)
            choice = BanditChoice(action=action, score=score, n=n, mean_reward=mean)
            if best is None or choice.score > best.score:
                best = choice

        assert best is not None
        return best

    def record(self, bucket: str, action: str, reward: float, meta: Dict) -> None:
        self.db.execute(
            "INSERT INTO outcomes (ts, bucket, action, reward, meta_json) VALUES (?, ?, ?, ?, ?)",
            (
                int(time.time()),
                bucket,
                action,
                float(reward),
                json.dumps(meta, ensure_ascii=False),
            ),
        )
        self.db.commit()


def reward_from_run(
    *,
    verifier_passed: bool,
    repaired: bool,
    major_or_fatal: bool,
    timeout: bool,
    elapsed_ms: int,
    fanout_count: int,
) -> float:
    normalized_latency = min(elapsed_ms / 180_000, 2.0)
    return (
        1.0 * float(verifier_passed)
        - 0.35 * float(repaired)
        - 0.40 * float(major_or_fatal)
        - 0.15 * float(timeout)
        - 0.10 * normalized_latency
        - 0.05 * max(fanout_count - 1, 0)
    )
