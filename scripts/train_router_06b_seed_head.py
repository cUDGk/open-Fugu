from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fugu_router_06b import AGENTS, MODES, TASK_TYPES, ZeroSixBTower, normalize_vec  # noqa: E402


WORKER_EXAMPLES: Dict[str, List[str]] = {
    "gemma": [
        "Explain the tradeoffs of local MoE routing in two concise paragraphs.",
        "Rewrite this system note so it is shorter and clearer.",
        "Summarize these requirements into a practical plan.",
        "Give a calm product-level explanation for a technical audience.",
        "Compare two approaches and recommend the simpler one.",
        "Turn scattered notes into a coherent architecture overview.",
        "Answer this conceptual question without writing code.",
        "Draft a clear final response from multiple internal notes.",
        "Make this workflow easier to understand for a non-specialist.",
        "List the important constraints and assumptions in plain language.",
    ],
    "qwen": [
        "Write a Python function that clamps a requested token limit safely.",
        "Fix this traceback and explain the minimal code change.",
        "Add a FastAPI endpoint and a focused unit test.",
        "Debug why this PowerShell script fails on Windows.",
        "Inspect a repository and implement the requested change.",
        "Optimize this llama.cpp startup command for local inference.",
        "Write TypeScript code for a small API client.",
        "Find the bug in this async Python function.",
        "Create a script that benchmarks an OpenAI-compatible endpoint.",
        "Refactor this module without changing behavior.",
    ],
    "glm": [
        "Review this answer for contradictions and unsupported claims.",
        "Audit the proposed implementation for security risks.",
        "Verify whether the result satisfies every requirement.",
        "Act as a strict critic and identify missing edge cases.",
        "Check if this benchmark conclusion is justified by the data.",
        "Find factual mistakes and ask for sources where needed.",
        "Evaluate whether this deployment plan is safe enough.",
        "Decide if this fix is correct or if it needs another repair pass.",
        "Review the code for hidden failure modes.",
        "Confirm whether the final response should be accepted or rejected.",
    ],
}


MODE_EXAMPLES: Dict[str, List[str]] = {
    "fast": [
        "Answer in one sentence.",
        "Give a quick direct response.",
        "Return only OK.",
        "List three simple steps.",
        "Rewrite this sentence more concisely.",
        "What is the short version?",
        "Give the command only.",
        "Briefly explain the term.",
    ],
    "balanced": [
        "Implement this small feature and mention the verification step.",
        "Design a lightweight workflow for this local service.",
        "Explain the practical steps and include one code snippet.",
        "Make a normal-quality plan and execute the obvious part.",
        "Give a concrete answer with tradeoffs.",
        "Fix this small script and explain what changed.",
        "Produce a useful answer without a long multi-agent workflow.",
        "Choose the likely best local model and answer the request.",
    ],
    "deep": [
        "Review this code for bugs and provide a corrected version.",
        "Analyze this architecture carefully and identify failure modes.",
        "Verify the implementation, then repair it if needed.",
        "This is correctness-sensitive; do not guess.",
        "Inspect the repository and run the relevant checks.",
        "The prompt is long and may contain conflicting requirements.",
        "Find edge cases, test gaps, and practical risks.",
        "Do a careful multi-step engineering answer.",
    ],
    "ultra": [
        "Use the strongest full workflow available.",
        "Do a complete planner worker critic final verification pass.",
        "Maximize answer quality even if it takes longer.",
        "This is a high-value task requiring multiple specialist passes.",
        "Produce the most rigorous local multi-agent result.",
        "Use deep decomposition, critique, synthesis, and final checking.",
        "Treat this as an ultra mode request.",
        "Full analysis and final verification are required.",
    ],
}


TASK_EXAMPLES: Dict[str, List[str]] = {
    "general": [
        "Explain what local MoE routing is useful for.",
        "Summarize these notes.",
        "Rewrite this message to be clearer.",
        "Compare local models and cloud APIs.",
        "Give practical tradeoffs in plain language.",
        "Answer a conceptual question.",
    ],
    "coding": [
        "Write Python code for this helper function.",
        "Fix a failing pytest test.",
        "Debug a PowerShell command.",
        "Inspect the repository and patch the module.",
        "Add a FastAPI route.",
        "Benchmark the local llama.cpp endpoint.",
    ],
    "design": [
        "Design the orchestration workflow.",
        "Explain the system architecture.",
        "Choose the model routing policy.",
        "Plan a local deployment layout.",
        "Define the module boundaries.",
        "Sketch the data flow between services.",
    ],
    "verification": [
        "Review the answer for correctness.",
        "Audit the implementation for risks.",
        "Verify every requirement is satisfied.",
        "Check for contradictions.",
        "Find missing tests and edge cases.",
        "Decide if this should pass or be repaired.",
    ],
    "research": [
        "Find sources and compare the claims.",
        "Summarize the latest paper.",
        "Give citations for this technical mechanism.",
        "Research how this repository works.",
        "Look up current documentation.",
        "Compare benchmark evidence from multiple sources.",
    ],
}


def class_head(
    tower: ZeroSixBTower,
    examples: Dict[str, List[str]],
    names: Iterable[str],
    *,
    contrast: float,
) -> np.ndarray:
    class_vectors: Dict[str, np.ndarray] = {}
    for name in names:
        vectors = []
        for prompt in examples[name]:
            transcript = tower.format_transcript([{"role": "user", "content": prompt}])
            vectors.append(normalize_vec(tower._hidden(transcript)))
        class_vectors[name] = normalize_vec(np.mean(np.stack(vectors, axis=0), axis=0))

    rows = []
    all_names = list(names)
    for name in all_names:
        other = [class_vectors[n] for n in all_names if n != name]
        if other:
            row = class_vectors[name] - contrast * np.mean(np.stack(other, axis=0), axis=0)
        else:
            row = class_vectors[name]
        rows.append(normalize_vec(row))
    return np.stack(rows, axis=0).astype(np.float32)


def preview(head: np.ndarray, names: List[str], tower: ZeroSixBTower, prompt: str) -> str:
    h = normalize_vec(tower._hidden(tower.format_transcript([{"role": "user", "content": prompt}])))
    scores = head @ h
    return f"{prompt!r} -> {names[int(np.argmax(scores))]} {np.round(scores, 3).tolist()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "moe_fugu_minipc_local.yaml"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "router_06b_seed_head.npz"))
    parser.add_argument("--contrast", type=float, default=0.30)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("router_06b", {})["head_path"] = None
    tower = ZeroSixBTower.from_config(cfg, ROOT)
    if tower is None:
        raise SystemExit("router_06b is disabled")
    tower._ensure_loaded()

    agent_names = list(AGENTS)
    mode_names = list(MODES)
    task_names = list(TASK_TYPES)
    agent_head = class_head(tower, WORKER_EXAMPLES, agent_names, contrast=args.contrast)
    mode_head = class_head(tower, MODE_EXAMPLES, mode_names, contrast=args.contrast)
    task_head = class_head(tower, TASK_EXAMPLES, task_names, contrast=args.contrast)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        agent_names=np.array(agent_names),
        mode_names=np.array(mode_names),
        task_names=np.array(task_names),
        agent_head=agent_head,
        mode_head=mode_head,
        task_head=task_head,
        normalize=np.array([1], dtype=np.int32),
    )

    print(f"saved {output}")
    print(preview(agent_head, agent_names, tower, "Write a Python function and tests."))
    print(preview(agent_head, agent_names, tower, "Review this answer for correctness."))
    print(preview(agent_head, agent_names, tower, "Summarize this architecture."))
    print(preview(mode_head, mode_names, tower, "Answer in one sentence."))
    print(preview(mode_head, mode_names, tower, "Use the strongest full workflow available."))


if __name__ == "__main__":
    main()
