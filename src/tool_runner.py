from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ToolResult:
    ok: bool
    command: str
    stdout: str
    stderr: str
    returncode: int


ALLOWED_PREFIXES = [
    ["python", "-m", "pytest"],
    ["pytest"],
    ["python", "-m", "ruff", "check"],
    ["ruff", "check"],
    ["python", "-m", "unittest"],
    ["npm", "test"],
    ["npm", "run", "test"],
    ["npm", "run", "typecheck"],
    ["npx", "tsc", "--noEmit"],
]


def allowed(cmd: List[str]) -> bool:
    return any(cmd[: len(prefix)] == prefix for prefix in ALLOWED_PREFIXES)


async def run_tool(cmd: List[str], cwd: str | Path, timeout_s: int = 120) -> ToolResult:
    if not allowed(cmd):
        return ToolResult(
            ok=False,
            command=" ".join(cmd),
            stdout="",
            stderr="Command is not allowed by local policy.",
            returncode=126,
        )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        return ToolResult(
            ok=False,
            command=" ".join(cmd),
            stdout="",
            stderr=f"Timed out after {timeout_s}s",
            returncode=124,
        )

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")

    return ToolResult(
        ok=proc.returncode == 0,
        command=" ".join(cmd),
        stdout=stdout[-12000:],
        stderr=stderr[-12000:],
        returncode=int(proc.returncode or 0),
    )
