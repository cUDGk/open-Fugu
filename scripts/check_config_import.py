from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MOE_FUGU_CONFIG", str(PROJECT_ROOT / "configs" / "moe_fugu_minipc_local.yaml"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.conductor_7940hs import LocalMoEFugu


def main() -> None:
    fugu = LocalMoEFugu()
    route = fugu.route([{"role": "user", "content": "Pythonの実装をレビューして"}])
    print(fugu.front_model)
    print({name: cfg.model_id for name, cfg in fugu.models.items()})
    print(route.model_dump())


if __name__ == "__main__":
    main()
