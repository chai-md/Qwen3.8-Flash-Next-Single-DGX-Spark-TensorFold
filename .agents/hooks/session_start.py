#!/usr/bin/env python3
"""SessionStart hook: inject concise DGX Spark reliability guardrails once per session."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def main() -> int:
    payload = common.read_payload()
    cwd = common.resolve_cwd(payload)
    info = common.git_info(cwd)
    role_note = (
        "This workspace is a secondary git worktree (author isolation): GPU serving/benchmarks are blocked here; "
        "run static/compile checks and hand off patches to bench-verifier."
        if info["is_secondary_worktree"]
        else "Main workspace: acquire the single-GPU lock (`tools/gpu_lock.py acquire --holder <role>`) before GPU runs."
    )
    msg = (
        "[DGX Spark Reliability Harness Active]\n"
        f"- {role_note}\n"
        "- Stage-1 (Stop gate): edits to patches/, scripts/, tools/, or README.md must pass `python3 tools/check_invariants.py`.\n"
        "- Stage-2 (git commit gate): committing patches/*.patch or scripts/config.sh requires a hash-bound GPU proof receipt via `python3 tools/receipt.py record ...` (exact.py + exact.py --pad 3000 + interleaved A/B + mem <= 112 GiB).\n"
        "- Protected verification files (`tools/exact.py`, `tools/check_invariants.py`, `.agents/hooks*`) require explicit human confirmation to modify."
    )
    common.emit({
        "injectSteps": [
            {
                "systemMessage": {
                    "systemMessage": msg,
                }
            }
        ]
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
