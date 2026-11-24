#!/usr/bin/env python3
"""PostToolUse hook: run fast static invariant checks after file edits and record status.

PostToolUse expects `{}` on stdout; findings are persisted per-worktree in
`<git-dir>/spark-hook-state.json` and surfaced immediately by `PreInvocation`
so the agent fixes violations on the very next step.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def main() -> int:
    payload = common.read_payload()
    cwd = common.resolve_cwd(payload)
    info = common.git_info(cwd)
    chk = common.check_script_path(info)

    if chk.is_file():
        r = subprocess.run(
            [sys.executable, str(chk), "--root", str(info["worktree_root"]), "--fast", "--json"],
            cwd=info["worktree_root"],
            capture_output=True,
            text=True,
        )
        try:
            data = json.loads(r.stdout)
        except Exception:
            data = {
                "ok": r.returncode == 0,
                "errors": [r.stderr.strip() or r.stdout.strip() or "check_invariants failed"],
            }
        data["updated_at"] = time.time()
        data["surfaced"] = False
        common.state_file(info).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    common.emit({})
    return 0


if __name__ == "__main__":
    sys.exit(main())
