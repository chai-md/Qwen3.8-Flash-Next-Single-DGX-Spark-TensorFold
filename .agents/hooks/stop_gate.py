#!/usr/bin/env python3
"""Stop hook (Stage-1 'Proving Innocence' gate).

Prevents an agent from declaring a turn complete when tracked files in the
workspace have been modified and fail `tools/check_invariants.py` (static patch
linters + shell syntax + all 7 offline repo unit tests).

Respects `continuationActive` and `terminationReason` per Antigravity's Stop
hook contract so it never creates an infinite loop.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

WATCHED_PREFIXES = ("patches/", "scripts/", "tools/")
WATCHED_EXACT = {"start.sh", "stop.sh", "README.md"}


def has_dirty_watched_files(wt_root: Path) -> bool:
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=wt_root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return False
    for line in r.stdout.splitlines():
        if len(line) < 4:
            continue
        path_part = line[3:].split(" -> ")[-1].strip().replace("\\", "/")
        if path_part in WATCHED_EXACT or any(path_part.startswith(p) for p in WATCHED_PREFIXES):
            return True
    return False


def main() -> int:
    payload = common.read_payload()
    if payload.get("terminationReason") != "NO_TOOL_CALL":
        common.emit({"decision": "allow"})
        return 0
    if payload.get("continuationActive"):
        common.emit({"decision": "allow"})
        return 0

    cwd = common.resolve_cwd(payload)
    info = common.git_info(cwd)
    wt_root: Path = info["worktree_root"]

    if not has_dirty_watched_files(wt_root):
        common.emit({"decision": "allow"})
        return 0

    chk = common.check_script_path(info)
    if not chk.is_file():
        common.emit({"decision": "allow"})
        return 0

    r = subprocess.run(
        [sys.executable, str(chk), "--root", str(wt_root), "--json"],
        cwd=wt_root,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        common.emit({"decision": "allow"})
        return 0

    try:
        data = json.loads(r.stdout)
        errs = data.get("errors", [])
        detail = "\n".join(f"- {e}" for e in errs[:8])
    except Exception:
        detail = (r.stdout + "\n" + r.stderr).strip()

    common.emit({
        "decision": "continue",
        "reason": (
            "Stage-1 Stop Gate blocked completion because deterministic invariants failed:\n"
            f"{detail}\n"
            "Fix the failures and verify with `python3 tools/check_invariants.py` before stopping."
        ),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
