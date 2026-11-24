"""Shared helpers for Antigravity lifecycle hooks in .agents/hooks/."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def read_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def resolve_cwd(payload: dict) -> Path:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip() and Path(cwd).exists():
        return Path(cwd).resolve()
    ws = payload.get("workspacePaths")
    if isinstance(ws, list) and ws and isinstance(ws[0], str) and Path(ws[0]).exists():
        return Path(ws[0]).resolve()
    # Fallback: parent of .agents/hooks/ -> repo root
    return Path(__file__).resolve().parent.parent.parent


def git_info(cwd: Path) -> dict:
    """Return worktree_root, main_root, git_dir, and is_secondary_worktree."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True)
    wt_root = Path(top.stdout.strip()).resolve() if top.returncode == 0 and top.stdout.strip() else cwd

    gd = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=wt_root, capture_output=True, text=True)
    if gd.returncode == 0 and gd.stdout.strip():
        git_dir = Path(gd.stdout.strip())
        if not git_dir.is_absolute():
            git_dir = (wt_root / git_dir).resolve()
    else:
        git_dir = wt_root / ".git"

    cd = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=wt_root, capture_output=True, text=True)
    if cd.returncode == 0 and cd.stdout.strip():
        common_dir = Path(cd.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (wt_root / common_dir).resolve()
        main_root = common_dir.parent
    else:
        common_dir = git_dir
        main_root = wt_root

    dot_git = wt_root / ".git"
    is_secondary_wt = dot_git.is_file()
    return {
        "worktree_root": wt_root,
        "main_root": main_root,
        "git_dir": git_dir,
        "common_dir": common_dir,
        "is_secondary_worktree": is_secondary_wt,
    }


def state_file(info: dict) -> Path:
    gdir: Path = info["git_dir"]
    gdir.mkdir(parents=True, exist_ok=True)
    return gdir / "spark-hook-state.json"


def check_script_path(info: dict) -> Path:
    for base in (info["worktree_root"], info["main_root"]):
        cand = base / "tools" / "check_invariants.py"
        if cand.is_file():
            return cand
    return Path(__file__).resolve().parent.parent.parent / "tools" / "check_invariants.py"


def receipt_script_path(info: dict) -> Path:
    for base in (info["worktree_root"], info["main_root"]):
        cand = base / "tools" / "receipt.py"
        if cand.is_file():
            return cand
    return Path(__file__).resolve().parent.parent.parent / "tools" / "receipt.py"


def gpu_lock_script_path(info: dict) -> Path:
    for base in (info["worktree_root"], info["main_root"]):
        cand = base / "tools" / "gpu_lock.py"
        if cand.is_file():
            return cand
    return Path(__file__).resolve().parent.parent.parent / "tools" / "gpu_lock.py"
