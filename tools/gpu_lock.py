#!/usr/bin/env python3
"""Host-wide single-GPU mutex shared across all git worktrees (/swarm workers).

On a single DGX Spark (GB10, 121 GiB unified memory), concurrent GPU containers
or kernel benchmarks collide on memory and poison timing measurements. This lock
lives under the main repository's `work/hooks/gpu.lock` (resolved via
`git rev-parse --git-common-dir`), so every `workspace: "share"` worktree sees
the same lease.

Usage:
  tools/gpu_lock.py acquire --holder bench-verifier [--reason "..."] [--ttl 7200]
  tools/gpu_lock.py release --holder bench-verifier [--force]
  tools/gpu_lock.py status
  tools/gpu_lock.py check --holder <id>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def resolve_lock_path(cwd: str | None = None) -> Path:
    env_override = os.environ.get("SPARK_GPU_LOCK_PATH")
    if env_override:
        p = Path(env_override)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    r = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        common = Path(r.stdout.strip())
        if not common.is_absolute():
            common = (Path(cwd or os.getcwd()) / common).resolve()
        repo_root = common.parent
    else:
        repo_root = Path(cwd or os.getcwd()).resolve()
    lock_dir = repo_root / "work" / "hooks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / "gpu.lock"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_lock(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def is_stale(data: dict, now: float | None = None) -> tuple[bool, str]:
    now = time.time() if now is None else now
    ttl = float(data.get("ttl", 7200))
    acquired = float(data.get("acquired_at", 0))
    if acquired > 0 and (now - acquired) > ttl:
        return True, f"expired (age {int(now - acquired)}s > ttl {int(ttl)}s)"
    pid = int(data.get("pid", 0) or 0)
    if pid > 0 and not pid_alive(pid):
        return True, f"owner pid {pid} is no longer running"
    return False, ""


def acquire(path: Path, holder: str, reason: str, ttl: int, pid: int) -> tuple[bool, str]:
    existing = read_lock(path)
    if existing is not None:
        stale, why = is_stale(existing)
        if stale:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        elif existing.get("holder") == holder:
            existing["acquired_at"] = time.time()
            existing["ttl"] = ttl
            existing["pid"] = pid
            if reason:
                existing["reason"] = reason
            path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            return True, f"renewed GPU lock for {holder}"
        else:
            return False, (
                f"GPU lock is held by {existing.get('holder')!r} "
                f"(pid={existing.get('pid')}, reason={existing.get('reason')!r})"
            )

    payload = {
        "holder": holder,
        "pid": pid,
        "cwd": os.getcwd(),
        "reason": reason,
        "acquired_at": time.time(),
        "ttl": ttl,
    }
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return True, f"acquired GPU lock for {holder} ({path})"
    except FileExistsError:
        cur = read_lock(path) or {}
        return False, f"GPU lock contention with {cur.get('holder')!r}"


def release(path: Path, holder: str, force: bool) -> tuple[bool, str]:
    existing = read_lock(path)
    if existing is None:
        return True, "GPU lock already free"
    if not force and existing.get("holder") != holder:
        return False, f"cannot release GPU lock held by {existing.get('holder')!r} as {holder!r}"
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return True, f"released GPU lock (was held by {existing.get('holder')!r})"


def check(path: Path, holder: str) -> tuple[bool, str]:
    existing = read_lock(path)
    if existing is None:
        return True, "free"
    stale, why = is_stale(existing)
    if stale:
        return True, f"stale ({why})"
    if holder and existing.get("holder") == holder:
        return True, f"held by self ({holder})"
    return False, (
        f"held by {existing.get('holder')!r} "
        f"(pid={existing.get('pid')}, reason={existing.get('reason')!r})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-GPU mutex for DGX Spark /swarm workflows")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_acq = sub.add_parser("acquire")
    p_acq.add_argument("--holder", required=True)
    p_acq.add_argument("--reason", default="gpu-benchmark")
    p_acq.add_argument("--ttl", type=int, default=7200)
    p_acq.add_argument("--pid", type=int, default=os.getppid())

    p_rel = sub.add_parser("release")
    p_rel.add_argument("--holder", default="")
    p_rel.add_argument("--force", action="store_true")

    sub.add_parser("status")

    p_chk = sub.add_parser("check")
    p_chk.add_argument("--holder", default="")

    args = ap.parse_args()
    path = resolve_lock_path()

    if args.cmd == "acquire":
        ok, msg = acquire(path, args.holder, args.reason, args.ttl, args.pid)
        print(msg)
        return 0 if ok else 1
    if args.cmd == "release":
        ok, msg = release(path, args.holder, args.force)
        print(msg)
        return 0 if ok else 1
    if args.cmd == "status":
        data = read_lock(path)
        if data is None:
            print(json.dumps({"locked": False, "path": str(path)}))
            return 0
        stale, why = is_stale(data)
        print(json.dumps({"locked": not stale, "stale": stale, "stale_reason": why, "path": str(path), **data}, indent=2))
        return 0
    if args.cmd == "check":
        ok, msg = check(path, args.holder)
        print(msg)
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
