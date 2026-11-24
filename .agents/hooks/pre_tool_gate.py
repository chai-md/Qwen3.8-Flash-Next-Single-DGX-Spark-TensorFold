#!/usr/bin/env python3
"""PreToolUse hook: hard deterministic guardrails before edits or shell commands.

Enforces:
  1. Protected verification / hook files -> `force_ask` (prevents silent test gaming).
  2. Forbidden out-of-workspace paths (e.g. /home/nobu, /etc, ~/.ssh) -> `deny`.
  3. Dangerous shell commands (force push, memory-reserve bypass, rm -rf /) -> `deny`.
  4. Single-GPU isolation for /swarm:
     - Secondary git worktrees (author-* workers) cannot run GPU containers/profilers -> `deny`.
     - If `work/hooks/gpu.lock` is held by another live holder -> `deny`.
  5. Stage-2 `git commit` gate:
     - Runs `tools/check_invariants.py` before any commit -> `deny` on failure.
     - If `patches/*.patch` or `scripts/config.sh` are part of the commit, verifies a
       matching hash-bound GPU proof receipt (`tools/receipt.py verify`) -> `deny` if missing/stale.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

PROTECTED_REL_PATHS = {
    ".agents/hooks.json",
    "tools/check_invariants.py",
    "tools/receipt.py",
    "tools/gpu_lock.py",
    "tools/exact.py",
    "tools/toolcheck.py",
    "tools/visioncheck.py",
    "tools/prompt_reuse.py",
    "tools/context_boundary.py",
    "tools/needle.py",
    "tools/test_hooks_harness.py",
}

FORBIDDEN_PATH_PREFIXES = (
    "/home/nobu",
    "/etc/",
    "/root/",
    "/var/",
    "/usr/",
)

DANGEROUS_CMD_PATTERNS = [
    (re.compile(r"\brm\s+-[^\n]*r[^\n]*\s+/(?:\s|$|\*)"), "destructive `rm -rf /` is blocked"),
    (re.compile(r"\bgit\s+push\b[^\n]*(\s--force\b|\s-f\b|\s--force-with-lease\b)"), "force-push is blocked by safety gate"),
    (re.compile(r"/home/nobu\b"), "accessing /home/nobu is forbidden by workspace boundary"),
    (re.compile(r"\bMEM_CHECK\s*=\s*0\b"), "disabling MEM_CHECK=0 is forbidden on DGX Spark unified memory"),
    (re.compile(r"\b(?:TENSORFOLD_MEMORY_RESERVE_GIB|MEM_FLOOR_GIB)\s*=\s*[0-8](?:\s|$|;)"), "lowering memory reserve below 9 GiB risks freezing DGX Spark"),
    (re.compile(r"\bdocker\s+system\s+prune\b[^\n]*-a"), "`docker system prune -a` is blocked"),
]

GPU_CMD_RE = re.compile(
    r"(\bdocker\s+run\b[^\n]*--gpus\b|\btensorfold-native\s+serve\b|(?:\./|bash\s+)start\.sh\b|\bnsys\s+profile\b|\bncu\b)"
)
GIT_COMMIT_RE = re.compile(r"(?:^|[;&|]\s*)git(?:\s+-c\s+\S+)*\s+commit\b")
GIT_ADD_RE = re.compile(r"(?:^|[;&|]\s*)git\s+add\b")


def is_runtime_file(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return rel == "scripts/config.sh" or (rel.startswith("patches/") and rel.endswith(".patch"))


def handle_file_tool(tool_name: str, args: dict, info: dict) -> dict:
    raw_path = args.get("Path") or args.get("path") or ""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {"decision": "allow"}

    p = Path(raw_path)
    resolved = p.resolve() if p.is_absolute() else (info["worktree_root"] / p).resolve()
    s = str(resolved)

    for pref in FORBIDDEN_PATH_PREFIXES:
        if s == pref.rstrip("/") or s.startswith(pref):
            return {"decision": "deny", "reason": f"Writing to forbidden path `{s}` is blocked."}

    home_ssh = str(Path.home() / ".ssh")
    if s == home_ssh or s.startswith(home_ssh + "/"):
        return {"decision": "deny", "reason": "Modifying ~/.ssh is blocked."}

    wt_root: Path = info["worktree_root"]
    try:
        rel = str(resolved.relative_to(wt_root)).replace("\\", "/")
    except ValueError:
        rel = ""

    if rel in PROTECTED_REL_PATHS or rel.startswith(".agents/hooks/"):
        return {
            "decision": "force_ask",
            "reason": (
                f"Protected verification/harness file `{rel}`: explicit human confirmation is required "
                "so autonomous agents cannot weaken invariants or tests."
            ),
        }

    return {"decision": "allow"}


def check_gpu_guard(cmd: str, payload: dict, info: dict) -> dict | None:
    if not GPU_CMD_RE.search(cmd):
        return None
    if re.search(r"\bDRY_RUN\s*=\s*1\b", cmd):
        return None

    if info["is_secondary_worktree"]:
        return {
            "decision": "deny",
            "reason": (
                "GPU serving/profiling is blocked in secondary git worktrees (`/swarm` author isolation). "
                "Run static/compile checks here (`tools/check_invariants.py`, `zig build check-native-cuda-host`, "
                "`nvcc --ptxas-options=-v`) and hand off GPU verification to the singleton `bench-verifier`."
            ),
        }

    # Check single-GPU lock ownership
    lock_script = common.gpu_lock_script_path(info)
    if lock_script.is_file():
        conv_id = str(payload.get("conversationId") or "")
        holder = os.environ.get("SPARK_SWARM_ROLE") or conv_id
        r = subprocess.run(
            [sys.executable, str(lock_script), "check", "--holder", holder],
            cwd=info["worktree_root"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return {
                "decision": "deny",
                "reason": (
                    f"Single-GPU mutex conflict ({r.stdout.strip()}). Wait for the current GPU holder "
                    "to finish or coordinate via `/swarm` `bench-verifier`."
                ),
            }
    return None


def staged_or_dirty_runtime_files(wt_root: Path, include_working_tree: bool) -> list[str]:
    files: set[str] = set()
    r = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=wt_root,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if line.strip() and is_runtime_file(line.strip()):
                files.add(line.strip())

    if include_working_tree:
        r2 = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=wt_root,
            capture_output=True,
            text=True,
        )
        if r2.returncode == 0:
            for line in r2.stdout.splitlines():
                if len(line) >= 4:
                    path_part = line[3:].split(" -> ")[-1].strip()
                    if is_runtime_file(path_part):
                        files.add(path_part)
    return sorted(files)


def check_commit_gate(cmd: str, info: dict) -> dict | None:
    if not GIT_COMMIT_RE.search(cmd):
        return None
    if re.search(r"\s(--help|-h|--dry-run)\b", cmd):
        return None

    wt_root: Path = info["worktree_root"]
    chk = common.check_script_path(info)
    if chk.is_file():
        r = subprocess.run(
            [sys.executable, str(chk), "--root", str(wt_root), "--json"],
            cwd=wt_root,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            summary = r.stdout.strip()
            try:
                data = json.loads(summary)
                errs = data.get("errors", [])
                summary = "; ".join(errs[:6])
            except Exception:
                pass
            return {
                "decision": "deny",
                "reason": f"Stage-1 invariant gate blocked `git commit`: {summary}",
            }

    chained_add = bool(GIT_ADD_RE.search(cmd))
    runtime_files = staged_or_dirty_runtime_files(wt_root, include_working_tree=chained_add)
    if not runtime_files:
        return None

    rec = common.receipt_script_path(info)
    if not rec.is_file():
        return None

    verify_cmd = [sys.executable, str(rec), "verify", "--json"]
    if not chained_add:
        verify_cmd.append("--staged")

    r2 = subprocess.run(verify_cmd, cwd=wt_root, capture_output=True, text=True)
    if r2.returncode != 0:
        fp = ""
        probs: list[str] = []
        try:
            data = json.loads(r2.stdout)
            fp = str(data.get("fingerprint", ""))[:16]
            probs = data.get("problems", [])
        except Exception:
            probs = [r2.stdout.strip() or r2.stderr.strip()]
        return {
            "decision": "deny",
            "reason": (
                f"Stage-2 GPU proof receipt gate blocked `git commit` touching {runtime_files} "
                f"(fingerprint={fp}): {'; '.join(probs)}. "
                "Run GPU verification (exact.py, exact.py --pad 3000, interleaved A/B, mem <= 112 GiB) "
                "and record a receipt with `python3 tools/receipt.py record --exact-pass --exact-pad3000-pass "
                "--ab-pass --mem-peak-gib <GIB> --prompt-reuse-pct <PCT> --summary \"...\"`."
            ),
        }
    return None


def handle_run_command(args: dict, payload: dict, info: dict) -> dict:
    cmd = args.get("CommandLine") or args.get("commandLine") or ""
    if not isinstance(cmd, str) or not cmd.strip():
        return {"decision": "allow"}

    for pat, reason in DANGEROUS_CMD_PATTERNS:
        if pat.search(cmd):
            return {"decision": "deny", "reason": reason}

    gpu_block = check_gpu_guard(cmd, payload, info)
    if gpu_block is not None:
        return gpu_block

    commit_block = check_commit_gate(cmd, info)
    if commit_block is not None:
        return commit_block

    return {"decision": "allow"}


def main() -> int:
    payload = common.read_payload()
    cwd = common.resolve_cwd(payload)
    info = common.git_info(cwd)
    tc = payload.get("toolCall") or {}
    name = str(tc.get("name") or "")
    args = tc.get("args") if isinstance(tc.get("args"), dict) else {}

    if name in ("edit_file", "write_file"):
        common.emit(handle_file_tool(name, args, info))
    elif name == "run_command":
        common.emit(handle_run_command(args, payload, info))
    else:
        common.emit({"decision": "allow"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
