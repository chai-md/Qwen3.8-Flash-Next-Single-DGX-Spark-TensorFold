#!/usr/bin/env python3
"""Offline self-test for the DGX Spark reliability harness (.agents/hooks + tools/).

Tests every layer of Lauren Tan's constraint pyramid deterministically using
isolated temporary git repos and git worktrees (no Docker, no GPU, no network):
  1. `.agents/hooks.json` schema & command paths valid.
  2. `tools/check_invariants.py` passes on current repo HEAD and rejects synthetic
     violations (unregistered TF_FLASHNEXT_* var, unguarded cudaDeviceSynchronize,
     CUDA printf, Zig std.debug.print, TODO/HACK slop, broken patch numbering,
     CHUNK_SOLO=12288, MEMORY_RESERVE_GIB=6).
  3. `tools/gpu_lock.py`: acquire, conflict rejection, same-holder renew, stale-PID
     auto-reap, release.
  4. `tools/receipt.py`: fingerprint, record, verify, tamper/drift invalidation,
     refusal when mem_peak_gib > 112 or gates missing.
  5. `.agents/hooks/session_start.py`: valid `injectSteps` camelCase output.
  6. `.agents/hooks/pre_tool_gate.py`:
     - `force_ask` on protected verification/hook files (`tools/exact.py`, `.agents/hooks.json`)
     - `deny` on forbidden paths (`/home/nobu/...`) and dangerous commands (`git push --force`, `MEM_CHECK=0`)
     - `deny` on GPU commands inside a secondary git worktree (`/swarm` author isolation)
     - `deny` on GPU commands when `gpu.lock` is held by another live holder
     - `deny` on `git commit` when `patches/*.patch` is staged without a matching GPU receipt
     - `allow` on `git commit` once a matching receipt is recorded
     - `allow` on docs-only `git commit` without a GPU receipt
  7. `.agents/hooks/post_edit_check.py` + `pre_invocation_inject.py`:
     persists static failure and surfaces it once on the next invocation.
  8. `.agents/hooks/stop_gate.py`:
     blocks premature stop (`decision: continue`) when dirty tracked files fail
     invariants, respects `continuationActive: true`, and allows stop once clean.

Usage:
  python3 tools/test_hooks_harness.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / ".agents"
HOOKS = AGENTS / "hooks"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        failures.append(name)


def run_hook(script: str, payload: dict, env_extra: dict | None = None) -> dict:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [sys.executable, str(HOOKS / script)],
        cwd=AGENTS,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        raise AssertionError(f"{script} exited {r.returncode}: {r.stderr}")
    return json.loads(r.stdout)


def make_mini_repo(base: Path) -> Path:
    repo = base / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)

    (repo / "patches").mkdir()
    (repo / "scripts").mkdir()
    (repo / "tools").mkdir()
    shutil.copy2(ROOT / "scripts" / "config.sh", repo / "scripts" / "config.sh")
    shutil.copy2(ROOT / "tools" / "check_invariants.py", repo / "tools" / "check_invariants.py")
    shutil.copy2(ROOT / "tools" / "receipt.py", repo / "tools" / "receipt.py")
    shutil.copy2(ROOT / "tools" / "gpu_lock.py", repo / "tools" / "gpu_lock.py")
    (repo / "start.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (repo / "stop.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (repo / "README.md").write_text("# Test\n", encoding="utf-8")
    (repo / "patches" / "0001-clean.patch").write_text(
        "diff --git a/zig/src/platform/environment.zig b/zig/src/platform/environment.zig\n"
        "--- a/zig/src/platform/environment.zig\n"
        "+++ b/zig/src/platform/environment.zig\n"
        "@@ -1,1 +1,2 @@\n"
        " const x = 1;\n"
        "+    .{ .name = \"TF_FLASHNEXT_MTP_CACHE_ONLY\", .scope = .flashnext, .kind = .flag, .forms = \"0 or 1\" },\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


# 1. hooks.json schema
hj = json.loads((AGENTS / "hooks.json").read_text(encoding="utf-8"))
guard = hj.get("spark-reliability-guard", {})
check(
    "hooks.json defines all 5 lifecycle events with valid handler scripts",
    all(k in guard for k in ("SessionStart", "PreToolUse", "PostToolUse", "PreInvocation", "Stop"))
    and (HOOKS / "session_start.py").is_file()
    and (HOOKS / "pre_tool_gate.py").is_file()
    and (HOOKS / "post_edit_check.py").is_file()
    and (HOOKS / "pre_invocation_inject.py").is_file()
    and (HOOKS / "stop_gate.py").is_file(),
    f"keys={list(guard.keys())}",
)

# 2. check_invariants passes on real repo HEAD
r_real = subprocess.run([sys.executable, str(ROOT / "tools" / "check_invariants.py"), "--json"], cwd=ROOT, capture_output=True, text=True)
real_data = json.loads(r_real.stdout)
check("check_invariants.py passes on real repository", r_real.returncode == 0 and real_data["ok"], str(real_data.get("errors")))

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    repo = make_mini_repo(tmp)

    # Synthetic invariant violation tests
    bad_cases = [
        (
            "unregistered TF_FLASHNEXT_* var",
            "patches/0002-bad-env.patch",
            "diff --git a/zig/src/foo.zig b/zig/src/foo.zig\n--- a/zig/src/foo.zig\n+++ b/zig/src/foo.zig\n@@ -1 +1 @@\n+const v = std.c.getenv(\"TF_FLASHNEXT_UNREGISTERED_XYZ\");\n",
            "not registered",
        ),
        (
            "unguarded cudaDeviceSynchronize",
            "patches/0002-bad-sync.patch",
            "diff --git a/zig/kernels/cuda/k.cu b/zig/kernels/cuda/k.cu\n--- a/zig/kernels/cuda/k.cu\n+++ b/zig/kernels/cuda/k.cu\n@@ -1 +1 @@\n+cudaDeviceSynchronize();\n",
            "graph capture",
        ),
        (
            "CUDA printf in kernel",
            "patches/0002-bad-printf.patch",
            "diff --git a/zig/kernels/cuda/k.cu b/zig/kernels/cuda/k.cu\n--- a/zig/kernels/cuda/k.cu\n+++ b/zig/kernels/cuda/k.cu\n@@ -1 +1 @@\n+printf(\"dbg %d\\n\", 1);\n",
            "printf(",
        ),
        (
            "Zig std.debug.print in production",
            "patches/0002-bad-dbg.patch",
            "diff --git a/zig/src/k.zig b/zig/src/k.zig\n--- a/zig/src/k.zig\n+++ b/zig/src/k.zig\n@@ -1 +1 @@\n+std.debug.print(\"hi\\n\", .{});\n",
            "std.debug.print",
        ),
        (
            "TODO/HACK slop marker",
            "patches/0002-bad-slop.patch",
            "diff --git a/zig/src/k.zig b/zig/src/k.zig\n--- a/zig/src/k.zig\n+++ b/zig/src/k.zig\n@@ -1 +1 @@\n+// HACK: skip check\n",
            "HACK",
        ),
        (
            "broken patch numbering gap",
            "patches/0003-gap.patch",
            "diff --git a/zig/src/k.zig b/zig/src/k.zig\n--- a/zig/src/k.zig\n+++ b/zig/src/k.zig\n@@ -1 +1 @@\n+const z = 3;\n",
            "unbroken",
        ),
    ]
    for label, rel, content, needle in bad_cases:
        p = repo / rel
        p.write_text(content, encoding="utf-8")
        rr = subprocess.run([sys.executable, str(repo / "tools" / "check_invariants.py"), "--root", str(repo), "--fast", "--json"], capture_output=True, text=True)
        d = json.loads(rr.stdout)
        joined = " | ".join(d.get("errors", []))
        check(f"invariant catches {label}", rr.returncode != 0 and needle in joined, joined)
        p.unlink()

    # Config violation: CHUNK_SOLO=12288
    cfg = repo / "scripts" / "config.sh"
    orig_cfg = cfg.read_text(encoding="utf-8")
    cfg.write_text(orig_cfg.replace("TF_FLASHNEXT_CHUNK_SOLO:-8192", "TF_FLASHNEXT_CHUNK_SOLO:-12288"), encoding="utf-8")
    rr = subprocess.run([sys.executable, str(repo / "tools" / "check_invariants.py"), "--root", str(repo), "--fast", "--json"], capture_output=True, text=True)
    d = json.loads(rr.stdout)
    check("invariant catches CHUNK_SOLO=12288 prompt-cache hazard", rr.returncode != 0 and "CHUNK_SOLO" in " ".join(d["errors"]), str(d["errors"]))
    cfg.write_text(orig_cfg, encoding="utf-8")

    # 3. GPU lock tests
    lock_file = tmp / "test-gpu.lock"
    env_lock = {**os.environ, "SPARK_GPU_LOCK_PATH": str(lock_file)}
    r1 = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "acquire", "--holder", "bench-verifier", "--pid", str(os.getpid())], env=env_lock, capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "acquire", "--holder", "author-1", "--pid", str(os.getpid())], env=env_lock, capture_output=True, text=True)
    r3 = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "check", "--holder", "author-1"], env=env_lock, capture_output=True, text=True)
    r4 = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "check", "--holder", "bench-verifier"], env=env_lock, capture_output=True, text=True)
    check(
        "gpu_lock enforces mutual exclusion across holders",
        r1.returncode == 0 and r2.returncode != 0 and r3.returncode != 0 and r4.returncode == 0,
        f"r1={r1.returncode} r2={r2.returncode} r3={r3.returncode} r4={r4.returncode}",
    )
    # Stale PID reap
    lock_file.write_text(json.dumps({"holder": "dead-worker", "pid": 99999999, "acquired_at": 1.0, "ttl": 7200}), encoding="utf-8")
    r5 = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "acquire", "--holder", "bench-verifier", "--pid", str(os.getpid())], env=env_lock, capture_output=True, text=True)
    r6 = subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "release", "--holder", "bench-verifier"], env=env_lock, capture_output=True, text=True)
    check("gpu_lock auto-reaps stale PID and releases cleanly", r5.returncode == 0 and r6.returncode == 0, f"r5={r5.stdout} r6={r6.stdout}")

    # 4. Receipt tests
    rdir = tmp / "receipts"
    env_rec = {**os.environ, "SPARK_RECEIPTS_DIR": str(rdir)}
    r_bad_mem = subprocess.run(
        [sys.executable, str(repo / "tools" / "receipt.py"), "record", "--exact-pass", "--exact-pad3000-pass", "--ab-pass", "--mem-peak-gib", "115.0", "--summary", "over cap"],
        cwd=repo, env=env_rec, capture_output=True, text=True,
    )
    check("receipt.py refuses mem_peak_gib > 112 GiB", r_bad_mem.returncode != 0, r_bad_mem.stderr)

    r_ok = subprocess.run(
        [sys.executable, str(repo / "tools" / "receipt.py"), "record", "--exact-pass", "--exact-pad3000-pass", "--ab-pass", "--mem-peak-gib", "85.5", "--prompt-reuse-pct", "99.3", "--summary", "verified"],
        cwd=repo, env=env_rec, capture_output=True, text=True,
    )
    r_ver = subprocess.run([sys.executable, str(repo / "tools" / "receipt.py"), "verify"], cwd=repo, env=env_rec, capture_output=True, text=True)
    check("receipt.py records and verifies matching fingerprint", r_ok.returncode == 0 and r_ver.returncode == 0, r_ok.stderr + r_ver.stdout)

    # Tamper runtime file -> verify must fail
    (repo / "patches" / "0001-clean.patch").write_text(
        (repo / "patches" / "0001-clean.patch").read_text(encoding="utf-8") + "+const z = 3;\n", encoding="utf-8"
    )
    r_stale = subprocess.run([sys.executable, str(repo / "tools" / "receipt.py"), "verify"], cwd=repo, env=env_rec, capture_output=True, text=True)
    check("receipt.py invalidates immediately after any patch edit", r_stale.returncode != 0, r_stale.stdout)
    subprocess.run(["git", "checkout", "--", "patches/0001-clean.patch"], cwd=repo, check=True)

    # 5. SessionStart hook
    out_ss = run_hook("session_start.py", {"cwd": str(repo), "hookEventName": "SessionStart"})
    check("SessionStart hook emits injectSteps systemMessage", "injectSteps" in out_ss and "DGX Spark" in out_ss["injectSteps"][0]["systemMessage"]["systemMessage"], str(out_ss))

    # 6. PreToolUse hook tests
    out_prot = run_hook("pre_tool_gate.py", {
        "cwd": str(repo),
        "toolCall": {"name": "edit_file", "args": {"Path": str(repo / "tools" / "exact.py")}},
    })
    check("PreToolUse requires force_ask on protected tools/exact.py", out_prot.get("decision") == "force_ask", str(out_prot))

    out_forb = run_hook("pre_tool_gate.py", {
        "cwd": str(repo),
        "toolCall": {"name": "write_file", "args": {"Path": "/home/nobu/evil.txt"}},
    })
    check("PreToolUse denies writes to /home/nobu", out_forb.get("decision") == "deny", str(out_forb))

    out_fpush = run_hook("pre_tool_gate.py", {
        "cwd": str(repo),
        "toolCall": {"name": "run_command", "args": {"CommandLine": "git push --force origin optimize"}},
    })
    check("PreToolUse denies git push --force", out_fpush.get("decision") == "deny", str(out_fpush))

    # Secondary git worktree (/swarm author isolation)
    wt = tmp / "author-wt"
    subprocess.run(["git", "worktree", "add", "-q", "-b", "author-1", str(wt)], cwd=repo, check=True)
    out_wt_gpu = run_hook("pre_tool_gate.py", {
        "cwd": str(wt),
        "toolCall": {"name": "run_command", "args": {"CommandLine": "./start.sh"}},
    })
    check("PreToolUse denies GPU serving in secondary git worktree (/swarm author)", out_wt_gpu.get("decision") == "deny", str(out_wt_gpu))

    out_wt_dry = run_hook("pre_tool_gate.py", {
        "cwd": str(wt),
        "toolCall": {"name": "run_command", "args": {"CommandLine": "DRY_RUN=1 ./start.sh"}},
    })
    check("PreToolUse allows DRY_RUN=1 ./start.sh in secondary worktree", out_wt_dry.get("decision") == "allow", str(out_wt_dry))

    # GPU mutex block in main workspace when held by another conversation
    subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "acquire", "--holder", "other-worker", "--pid", str(os.getpid())], env=env_lock, check=True)
    out_mutex = run_hook("pre_tool_gate.py", {
        "cwd": str(repo),
        "conversationId": "my-conv",
        "toolCall": {"name": "run_command", "args": {"CommandLine": "docker run --gpus all tensorfold-qwen38"}},
    }, env_extra={"SPARK_GPU_LOCK_PATH": str(lock_file)})
    check("PreToolUse blocks GPU run when gpu.lock held by another worker", out_mutex.get("decision") == "deny", str(out_mutex))
    subprocess.run([sys.executable, str(ROOT / "tools" / "gpu_lock.py"), "release", "--holder", "other-worker"], env=env_lock, check=True)

    # Stage-2 git commit gate: docs-only commit allowed without GPU receipt
    (repo / "README.md").write_text("# Updated docs\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    out_doc_commit = run_hook("pre_tool_gate.py", {
        "cwd": str(repo),
        "toolCall": {"name": "run_command", "args": {"CommandLine": 'git commit -m "docs"'}},
    }, env_extra={"SPARK_RECEIPTS_DIR": str(rdir)})
    check("PreToolUse allows docs-only git commit without GPU receipt", out_doc_commit.get("decision") == "allow", str(out_doc_commit))
    subprocess.run(["git", "reset", "-q", "HEAD", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "--", "README.md"], cwd=repo, check=True)

    # Stage-2 git commit gate: patch staged without receipt -> DENY; with receipt -> ALLOW
    (repo / "patches" / "0002-good.patch").write_text(
        "diff --git a/zig/src/b.zig b/zig/src/b.zig\n--- a/zig/src/b.zig\n+++ b/zig/src/b.zig\n@@ -1 +1 @@\n+const ok = true;\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "patches/0002-good.patch"], cwd=repo, check=True)
    out_no_rec = run_hook("pre_tool_gate.py", {
        "cwd": str(repo),
        "toolCall": {"name": "run_command", "args": {"CommandLine": 'git commit -m "new patch"'}},
    }, env_extra={"SPARK_RECEIPTS_DIR": str(rdir)})
    check("PreToolUse denies git commit of patch without GPU receipt", out_no_rec.get("decision") == "deny", str(out_no_rec))

    subprocess.run(
        [sys.executable, str(repo / "tools" / "receipt.py"), "record", "--staged", "--exact-pass", "--exact-pad3000-pass", "--ab-pass", "--mem-peak-gib", "86.0", "--prompt-reuse-pct", "99.1", "--summary", "good patch"],
        cwd=repo, env=env_rec, check=True,
    )
    out_with_rec = run_hook("pre_tool_gate.py", {
        "cwd": str(repo),
        "toolCall": {"name": "run_command", "args": {"CommandLine": 'git commit -m "new patch"'}},
    }, env_extra={"SPARK_RECEIPTS_DIR": str(rdir)})
    check("PreToolUse allows git commit of patch once valid GPU receipt exists", out_with_rec.get("decision") == "allow", str(out_with_rec))

    # 7 & 8. PostToolUse -> PreInvocation feedback + Stop gate
    (repo / "patches" / "0002-good.patch").write_text(
        "diff --git a/zig/src/b.zig b/zig/src/b.zig\n--- a/zig/src/b.zig\n+++ b/zig/src/b.zig\n@@ -1 +1 @@\n+// TODO: unfinished\n",
        encoding="utf-8",
    )
    out_post = run_hook("post_edit_check.py", {"cwd": str(repo), "toolCall": {"name": "edit_file", "args": {}}})
    out_pre_inv = run_hook("pre_invocation_inject.py", {"cwd": str(repo)})
    out_pre_inv2 = run_hook("pre_invocation_inject.py", {"cwd": str(repo)})
    check(
        "PostToolUse records failure and PreInvocation injects it once",
        out_post == {} and "injectSteps" in out_pre_inv and "TODO" in out_pre_inv["injectSteps"][0]["systemMessage"]["systemMessage"] and out_pre_inv2 == {},
        f"post={out_post} pre1={out_pre_inv} pre2={out_pre_inv2}",
    )

    out_stop_blk = run_hook("stop_gate.py", {"cwd": str(repo), "terminationReason": "NO_TOOL_CALL", "continuationActive": False})
    out_stop_cont = run_hook("stop_gate.py", {"cwd": str(repo), "terminationReason": "NO_TOOL_CALL", "continuationActive": True})
    check(
        "Stop gate blocks premature Done on failing invariant and respects continuationActive",
        out_stop_blk.get("decision") == "continue" and out_stop_cont.get("decision") == "allow",
        f"blk={out_stop_blk} cont={out_stop_cont}",
    )

if failures:
    print(f"\n{len(failures)} harness check(s) failed")
    sys.exit(1)
print("\nall harness checks passed")
