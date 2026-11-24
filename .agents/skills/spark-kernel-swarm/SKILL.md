---
name: spark-kernel-swarm
description: >-
  Runbook for high-cost Zig/CUDA kernel optimization on a single NVIDIA DGX Spark (GB10)
  using /swarm together with the repository's deterministic /hooks guardrails. Use when
  starting a /swarm session to analyze, author, review, and GPU-verify TensorFold patches.
---

# DGX Spark High-Reliability Kernel Optimization (`/swarm` + `/hooks`)

Trust precedes parallelism. Every `/swarm` worker inherits `.agents/hooks.json`, so
deterministic checks, GPU exclusivity, and Stage-2 commit receipts are enforced by
the harness rather than prose.

## 1. Swarm Role Topology (Single-GPU Safe)

| Role | `workspace` | Allowed Work | Blocked by Hooks |
| --- | --- | --- | --- |
| `coordinator` | `inherit` | Hypothesis queue, `work/opt/PLAN.md` & `ledger.jsonl`, merge verified diffs, `git commit` | Committing `patches/` or `scripts/config.sh` without a matching GPU receipt |
| `profiler` | `inherit` | Read-only TensorFold Zig/CUDA source analysis, roofline/byte math, Nsight/profile breakdown | Editing protected tests/hooks |
| `author-<k>` | `share` | Write `patches/000N-*.patch` in isolated git worktree; run `tools/check_invariants.py`, `zig build check-native-cuda-host`, `nvcc --ptxas-options=-v` | **All GPU containers / `./start.sh` / `nsys` / `ncu`** (blocked in secondary worktrees) |
| `critic` | `inherit` | Adversarial diff review: numerical bounds, CUDA graph capture hazards, `environment.zig` registration, memory budget | Editing protected tests/hooks |
| `bench-verifier` *(singleton)* | `inherit` | Acquire `tools/gpu_lock.py`, build image, run `ctl-a / cand / ctl-b` + `tools/exact.py` + `exact.py --pad 3000` + `prompt_reuse.py`, emit receipt via `tools/receipt.py record` | Concurrent GPU runs while lock held by another worker |

## 2. Paved Road (Mandatory Pipeline)

1. **Hypothesize (`profiler` → `coordinator`)**: Quantify the bottleneck against `work/opt/THEORY.md` (measured GB10: 225.2 GB/s D2D, 97.5 TFLOPS BF16; practical prefill ceiling ~3.8–3.9K tok/s). Specify exact kernel/file, predicted delta, and memory cost.
2. **Implement in Isolation (`author-<k>`, `workspace: "share"`)**:
   - Edit cleanly; any new `TF_FLASHNEXT_*` variable must be registered in `zig/src/platform/environment.zig`.
   - Zero `TODO`/`FIXME`/`HACK`, zero `std.debug.print` in non-test Zig, zero unguarded `cuda*Synchronize()` in CUDA graph capture paths.
   - Verify `python3 tools/check_invariants.py` passes (enforced automatically by `Stop` hook).
3. **Adversarial Review (`critic`)**: Audit the patch diff for FP8/MXFP8 rounding drift, KV/prompt-cache memory pinch (`CHUNK_SOLO <= 8192`), and CUDA graph safety before consuming GPU time.
4. **GPU Proof & Receipt (`bench-verifier`, singleton)**:
   ```bash
   python3 tools/gpu_lock.py acquire --holder bench-verifier --reason "<patch-name>"
   # Build image, run interleaved ctl-a / cand / ctl-b on HOST=127.0.0.1,
   # run tools/exact.py, tools/exact.py --pad 3000, tools/prompt_reuse.py 30000
   python3 tools/receipt.py record \
     --holder bench-verifier \
     --image "<image-tag>" \
     --exact-pass --exact-pad3000-pass --ab-pass \
     --mem-peak-gib <PEAK_GIB> \
     --prompt-reuse-pct <WORST_PCT> \
     --summary "<concise A/B delta>"
   python3 tools/gpu_lock.py release --holder bench-verifier
   ```
5. **Land (`coordinator`)**: Merge the verified patch into the main workspace, confirm `python3 tools/receipt.py verify --staged`, and `git commit` (automatically gated by Stage-1 + Stage-2 hooks).

## 3. Turning Every Failure Into a Constraint

Whenever any worker hits a new failure mode (compiler quirk, CUDA graph capture error, memory eviction, silent regression), add a deterministic check to `tools/check_invariants.py` and a regression test to `tools/test_hooks_harness.py` before launching the next wave.
