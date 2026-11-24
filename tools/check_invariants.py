#!/usr/bin/env python3
"""Deterministic codebase invariant checker (Lauren Tan's Layer-2 static gate).

Zero model judgment: every mistake learned on DGX Spark / TensorFold becomes a
fast, repeatable check run by Antigravity hooks (`PostToolUse`, `Stop`, and
`PreToolUse` before `git commit`).

Checks:
  1. patches/ unbroken 0001..N sequence, valid unified diff, no conflict markers.
  2. Every TF_FLASHNEXT_* env var introduced in patches/*.patch (non-test Zig)
     or exported in scripts/config.sh is either a known v1.3.4 upstream/legacy
     variable OR registered in `zig/src/platform/environment.zig` inside patches/.
  3. CUDA Graph & hot-path safety in patches/*.patch added lines:
     - no unguarded cudaDeviceSynchronize / cuCtxSynchronize / cudaStreamSynchronize
       (escape hatch: explicit `// GRAPH_SAFE: <reason>`)
     - no printf( in non-test .cu/.cuh additions (escape hatch: `// DEBUG_OK:`)
     - no std.debug.print in non-test .zig additions
     - no slop markers (TODO/FIXME/XXX/HACK/WORKAROUND) in newly added patch lines
  4. DGX Spark capacity & config guardrails in scripts/config.sh:
     - CONTEXT=262144, PARALLEL=5, VISION=1, KV_DTYPE=fp8
     - TENSORFOLD_MEMORY_RESERVE_GIB >= 9
     - TF_FLASHNEXT_CHUNK_SOLO <= 8192 (10240/12288 pinch prompt cache and regress c5)
     - TF_FLASHNEXT_PROMPT_MX128 in {0, 1024}
  5. `bash -n` syntax validation on start.sh, stop.sh, scripts/*.sh.
  6. All 7 offline repo unit tests (unless `--fast` is passed).

Usage:
  tools/check_invariants.py [--root PATH] [--fast] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Known upstream TensorFold v1.3.4 + existing MiaAI-Lab patch (0002/0003) TF_FLASHNEXT_* vars.
# Any NEW TF_FLASHNEXT_* variable MUST also be added to `zig/src/platform/environment.zig`.
KNOWN_TF_FLASHNEXT_VARS = {
    "TF_FLASHNEXT_ATTN_LANES",
    "TF_FLASHNEXT_CHUNK",
    "TF_FLASHNEXT_CHUNK_SOLO",
    "TF_FLASHNEXT_CONFIDENCE",
    "TF_FLASHNEXT_DECODE_GRAPH",
    "TF_FLASHNEXT_DRAFTS",
    "TF_FLASHNEXT_DRAFTS_GREEDY",
    "TF_FLASHNEXT_DRAFT_HEAD",
    "TF_FLASHNEXT_FACT_ROWS",
    "TF_FLASHNEXT_FUSED_SPLIT",
    "TF_FLASHNEXT_GATHER_EARLY",
    "TF_FLASHNEXT_GATHER_READ",
    "TF_FLASHNEXT_GDN_FUSED",
    "TF_FLASHNEXT_GDN_LANES",
    "TF_FLASHNEXT_GPU_TOPK",
    "TF_FLASHNEXT_HC_FUSE",
    "TF_FLASHNEXT_HEAD_KV",
    "TF_FLASHNEXT_KEEP_INPASS",
    "TF_FLASHNEXT_KEEP_LATE",
    "TF_FLASHNEXT_KV_PAGES",
    "TF_FLASHNEXT_LOAD_THREADS",
    "TF_FLASHNEXT_PLE_IO",
    "TF_FLASHNEXT_PROFILE",
    "TF_FLASHNEXT_PROMPT_EXPERT_TILE",
    "TF_FLASHNEXT_PROMPT_MX128",
    "TF_FLASHNEXT_READOUT_FUSED",
    "TF_FLASHNEXT_READOUT_ROWS",
    "TF_FLASHNEXT_SAMPLED_HEADS",
    "TF_FLASHNEXT_SHARED_DRAFTS",
    "TF_FLASHNEXT_SHARED_SIDE",
    "TF_FLASHNEXT_TABLE_WARM",
    "TF_FLASHNEXT_WARM_AHEAD",
}

OFFLINE_UNIT_TESTS = [
    "tools/test_ablit_view.py",
    "tools/test_video_input.py",
    "tools/test_pdf_input.py",
    "tools/test_start_meminfo.py",
    "tools/test_stop_health.py",
    "tools/test_harden_recipe.py",
    "tools/test_readme_claims.py",
]

SLOP_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK|WORKAROUND)\b")
SYNC_RE = re.compile(r"\b(cudaDeviceSynchronize|cuCtxSynchronize|cudaStreamSynchronize)\s*\(")
GETENV_RE = re.compile(r'getenv\s*\(\s*"(TF_FLASHNEXT_[A-Z0-9_]+)"\s*\)')
ENV_REG_RE = re.compile(r'\.name\s*=\s*"(TF_FLASHNEXT_[A-Z0-9_]+)"')
CONFLICT_RE = re.compile(r"^(<{7}|={7}|>{7})(\s|$)")


def is_test_target(path: str) -> bool:
    p = path.lower()
    return "_test." in p or "/test/" in p or "/tests/" in p or p.startswith("tests/")


def check_patches(root: Path) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    registered_in_patches: set[str] = set()
    read_in_patches: list[tuple[str, int, str]] = []

    patches_dir = root / "patches"
    if not patches_dir.is_dir():
        return ["patches/ directory is missing"], registered_in_patches

    patch_files = sorted(patches_dir.glob("*.patch"))
    numbered = [p for p in patch_files if re.match(r"^\d{4}-", p.name)]
    unnumbered = [p.name for p in patch_files if not re.match(r"^\d{4}-", p.name)]
    if unnumbered:
        errors.append(f"unnumbered patch files in patches/: {unnumbered}")

    nums = [int(p.name[:4]) for p in numbered]
    expected = list(range(1, len(numbered) + 1))
    if nums != expected:
        errors.append(f"patches/ numbering must be unbroken 0001..{len(numbered):04d}, got {nums}")

    for p in numbered:
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if not any(l.startswith("diff --git ") for l in lines):
            errors.append(f"{p.name}: missing `diff --git` header")
        if not any(l.startswith("@@ ") for l in lines):
            errors.append(f"{p.name}: missing unified diff hunk (`@@ `)")

        cur_file = ""
        for idx, line in enumerate(lines, 1):
            if line.startswith("diff --git "):
                parts = line.split()
                cur_file = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else ""
                continue
            if line.startswith("GIT binary patch"):
                errors.append(f"{p.name}:{idx}: binary patch is forbidden")
                continue
            if not line.startswith("+") or line.startswith("+++"):
                continue

            added = line[1:]
            if CONFLICT_RE.match(added):
                errors.append(f"{p.name}:{idx}: merge conflict marker in added line")

            if cur_file.endswith("environment.zig"):
                for m in ENV_REG_RE.finditer(added):
                    registered_in_patches.add(m.group(1))

            in_test = is_test_target(cur_file)
            is_code = cur_file.endswith((".zig", ".cu", ".cuh", ".c", ".h", ".cpp", ".hpp"))

            if is_code and not in_test:
                for m in GETENV_RE.finditer(added):
                    read_in_patches.append((p.name, idx, m.group(1)))

                if SYNC_RE.search(added) and "GRAPH_SAFE:" not in added:
                    errors.append(
                        f"{p.name}:{idx} ({cur_file}): unguarded CUDA synchronize call breaks graph capture "
                        "(remove or annotate with `// GRAPH_SAFE: <reason>`)"
                    )

                if cur_file.endswith((".cu", ".cuh")) and re.search(r"\bprintf\s*\(", added) and "DEBUG_OK:" not in added:
                    errors.append(
                        f"{p.name}:{idx} ({cur_file}): raw `printf(` in CUDA kernel path "
                        "(remove or annotate with `// DEBUG_OK: <reason>`)"
                    )

                if cur_file.endswith(".zig") and "std.debug.print" in added and "DEBUG_OK:" not in added:
                    errors.append(
                        f"{p.name}:{idx} ({cur_file}): `std.debug.print` in non-test Zig code "
                        "(remove or move to *_test.zig)"
                    )

                if SLOP_RE.search(added):
                    errors.append(
                        f"{p.name}:{idx} ({cur_file}): forbidden slop/escape-hatch marker "
                        f"({SLOP_RE.search(added).group(1)}) in added patch line"
                    )

    known_all = KNOWN_TF_FLASHNEXT_VARS | registered_in_patches
    for pname, idx, var in read_in_patches:
        if var not in known_all:
            errors.append(
                f"{pname}:{idx}: `{var}` read via getenv() is not registered in "
                "zig/src/platform/environment.zig (TensorFold v1.3.4 rejects unknown TF_FLASHNEXT_* vars at boot)"
            )

    return errors, registered_in_patches


def _extract_default(text: str, name: str) -> str | None:
    m = re.search(rf'^(?:export\s+)?{re.escape(name)}="\$\{{{re.escape(name)}:-([^"]*)\}}"', text, re.M)
    return m.group(1) if m else None


def check_config(root: Path, registered_in_patches: set[str]) -> list[str]:
    errors: list[str] = []
    cfg_path = root / "scripts" / "config.sh"
    if not cfg_path.is_file():
        return ["scripts/config.sh is missing"]
    text = cfg_path.read_text(encoding="utf-8")

    known_all = KNOWN_TF_FLASHNEXT_VARS | registered_in_patches
    for idx, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^\s*export\s+(TF_FLASHNEXT_[A-Z0-9_]+)=", line)
        if m and m.group(1) not in known_all:
            errors.append(
                f"scripts/config.sh:{idx}: exported `{m.group(1)}` is neither an upstream v1.3.4 var "
                "nor registered in environment.zig by patches/"
            )

    expected_exact = {
        "CONTEXT": "262144",
        "PARALLEL": "5",
        "VISION": "1",
        "KV_DTYPE": "fp8",
    }
    for k, want in expected_exact.items():
        got = _extract_default(text, k)
        if got != want:
            errors.append(f"scripts/config.sh: default {k}={got!r} violates Spark baseline {want!r}")

    reserve = _extract_default(text, "TENSORFOLD_MEMORY_RESERVE_GIB")
    if reserve is None or not reserve.isdigit() or int(reserve) < 9:
        errors.append(
            f"scripts/config.sh: TENSORFOLD_MEMORY_RESERVE_GIB={reserve!r} must be >= 9 "
            "(exhausting 121 GiB unified memory freezes DGX Spark)"
        )

    chunk_solo = _extract_default(text, "TF_FLASHNEXT_CHUNK_SOLO")
    if chunk_solo is not None:
        if not chunk_solo.isdigit() or int(chunk_solo) > 8192 or int(chunk_solo) < 1536:
            errors.append(
                f"scripts/config.sh: TF_FLASHNEXT_CHUNK_SOLO={chunk_solo!r} must be within 1536..8192 "
                "(10240/12288 pinch prompt cache below 2 GiB and regress c5)"
            )

    mx128 = _extract_default(text, "TF_FLASHNEXT_PROMPT_MX128")
    if mx128 not in ("0", "1024"):
        errors.append(f"scripts/config.sh: TF_FLASHNEXT_PROMPT_MX128={mx128!r} must be '1024' or '0'")

    return errors


def check_shell_syntax(root: Path) -> list[str]:
    errors: list[str] = []
    targets = [root / "start.sh", root / "stop.sh"] + sorted((root / "scripts").glob("*.sh"))
    for sh in targets:
        if not sh.is_file():
            continue
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        if r.returncode != 0:
            errors.append(f"{sh.relative_to(root)}: bash -n failed: {r.stderr.strip()}")
    return errors


def check_unit_tests(root: Path) -> list[str]:
    errors: list[str] = []
    for rel in OFFLINE_UNIT_TESTS:
        p = root / rel
        if not p.is_file():
            continue
        r = subprocess.run([sys.executable, str(p)], cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stdout + "\n" + r.stderr).strip().splitlines()[-6:]
            errors.append(f"{rel} FAILED:\n  " + "\n  ".join(tail))
    return errors


def run_all(root: Path, fast: bool = False) -> dict:
    patch_errs, reg_vars = check_patches(root)
    config_errs = check_config(root, reg_vars)
    shell_errs = check_shell_syntax(root)
    unit_errs = [] if fast else check_unit_tests(root)

    checks = {
        "patches": patch_errs,
        "config": config_errs,
        "shell_syntax": shell_errs,
        "unit_tests": unit_errs,
    }
    all_errors = patch_errs + config_errs + shell_errs + unit_errs
    return {
        "ok": len(all_errors) == 0,
        "fast": fast,
        "root": str(root),
        "error_count": len(all_errors),
        "checks": checks,
        "errors": all_errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic codebase invariant checker")
    ap.add_argument("--root", default=None, help="Repository root (default: parent of tools/)")
    ap.add_argument("--fast", action="store_true", help="Skip offline Python unit tests (<0.2s)")
    ap.add_argument("--json", action="store_true", help="Emit JSON report")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    result = run_all(root, fast=args.fast)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, errs in result["checks"].items():
            if name == "unit_tests" and args.fast:
                continue
            if errs:
                print(f"FAIL {name} ({len(errs)} error(s)):")
                for e in errs:
                    print(f"  - {e}")
            else:
                print(f"ok   {name}")
        if result["ok"]:
            print("\nall invariants passed")
        else:
            print(f"\n{result['error_count']} invariant failure(s)")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
