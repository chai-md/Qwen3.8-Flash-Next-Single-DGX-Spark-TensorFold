#!/usr/bin/env python3
"""Hash-bound GPU verification receipts for Stage-2 commit gating.

Whenever runtime/kernel files (`patches/*.patch`, `scripts/config.sh`) are
committed, the `PreToolUse` git-commit hook requires a matching verification
receipt proving:
  1. `tools/exact.py` PASS (serial == draft == concurrent)
  2. `tools/exact.py --pad 3000` PASS
  3. Interleaved A/B benchmark (`ctl-a / cand / ctl-b`) PASS with no regressions
  4. Peak unified memory <= 112.0 GiB (respecting 9 GiB reserve on 121 GiB Spark)
  5. Prompt reuse >= 90.0% (when prompt cache / chunk / KV settings are touched)

Receipts are keyed by the SHA-256 fingerprint of `patches/*.patch` +
`scripts/config.sh`, so ANY post-test edit invalidates the receipt automatically.

Usage:
  tools/receipt.py fingerprint [--staged]
  tools/receipt.py record --exact-pass --exact-pad3000-pass --ab-pass \\
      --mem-peak-gib 85.57 --prompt-reuse-pct 99.3 --summary "..."
  tools/receipt.py verify [--staged] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNTIME_PREFIXES = ("patches/",)
RUNTIME_EXACT_FILES = ("scripts/config.sh",)
MEM_CAP_GIB = 112.0
PROMPT_REUSE_MIN_PCT = 90.0


def repo_paths(cwd: str | None = None) -> tuple[Path, Path]:
    cwd_p = Path(cwd or os.getcwd()).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd_p, capture_output=True, text=True)
    worktree_root = Path(top.stdout.strip()).resolve() if top.returncode == 0 and top.stdout.strip() else cwd_p

    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=cwd_p, capture_output=True, text=True)
    if common.returncode == 0 and common.stdout.strip():
        cpath = Path(common.stdout.strip())
        if not cpath.is_absolute():
            cpath = (cwd_p / cpath).resolve()
        main_root = cpath.parent
    else:
        main_root = worktree_root
    return worktree_root, main_root


def receipts_dir(cwd: str | None = None) -> Path:
    override = os.environ.get("SPARK_RECEIPTS_DIR")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    _, main_root = repo_paths(cwd)
    d = main_root / "work" / "hooks" / "receipts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_runtime_file(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if rel in RUNTIME_EXACT_FILES:
        return True
    return any(rel.startswith(pref) and rel.endswith(".patch") for pref in RUNTIME_PREFIXES)


def _list_staged_runtime_files(root: Path) -> list[str]:
    r = subprocess.run(
        ["git", "ls-files", "--cached", "--", "patches/", "scripts/config.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    files = sorted(f for f in r.stdout.splitlines() if f and is_runtime_file(f))
    return files


def compute_fingerprint(cwd: str | None = None, staged: bool = False) -> tuple[str, list[str]]:
    root, _ = repo_paths(cwd)
    h = hashlib.sha256()
    included: list[str] = []

    if staged:
        files = _list_staged_runtime_files(root)
        for rel in files:
            r = subprocess.run(["git", "show", f":{rel}"], cwd=root, capture_output=True)
            if r.returncode != 0:
                continue
            h.update(rel.encode("utf-8") + b"\0")
            h.update(r.stdout)
            h.update(b"\0")
            included.append(rel)
    else:
        candidates: list[str] = []
        patches_dir = root / "patches"
        if patches_dir.is_dir():
            for p in sorted(patches_dir.glob("*.patch")):
                candidates.append(f"patches/{p.name}")
        if (root / "scripts" / "config.sh").is_file():
            candidates.append("scripts/config.sh")
        for rel in sorted(candidates):
            data = (root / rel).read_bytes()
            h.update(rel.encode("utf-8") + b"\0")
            h.update(data)
            h.update(b"\0")
            included.append(rel)

    return h.hexdigest(), included


def receipt_path_for(fp: str, cwd: str | None = None) -> Path:
    return receipts_dir(cwd) / f"{fp[:16]}.json"


def validate_receipt_payload(data: dict, expected_fp: str) -> list[str]:
    problems: list[str] = []
    if data.get("fingerprint") != expected_fp:
        problems.append("fingerprint mismatch (runtime files changed after verification)")
    gates = data.get("gates", {})
    if gates.get("exact_py") is not True:
        problems.append("tools/exact.py did not pass")
    if gates.get("exact_pad3000") is not True:
        problems.append("tools/exact.py --pad 3000 did not pass")
    if gates.get("ab_interleaved") is not True:
        problems.append("interleaved A/B benchmark (ctl-a/cand/ctl-b) did not pass")
    mem = data.get("mem_peak_gib")
    if not isinstance(mem, (int, float)) or mem <= 0 or mem > MEM_CAP_GIB:
        problems.append(f"mem_peak_gib={mem!r} missing or exceeds {MEM_CAP_GIB} GiB cap")
    reuse = data.get("prompt_reuse_pct")
    if reuse is not None and (not isinstance(reuse, (int, float)) or reuse < PROMPT_REUSE_MIN_PCT):
        problems.append(f"prompt_reuse_pct={reuse!r} below {PROMPT_REUSE_MIN_PCT}% floor")
    if not str(data.get("summary", "")).strip():
        problems.append("summary is empty")
    return problems


def cmd_record(args: argparse.Namespace) -> int:
    fp, included = compute_fingerprint(staged=args.staged)
    metrics = {}
    if args.metrics_json:
        metrics = json.loads(args.metrics_json)
    payload = {
        "fingerprint": fp,
        "files": included,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "holder": args.holder,
        "image": args.image,
        "gates": {
            "exact_py": bool(args.exact_pass),
            "exact_pad3000": bool(args.exact_pad3000_pass),
            "ab_interleaved": bool(args.ab_pass),
        },
        "mem_peak_gib": args.mem_peak_gib,
        "prompt_reuse_pct": args.prompt_reuse_pct,
        "summary": args.summary,
        "metrics": metrics,
    }
    problems = validate_receipt_payload(payload, fp)
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    out = receipt_path_for(fp)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"recorded receipt {out} (fingerprint={fp[:16]})")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    fp, included = compute_fingerprint(staged=args.staged)
    path = receipt_path_for(fp)
    problems: list[str] = []
    data: dict | None = None
    if not path.is_file():
        problems.append(
            f"no GPU verification receipt found for fingerprint {fp[:16]} "
            f"(expected {path})"
        )
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                problems.append("receipt JSON is not an object")
            else:
                problems.extend(validate_receipt_payload(data, fp))
        except Exception as e:
            problems.append(f"failed to read receipt: {e}")

    ok = not problems
    if args.json:
        print(json.dumps({
            "ok": ok,
            "fingerprint": fp,
            "receipt_path": str(path),
            "files": included,
            "problems": problems,
            "receipt": data,
        }, indent=2))
    else:
        if ok:
            print(f"ok   receipt {fp[:16]} verified ({path})")
        else:
            for p in problems:
                print(f"FAIL {p}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Hash-bound GPU verification receipts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fp = sub.add_parser("fingerprint")
    p_fp.add_argument("--staged", action="store_true")

    p_rec = sub.add_parser("record")
    p_rec.add_argument("--staged", action="store_true")
    p_rec.add_argument("--holder", default="bench-verifier")
    p_rec.add_argument("--image", default="")
    p_rec.add_argument("--exact-pass", action="store_true")
    p_rec.add_argument("--exact-pad3000-pass", action="store_true")
    p_rec.add_argument("--ab-pass", action="store_true")
    p_rec.add_argument("--mem-peak-gib", type=float, required=True)
    p_rec.add_argument("--prompt-reuse-pct", type=float, default=None)
    p_rec.add_argument("--summary", required=True)
    p_rec.add_argument("--metrics-json", default="")

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--staged", action="store_true")
    p_ver.add_argument("--json", action="store_true")

    args = ap.parse_args()
    if args.cmd == "fingerprint":
        fp, _ = compute_fingerprint(staged=args.staged)
        print(fp)
        return 0
    if args.cmd == "record":
        return cmd_record(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
