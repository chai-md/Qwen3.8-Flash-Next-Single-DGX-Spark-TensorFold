#!/usr/bin/env python3
"""PreInvocation hook: surface any static invariant failures captured after the last edit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def main() -> int:
    payload = common.read_payload()
    cwd = common.resolve_cwd(payload)
    info = common.git_info(cwd)
    sf = common.state_file(info)

    if not sf.is_file():
        common.emit({})
        return 0

    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        common.emit({})
        return 0

    if data.get("ok", True) or data.get("surfaced", False):
        common.emit({})
        return 0

    errors = data.get("errors", [])
    data["surfaced"] = True
    try:
        sf.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    bullet_list = "\n".join(f"  - {e}" for e in errors[:8])
    extra = f"\n  (+ {len(errors) - 8} more)" if len(errors) > 8 else ""
    common.emit({
        "injectSteps": [
            {
                "systemMessage": {
                    "systemMessage": (
                        "[Static Invariant Violation Detected After Last Edit]\n"
                        "Fix these deterministic check failures before continuing:\n"
                        f"{bullet_list}{extra}"
                    )
                }
            }
        ]
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
