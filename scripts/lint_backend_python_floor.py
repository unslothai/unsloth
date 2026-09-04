#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refuse backend source that needs a newer interpreter than the matrix floor.

A pull request runs Backend CI on the NEWEST interpreter only. Every older leg still runs
on the push to main, so a version-specific break is caught at merge rather than never, but
between opening a pull request and merging it nothing executes the backend on the oldest
one. This closes as much of that gap as a static check can.

Syntax is the easy half, and ``tests/test_python39_compatibility.py`` already covers it by
parsing at the version ``pyproject.toml`` declares. Syntax is also not the shape this
regression takes. The realistic mistake is reaching for a stdlib name that does not exist
yet -- ``core/research_runs.py`` already uses ``anext``, which is 3.10 -- and that parses
perfectly on every version and fails only when the line runs.

So this asks vermin, which reads both syntax and stdlib API availability, and compares the
answer against the oldest leg the workflow's own matrix declares rather than a number
written here. Raise the floor in the matrix and this follows; use a symbol from above it
and this fails in seconds, on every pull request, instead of on main in 23 minutes.

What it cannot do, stated so nobody mistakes it for the legs it partly replaces: it does
not run anything. Two interpreters that both accept a line can still behave differently on
it, and a ``sys.version_info`` branch is only ever parsed here, never taken. That is what
the full matrix on main is for.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"

# Both trees the matrix legs actually execute.
# studio-backend-ci lists 'unsloth_cli/**' in its own paths filter and runs `pytest unsloth_cli/tests` as a step on
# every leg, so a post-floor stdlib name on a shipped CLI path was covered by the old 3.10 leg exactly as a backend
# one was. Scanning only the backend would have moved that coverage to the push to main while looking like it had
# replaced it.
ROOTS = (
    REPO / "studio" / "backend",
    REPO / "unsloth_cli",
)

# Everything shipped under studio/backend is scanned.
# The first version of this listed the packages instead, and that is exactly the wrong shape for a floor check: it named
# core, utils and routes and silently missed 116 files, including all of hub, plugins, models, storage, auth, picker and
# state, plus _platform_compat.py which main.py imports directly.
# It also named "loggers.py", which is a directory, so that entry matched nothing at all.
# studio-backend-ci runs `pytest tests/` from studio/backend on every leg, so a 3.11 API in a test file is executed by
# the 3.10 leg exactly as one in a shipped module is.
# With the pull request down to a single 3.13 leg, that leg and this lint would both pass and the failure would arrive
# on the push to main, which is the whole gap this exists to close.
EXCLUDE_PARTS = ("vendor", "node_modules", "__pycache__", ".venv")


# An above-floor symbol reached deliberately is suppressed AT THE SITE, with `# novermin` and a comment saying why, not
# by dropping its file from the scan.
# The one live case is locale.getencoding() in the data-designer plugin's state_store, inside a try/except
# AttributeError with a pre-3.11 fallback.


# The floor is DECLARED, in the workflow, next to where the legs used to be.
# Deriving it from the matrix became self-defeating once the matrix ran one interpreter: a 3.13-only matrix would move
# the floor to 3.13 and leave this asserting that code written for 3.13 runs on 3.13. Deriving it from pyproject.toml
# is not the answer either, because that says >= 3.9 and is not true today: unsloth/models/_utils.py already uses
# tempfile.TemporaryDirectory(ignore_cleanup_errors), which is 3.10, so a 3.9 target fails on the tree as it stands.
# So it is a number, written down once, in the workflow that would otherwise have tested it, and read from there.
FLOOR_KEY = "PYTHON_FLOOR"


def declared_floor() -> tuple[int, int]:
    """The floor the workflow declares, as (major, minor)."""
    text = WORKFLOW.read_text(encoding = "utf-8")
    found = re.search(rf"^\s*{FLOOR_KEY}:\s*['\"]?(\d+)\.(\d+)['\"]?\s*$", text, re.M)
    if not found:
        raise SystemExit(
            f"{WORKFLOW.name} declares no {FLOOR_KEY}, so this lint has no target. It is "
            f"declared there rather than here so that the number lives with the CI that "
            f"used to test it."
        )
    return int(found.group(1)), int(found.group(2))


def targets() -> list[str]:
    """Every .py the matrix legs ship or execute, found rather than listed."""
    found = []
    for root in ROOTS:
        if not root.is_dir():
            raise SystemExit(f"{root} is gone; the scan would silently stop covering it")
        found.extend(
            str(path)
            for path in sorted(root.rglob("*.py"))
            if not any(part in EXCLUDE_PARTS for part in path.relative_to(root).parts)
        )
    if not found:
        raise SystemExit(f"no python files found under {ROOTS}; the scan would pass on nothing")
    return found


def main() -> int:
    floor = declared_floor()
    target = f"{floor[0]}.{floor[1]}"
    # The console script, not `python -m vermin`: the package has no __main__, so that
    # form exits nonzero for the wrong reason and this lint would fail on every run while
    # looking like it had found something.
    vermin = shutil.which("vermin")
    if vermin is None:
        raise SystemExit(
            "vermin is not installed, so the backend floor is unchecked. Install it in "
            "the job that runs this, rather than letting the check quietly pass."
        )
    files = targets()
    print(f"[floor] {len(files)} files must run on Python {target}, " f"the declared floor")
    command = [
        vermin,
        "--no-tips",
        "--violations",
        f"-t={target}",
        *files,
    ]
    result = subprocess.run(command, capture_output = True, text = True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode == 0:
        print(f"[floor] OK: nothing needs more than {target}")
        return 0
    print(
        f"::error title=Backend needs a newer Python than the matrix floor::"
        f"something under studio/backend or unsloth_cli requires more than Python {target}, "
        f"which is the "
        f"floor studio-backend-ci declares. Nothing runs that interpreter any more, so "
        f"this check is the only thing standing between an above-floor symbol and a user "
        f"on that version. Either guard the usage behind a sys.version_info check, or "
        f"raise {FLOOR_KEY} in the workflow and say why."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
