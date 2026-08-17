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

BACKEND = REPO / "studio" / "backend"

# Everything shipped under studio/backend is scanned. The first version of this listed the
# packages instead, and that is exactly the wrong shape for a floor check: it named core,
# utils and routes and silently missed 116 files, including all of hub, plugins, models,
# storage, auth, picker and state, plus _platform_compat.py which main.py imports directly.
# It also named "loggers.py", which is a directory, so that entry matched nothing at all.
# A check that covers most of the tree reads exactly like one that covers all of it.
#
# So the tree is the input and only these come out:
#   tests   -- run by whatever interpreter runs them, and not shipped
#   vendor  -- third party, pinned to its own support range
EXCLUDE_PARTS = ("tests", "vendor", "node_modules", "__pycache__", ".venv")

# An above-floor symbol reached deliberately is suppressed AT THE SITE, with `# novermin`
# and a comment saying why, not by dropping its file from the scan. Excluding the file
# would leave everything else in it permanently unchecked, which is the same mistake as
# the package allowlist this replaced, one level down.
#
# The one live case is locale.getencoding() in the data-designer plugin's state_store,
# inside a try/except AttributeError with a pre-3.11 fallback. vermin reads names rather
# than control flow, so it cannot see that the guard is already there.
#
# Comment parsing is therefore ON, which is what makes the annotation work.


def matrix_floor() -> tuple[int, int]:
    """The oldest interpreter the workflow's matrix names, read from the workflow.

    Taken from the FULL list rather than the pull-request subset: the subset is the
    ceiling only, and the floor this lint defends is the oldest version main still runs.
    """
    text = WORKFLOW.read_text(encoding = "utf-8")
    lists = re.findall(r"fromJSON\('(\[[^']*\])'\)", text)
    if not lists:
        raise SystemExit(f"no fromJSON matrix found in {WORKFLOW.name}; this lint cannot aim")
    versions = []
    for item in lists:
        versions.extend(json.loads(item))
    parsed = sorted(tuple(int(part) for part in v.split(".")) for v in versions)
    return parsed[0]


def targets() -> list[str]:
    """Every shipped .py under the backend, found rather than listed."""
    found = [
        str(path)
        for path in sorted(BACKEND.rglob("*.py"))
        if not any(part in EXCLUDE_PARTS for part in path.relative_to(BACKEND).parts)
    ]
    if not found:
        raise SystemExit(f"no python files found under {BACKEND}; the scan would pass on nothing")
    return found


def main() -> int:
    floor = matrix_floor()
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
    print(
        f"[floor] {len(files)} backend files must run on Python {target}, "
        f"the oldest leg in the matrix"
    )
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
        print(f"[floor] OK: nothing in the backend needs more than {target}")
        return 0
    print(
        f"::error title=Backend needs a newer Python than the matrix floor::"
        f"something under studio/backend requires more than Python {target}, which is the "
        f"oldest leg studio-backend-ci runs. A pull request only runs the newest leg, so "
        f"this would otherwise fail on the push to main. Either guard the usage behind a "
        f"sys.version_info check, or raise the floor in the workflow matrix and say why."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
