# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lint CI's file scans read the committed tree, not the working directory.

The Lint CI job runs its checks as one parallel group next to background lanes, and the
load-orchestrator lane builds a venv under ``.lanes/`` while they run. The debugger check
walked ``pathlib.Path(".").rglob("*.py")``, listed a file pip was about to remove
(``.lanes/venv-load-orchestrator/.../~ip/__init__.py``), and crashed on ``read_text`` with
FileNotFoundError, failing PR #13700's lint for a file nobody committed. The YAML and JSON
parse checks walk the same way and would report such a file as unparseable instead.
"""

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "lint-ci.yml"
TEXT = WORKFLOW.read_text(encoding = "utf-8")
BLOCKS = [
    m.group("body") for m in re.finditer(r"python <<'PY'\n(?P<body>.*?)\n[ \t]*PY\n", TEXT, re.S)
]


def test_no_lint_step_walks_the_working_tree():
    walkers = [
        block.splitlines()[0].strip()
        for block in BLOCKS
        if re.search(r"\.rglob\(|os\.walk\(", block)
    ]
    assert not walkers, (
        "a Lint CI step walks the working tree, which the background lanes write into while it "
        f"runs; list committed files with git ls-files instead: {walkers}"
    )


def test_the_file_scans_list_committed_files():
    scans = {
        "debugger": "no leftover debugger calls",
        "yaml": "YAML files parse cleanly",
        "json": "JSON files parse cleanly",
    }
    for name, marker in scans.items():
        block = next((b for b in BLOCKS if marker in b), None)
        assert block is not None, f"the {name} scan is gone or reworded; update this guard"
        assert (
            '"git", "ls-files"' in block and "tracked(" in block
        ), f"the {name} scan does not read git ls-files"
