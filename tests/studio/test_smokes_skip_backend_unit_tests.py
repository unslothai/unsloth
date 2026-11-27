# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Studio smokes that boot the app skip edits confined to studio/backend/tests/**.

That is only safe while nothing the smokes run reads that tree: the running backend must not
import from it, and the smoke workflows (and the repo scripts they call) must not open it.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
BACKEND = REPO / "studio" / "backend"
EXCLUDE = "!studio/backend/tests/**"
SMOKES = sorted(p for p in WORKFLOWS.glob("*.yml") if EXCLUDE in p.read_text(encoding = "utf-8"))


def test_some_smokes_use_the_exclusion():
    assert len(SMOKES) >= 5, [p.name for p in SMOKES]


@pytest.mark.parametrize("path", SMOKES, ids = [p.name for p in SMOKES])
def test_the_exclusion_only_narrows_studio_backend(path):
    on = yaml.safe_load(path.read_text(encoding = "utf-8"))[True]
    for event in ("pull_request", "push"):
        paths = (on.get(event) or {}).get("paths") or []
        if not paths:
            continue
        assert EXCLUDE in paths, f"{path.name}: {event} lacks {EXCLUDE}"
        i = paths.index(EXCLUDE)
        assert paths[i - 1] == "studio/backend/**", f"{path.name}: {event} exclusion is misplaced"
        later = [p for p in paths[i + 1 :] if p.startswith("studio/backend/tests")]
        assert not later, f"{path.name}: {event} re-includes {later}; drop the exclusion instead"


@pytest.mark.parametrize("path", SMOKES, ids = [p.name for p in SMOKES])
def test_the_smoke_never_opens_backend_unit_tests(path):
    texts = [path.read_text(encoding = "utf-8")]
    for rel in set(re.findall(r"\.github/scripts/[\w./-]+", texts[0])):
        script = REPO / rel
        if script.is_file():
            texts.append(script.read_text(encoding = "utf-8"))
    for text in texts:
        code = "\n".join(
            line.split("#", 1)[0] for line in text.splitlines() if line.strip() != f"- '{EXCLUDE}'"
        )
        assert "studio/backend/tests" not in code and "backend/tests/" not in code, path.name


def test_the_running_backend_never_imports_its_unit_tests():
    offenders = []
    for source in BACKEND.rglob("*.py"):
        rel = source.relative_to(BACKEND)
        if rel.parts[0] == "tests":
            continue
        tree = ast.parse(source.read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for name in names:
                if name == "tests" or name.startswith(
                    ("tests.", "backend.tests", "studio.backend.tests")
                ):
                    offenders.append(f"{rel}: {name}")
    assert not offenders, offenders
