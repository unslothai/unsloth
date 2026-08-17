# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend CI runs one interpreter on a pull request, and that rests on two other things.

Measured on one runner over the same tree, the four interpreter legs collect the same
26,320 tests and differ by exactly one: ``test_demonstrates_the_underlying_stdlib_regression``
is gated on ``sys.version_info >= (3, 12)``, so 3.10 and 3.11 report 26193 passed / 127
skipped while 3.12 and 3.13 report 26194 / 126.

Dropping three legs is only defensible while both of these hold, so both are asserted here:

* the floor is still checked, statically, by ``tests/test_python39_compatibility.py``,
  which parses every packaged module at the version ``pyproject.toml`` declares. That is a
  LOWER floor than the matrix ever ran, and it is what actually catches the older-syntax
  class of breakage;
* main still runs all four, so a stdlib behaviour difference on an interior version is
  caught on merge rather than never.
"""

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"
FLOOR_CHECK = REPO / "tests" / "test_python39_compatibility.py"


def _matrices() -> tuple[list[str], list[str]]:
    """The pull-request list and the full list, read out of the matrix expression."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    expression = str(document["jobs"]["pytest"]["strategy"]["matrix"]["python"])
    lists = re.findall(r"fromJSON\('(\[[^']*\])'\)", expression)
    assert len(lists) == 2, f"expected a pull-request list and a full list, found {lists}"
    return tuple(ast.literal_eval(item) for item in lists)  # type: ignore[return-value]


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_the_pull_request_leg_is_the_newest_one():
    """Of the four, the ceiling is the one worth keeping.

    Removals and deprecations land there first, and the single version-gated test in the
    suite is a ``>= 3.12`` gate, so on the ceiling it RUNS rather than skipping. A subset
    that kept an older leg instead would trade a test that executes for one that does not.
    """
    subset, full = _matrices()
    assert set(subset) <= set(full), f"{subset} is not a subset of {full}"
    assert max(subset, key = _version) == max(full, key = _version), (
        f"the pull-request matrix {subset} does not include the newest interpreter in "
        f"{full}, which is where removals and deprecations show up first"
    )


def test_the_declared_floor_is_still_checked_statically():
    """The static check is what replaced the older legs, so it has to still be there.

    Asserted through what it DOES rather than by filename: it must read the floor from
    pyproject.toml and hand it to ast.parse as feature_version, which is what makes it a
    floor check rather than a parse on whatever interpreter happens to run it.
    """
    assert FLOOR_CHECK.is_file(), (
        f"{FLOOR_CHECK.name} is gone. It is the reason the pull-request matrix can drop "
        f"the older interpreters, so removing it means restoring them."
    )
    text = FLOOR_CHECK.read_text(encoding = "utf-8")
    assert "requires-python" in text, "the floor is no longer read from pyproject.toml"
    assert "feature_version" in text, (
        "the check no longer parses at the declared floor, so it would pass on syntax that "
        "the floor cannot parse"
    )


def test_the_static_floor_is_below_the_matrix_floor():
    """Otherwise the static check is not covering the versions the matrix stopped running."""
    text = (REPO / "pyproject.toml").read_text(encoding = "utf-8")
    declared = re.search(r"^requires-python\s*=\s*[\"'][^\"']*>=\s*(\d+)\.(\d+)", text, re.M)
    assert declared, "no >= lower bound in requires-python"
    floor = (int(declared.group(1)), int(declared.group(2)))
    _subset, full = _matrices()
    assert floor <= min(_version(leg) for leg in full), (
        f"the declared floor {floor} is above the oldest matrix leg, so the static check "
        f"is not covering what the matrix stopped running"
    )


def test_the_full_matrix_still_runs_on_main():
    """The single leg is only defensible because main runs everything."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") or {}
    assert "main" in (push.get("branches") or []), (
        "Backend CI no longer runs on push to main, so the interpreter versions dropped "
        "from the pull-request matrix would not be tested anywhere"
    )
