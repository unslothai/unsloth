# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend CI runs both ends of its matrix on a pull request, and all four on main.

Measured on one runner over the same tree, the four interpreter legs collect the same
26,320 tests and differ by exactly one: ``test_demonstrates_the_underlying_stdlib_regression``
is gated on ``sys.version_info >= (3, 12)``, so 3.10 and 3.11 report 26193 passed / 127
skipped while 3.12 and 3.13 report 26194 / 126.

Dropping the two interior legs rests on three things, all asserted here:

* both ENDS are kept. A single leg was tried and is wrong: the backend has
  version-conditional branches that only a run can exercise, so the floor has to execute;
* the syntax floor is still checked statically by ``tests/test_python39_compatibility.py``,
  at the version ``pyproject.toml`` declares, which is lower than any leg here;
* main still runs all four, so a difference peculiar to 3.11 or 3.12 is caught on merge
  rather than never.
"""

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"
FLOOR_CHECK = REPO / "tests" / "test_python39_compatibility.py"
BACKEND = REPO / "studio" / "backend"


def _matrices() -> tuple[list[str], list[str]]:
    """The pull-request list and the full list, read out of the matrix expression."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    expression = str(document["jobs"]["pytest"]["strategy"]["matrix"]["python"])
    lists = re.findall(r"fromJSON\('(\[[^']*\])'\)", expression)
    assert len(lists) == 2, f"expected a pull-request list and a full list, found {lists}"
    return tuple(ast.literal_eval(item) for item in lists)  # type: ignore[return-value]


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_the_pull_request_matrix_keeps_both_ends():
    """The floor exercises the older branches, the ceiling catches removals."""
    subset, full = _matrices()
    assert set(subset) <= set(full), f"{subset} is not a subset of {full}"
    assert min(subset, key = _version) == min(full, key = _version)
    assert max(subset, key = _version) == max(full, key = _version)


def _boundaries() -> dict[str, tuple[int, ...]]:
    """Every ``sys.version_info`` comparison in the backend, source and tests alike.

    Source matters more than tests here, and missing that is what made a single-leg
    matrix look defensible: ``sitecustomize.py`` repoints pathlib's pre-3.11
    ``_NormalAccessor``, ``native_path_leases.py`` and ``third_party_source.py`` branch
    on >= 3.12. A static parse cannot exercise any of them, because it parses rather
    than runs.
    """
    found: dict[str, tuple[int, ...]] = {}
    for path in sorted(BACKEND.rglob("*.py")):
        if "vendor" in path.parts:  # third-party, pinned to its own support range
            continue
        text = path.read_text(encoding = "utf-8", errors = "replace")
        for match in re.finditer(r"version_info\s*[<>]=?\s*\((\d+),\s*(\d+)\)", text):
            found[f"{path.name}:{match.start()}"] = (int(match.group(1)), int(match.group(2)))
    return found


def _straddles(legs: list[str], boundary: tuple[int, ...]) -> bool:
    versions = [_version(leg) for leg in legs]
    return any(v < boundary for v in versions) and any(v >= boundary for v in versions)


def test_the_subset_sees_every_boundary_the_full_matrix_sees():
    """The subset must lose no version boundary the full matrix draws.

    Compared against the full matrix rather than in absolute terms: several boundaries
    (>= 3.10, < 3.14) sit outside the matrix range entirely, so no list here can see
    them, and holding the subset to a standard the full matrix does not meet would fail
    forever. What matters is the delta, which is what the trade is about.
    """
    subset, full = _matrices()
    boundaries = _boundaries()
    assert boundaries, "no version comparisons found in the backend; the scan is wrong"
    lost = {
        where: boundary
        for where, boundary in boundaries.items()
        if _straddles(full, boundary) and not _straddles(subset, boundary)
    }
    assert not lost, (
        f"the pull-request matrix {subset} lands entirely on one side of {lost} while the "
        f"full matrix {full} does not, so those branches stop being executed on both "
        f"sides before a merge. A static parse does not cover them: it parses, it does "
        f"not run."
    )


def test_the_declared_floor_is_still_checked_statically():
    """The static check is what replaced the older legs, so it has to still be there.

    Asserted through what it DOES rather than by filename: it must read the floor from
    pyproject.toml and hand it to ast.parse as feature_version, which is what makes it a
    floor check rather than a parse on whatever interpreter happens to run it.
    """
    assert FLOOR_CHECK.is_file(), (
        f"{FLOOR_CHECK.name} is gone. It is what covers the declared floor, which is below "
        f"every leg this matrix runs, so removing it leaves that floor untested."
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
    """Dropping the interior legs is only defensible because main runs everything."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") or {}
    assert "main" in (push.get("branches") or []), (
        "Backend CI no longer runs on push to main, so the interpreter versions dropped "
        "from the pull-request matrix would not be tested anywhere"
    )
