# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Backend CI matrix runs a subset on pull requests, and that subset has to mean something.

Measured on one runner over the same tree, the four interpreter legs collect the same
26,320 tests and differ by exactly one: ``test_demonstrates_the_underlying_stdlib_regression``
is gated on ``sys.version_info >= (3, 12)``, so 3.10 and 3.11 report 26193 passed / 127
skipped and 3.12 and 3.13 report 26194 / 126. A subset that lands entirely on one side of
that gate would drop the only distinction the matrix draws, while still costing two legs.

So the subset is checked here rather than trusted: it keeps the ends of the full matrix,
and it spans every version gate the backend tests actually declare.
"""

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"
BACKEND_TESTS = REPO / "studio" / "backend" / "tests"


def _matrices() -> tuple[list[str], list[str]]:
    """The pull-request subset and the full list, read out of the matrix expression."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    expression = str(document["jobs"]["pytest"]["strategy"]["matrix"]["python"])
    lists = re.findall(r"fromJSON\('(\[[^']*\])'\)", expression)
    assert len(lists) == 2, f"expected a pull-request list and a full list, found {lists}"
    return tuple(ast.literal_eval(item) for item in lists)  # type: ignore[return-value]


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_the_pull_request_subset_keeps_both_ends_of_the_full_matrix():
    """Dropping an interior version is the trade; dropping a boundary is not.

    The floor is what catches syntax and stdlib use that does not exist yet, the ceiling
    is what catches removals and deprecations. A subset missing either stops testing the
    thing the matrix is for.
    """
    subset, full = _matrices()
    assert set(subset) <= set(full), f"{subset} is not a subset of {full}"
    assert min(subset, key = _version) == min(full, key = _version)
    assert max(subset, key = _version) == max(full, key = _version)


def _straddled(legs: list[str], gate: tuple[int, ...]) -> bool:
    versions = [_version(leg) for leg in legs]
    return any(v < gate for v in versions) and any(v >= gate for v in versions)


def test_the_subset_sees_every_version_gate_the_full_matrix_sees():
    """The subset must lose no distinction the full matrix draws.

    Not "straddle every gate": ``test_llama_admission.py`` gates on >= 3.10 and the full
    matrix starts AT 3.10, so no leg sits below it and neither list can see that one.
    Holding the subset to a standard the full matrix does not meet would just fail here
    forever. What matters is the delta, which is what the trade is about.

    Read from the tests rather than hardcoded, so a new gate that the subset straddles
    poorly fails here instead of quietly going untested on pull requests.
    """
    subset, full = _matrices()
    gates: dict[str, tuple[int, ...]] = {}
    for path in sorted(BACKEND_TESTS.glob("test_*.py")):
        text = path.read_text(encoding = "utf-8", errors = "replace")
        for match in re.finditer(r"version_info\s*[<>]=?\s*\((\d+),\s*(\d+)\)", text):
            gates[f"{path.name}:{match.start()}"] = (int(match.group(1)), int(match.group(2)))
    assert gates, "no version gates found; the scan is wrong, not the matrix"
    lost = {
        where: gate
        for where, gate in gates.items()
        if _straddled(full, gate) and not _straddled(subset, gate)
    }
    assert not lost, (
        f"the pull-request matrix {subset} lands entirely on one side of {lost} while the "
        f"full matrix {full} does not, so those tests stop being compared across the "
        f"boundary they exist to describe. Add a leg on the other side of the gate."
    )


def test_the_full_matrix_still_runs_somewhere():
    """The subset is only defensible because main runs everything."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") or {}
    assert "main" in (push.get("branches") or []), (
        "Backend CI no longer runs on push to main, so the interior interpreter versions "
        "dropped from the pull-request matrix would not be tested at all"
    )
