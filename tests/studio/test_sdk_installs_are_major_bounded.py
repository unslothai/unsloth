# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SDKs the probes call by name must not float across a major.

On 2026-08-20 `anthropic` 1.0.0 was published. The inference smoke workflows installed
`anthropic>=0.40` with no upper bound, pip resolved the new major, and v1 had removed
`temperature`, `top_p` and `top_k` as accepted arguments on `messages.create()`. Every
probe that pins `temperature = 0.0` for determinism started raising:

    TypeError: Messages.create() got an unexpected keyword argument 'temperature'

75 job failures in about four hours, across every PR that ran those workflows, none of
them caused by the PR they were reported against. #9432 pinned `<1` to stop it.

This guard is deliberately NOT "every `>=` needs an upper bound". Most pins here are
libraries whose majors do not reach our code, and asserting on all of them would be noise
that gets suppressed. It covers the packages whose *call surface* our own probe scripts
use directly with keyword arguments, which is exactly the surface a major is allowed to
remove. Adding to this set needs that property; removing from it needs a reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Packages our probe code calls directly by keyword. A new major may drop a keyword
# without any change on our side, so the version we test against has to be a decision.
GUARDED = ("anthropic", "openai", "playwright")

# 'name>=1.2' / "name>=1.2,<2" / bare name>=1.2, as they appear inside a `pip install`.
_SPEC = re.compile(
    r"""(?P<quote>['"]?)(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"""
    r"""(?P<spec>(?:[<>=!~]=?[^,'"\s]+)(?:,[<>=!~]=?[^,'"\s]+)*)(?P=quote)"""
)


def _install_lines() -> list[tuple[Path, int, str]]:
    """Every `pip install` line in every workflow, minus commented-out ones.

    A YAML/shell comment that merely mentions a package must not be able to satisfy or
    trip this guard -- #9432 added comments naming `anthropic` right beside the pins.
    """
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for number, line in enumerate(path.read_text(encoding = "utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "pip install" in line:
                found.append((path, number, line))
    return found


def _specs_for(package: str) -> list[tuple[Path, int, str]]:
    hits = []
    for path, number, line in _install_lines():
        for match in _SPEC.finditer(line):
            if match.group("name").lower().replace("_", "-") == package:
                hits.append((path, number, match.group("spec")))
    return hits


@pytest.mark.parametrize("package", GUARDED)
def test_the_sdk_is_pinned_below_the_next_major(package: str) -> None:
    unbounded = [
        (path, number, spec) for path, number, spec in _specs_for(package) if "<" not in spec
    ]
    assert not unbounded, (
        f"{package} is installed without an upper bound at "
        + ", ".join(f"{p.name}:{n} ({s})" for p, n, s in unbounded)
        + f". A new {package} major would reach CI with no change on our side; that is how"
        " anthropic 1.0.0 broke 75 jobs. Pin the major you mean to test against."
    )


def test_the_guard_is_reading_real_pins() -> None:
    """An empty scan would make every assertion above pass silently.

    If the workflows stop installing these, this test is the one that says so, rather
    than the suite going quietly green while it checks nothing.
    """
    for package in GUARDED:
        assert _specs_for(package), (
            f"no `pip install` line in .github/workflows pins {package} any more; either"
            " the probes moved or the regex stopped matching, and the guard above is now"
            " vacuous"
        )


def test_a_commented_out_pin_is_not_mistaken_for_an_install() -> None:
    lines = _install_lines()
    assert lines, "no pip install lines found at all"
    assert all(not line.lstrip().startswith("#") for _, _, line in lines)
