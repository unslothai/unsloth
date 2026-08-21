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

# Packages our probe code calls directly by keyword, each with the first major it must
# NOT reach. A pin has to exclude that major; anything at or below it is fine, so the
# narrower `playwright>=1.55,<1.58` window in one workflow still satisfies this.
#
# Asserting the boundary rather than "some upper bound exists" is what makes the guard
# mean anything: `openai>=1.50,<999` contains a `<` and still admits every major this
# exists to keep out.
#
# Moving one of these is a deliberate act. Bump the number here in the same commit that
# widens the pin, and only after CI has actually run against that major.
GUARDED = {
    # v1 removed temperature, top_p and top_k from messages.create().
    "anthropic": (1,),
    # 3.3.1 resolves today and the probes pass, so the bound sits above it, not at <2.
    "openai": (4,),
    # Still 1.x upstream.
    "playwright": (2,),
}

# 'name>=1.2' / "name>=1.2,<2" / bare name>=1.2, as they appear inside a `pip install`.
_SPEC = re.compile(
    r"""(?P<quote>['"]?)(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"""
    r"""(?P<spec>(?:[<>=!~]=?[^,'"\s]+)(?:,[<>=!~]=?[^,'"\s]+)*)(?P=quote)"""
)

_UPPER = re.compile(r"(?P<op><=|<)(?P<version>[0-9][0-9.]*)")


def _install_commands_in(text: str) -> list[tuple[int, str]]:
    """Every `pip install` COMMAND in a workflow, continuations joined, comments dropped.

    Joining matters: `pip install \\` over several lines is the house style at 19 sites
    here, and a line-at-a-time scan sees only the first one. A guarded SDK added on a
    continuation line would then be invisible while the anti-vacuity test below stayed
    satisfied by the single-line pins -- the guard would pass and check nothing.

    Comments are dropped because #9432 added comments naming `anthropic` right beside
    the pins, and a comment must not be able to satisfy or trip this either way.
    """
    commands = []
    pending: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if not pending:
            start = number
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            pending.append(stripped[:-1])
            continue
        pending.append(stripped)
        joined = " ".join(pending)
        pending = []
        if "pip install" in joined:
            commands.append((start, joined))
    if pending:
        joined = " ".join(pending)
        if "pip install" in joined:
            commands.append((start, joined))
    return commands


def _install_lines() -> list[tuple[Path, int, str]]:
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for number, command in _install_commands_in(path.read_text(encoding = "utf-8")):
            found.append((path, number, command))
    return found


def _specs_for(package: str) -> list[tuple[Path, int, str]]:
    hits = []
    for path, number, line in _install_lines():
        for match in _SPEC.finditer(line):
            if match.group("name").lower().replace("_", "-") == package:
                hits.append((path, number, match.group("spec")))
    return hits


def _version(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.strip(".").split(".") if part.isdigit())


def _excludes(spec: str, major: tuple[int, ...]) -> bool:
    """Does this specifier keep `major` out?

    `<2` and `<1.58` both exclude 2.x; `<=2` does not, since it admits 2.0 itself.
    Compared as integer tuples rather than through `packaging`, which the
    workflow-trigger-lint job does not install.
    """
    for match in _UPPER.finditer(spec):
        bound = _version(match.group("version"))
        if not bound:
            continue
        if bound <= major if match.group("op") == "<" else bound < major:
            return True
    return False


@pytest.mark.parametrize("package", GUARDED)
def test_the_sdk_is_pinned_below_the_next_major(package: str) -> None:
    major = GUARDED[package]
    wanted = ".".join(str(part) for part in major)
    admits = [
        (path, number, spec)
        for path, number, spec in _specs_for(package)
        if not _excludes(spec, major)
    ]
    assert not admits, (
        f"{package} is installed in a way that admits {wanted} at "
        + ", ".join(f"{p.name}:{n} ({s})" for p, n, s in admits)
        + f". A new {package} major would reach CI with no change on our side; that is how"
        f" anthropic 1.0.0 broke 75 jobs. Pin below {wanted}, or move the boundary in this"
        " file in the same commit once CI has run against that major."
    )


def test_a_bound_above_the_next_major_is_not_accepted() -> None:
    """`<` alone is not the property; keeping the major out is.

    Checking only for the presence of an upper bound let `openai>=1.50,<999` through,
    which admits every major the pin exists to exclude.
    """
    assert not _excludes(">=1.50,<999", GUARDED["openai"])
    assert not _excludes(">=1.50,<5", GUARDED["openai"])
    assert not _excludes(">=1.50", GUARDED["openai"])
    # <=4 admits 4.0 itself, so it does not exclude the 4 series.
    assert not _excludes(">=1.50,<=4", GUARDED["openai"])
    assert _excludes(">=1.50,<4", GUARDED["openai"])
    # A narrower window than the boundary is stricter, and still fine.
    assert _excludes(">=1.55,<1.58", GUARDED["playwright"])
    assert _excludes(">=1.45,<2", GUARDED["playwright"])


def test_a_pin_on_a_continuation_line_is_still_seen() -> None:
    """`pip install \\` over several lines is the house style at 19 sites here.

    A line-at-a-time scan sees only the first line, so a guarded SDK added below it would
    be invisible while the anti-vacuity test stayed satisfied by the single-line pins.
    """
    text = (
        "      - name: Install\n"
        "        run: |\n"
        "          pip install 'pytest>=8' \\\n"
        "            'openai>=1.50' \\\n"
        "            'anthropic>=0.40,<1'\n"
    )
    commands = _install_commands_in(text)
    assert len(commands) == 1, commands
    number, command = commands[0]
    assert number == 3, "the command should be reported at the line it starts on"
    assert "openai>=1.50" in command and "anthropic>=0.40,<1" in command


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
