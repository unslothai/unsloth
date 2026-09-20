# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a superseded probe matrix costs has to be measured, not read off the timeout.

`runner-pool-probe.yml` carries a rationale for cancelling superseded runs, and that
rationale is what anyone deciding whether to keep the `concurrency:` block will act on.
It has already carried two wrong costs: billed minutes, which this repository does not
pay, and then "holding four of those five for up to ten minutes", which is the value of
`timeout-minutes:`. That value is the cutoff for a cell that hangs. A cell that works
holds its runner for about five seconds, because the whole job is one echo.

What is pinned here is deliberately NARROW, and the narrowness is the point. An earlier
version of this file tried to judge the claim as prose: which verb governed which
duration, whether a hanging qualifier was positive or denied, whether a subject had one
holder or two. Round after round of review found ways through it, in both directions,
and every repair opened new surface. Regular expressions do not read English, and a
guard that is wrong in the false-positive direction is worse than no guard: it fails CI
on a rationale that is true.

So two mechanical rules, neither of which needs a parser:

  1. The timeout value is never written as a prose duration. Refer to `timeout-minutes`,
     which is the value's actual home. Every wrong cost this comment has carried was a
     prose duration that happened to equal it, and a comment that never writes it cannot
     make that mistake. The cost is that a true sentence like "a hung cell holds its
     runner for ten minutes" is refused too; write "until the timeout" instead.

  2. The occupancy figures agree with the measurement beside them. In the sentence that
     reports the median, every figure is in seconds and within 3x of that median.

Neither rule catches an arbitrary false statement elsewhere in the comment, and this
file does not pretend to. It catches the two mistakes that were actually made.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "runner-pool-probe.yml"

_TIMEOUT = re.compile(r"^\s*timeout-minutes:\s*(\d+)\s*$", re.M)

# Spelled out as well as in digits: the claim that started this was in words.
_AS_A_WORD = {
    5: "five",
    10: "ten",
    15: "fifteen",
    20: "twenty",
    30: "thirty",
    60: "sixty",
}

# "median 4s" or "median 4 seconds": what the comment reports having observed.
_MEDIAN = re.compile(r"\bmedian\s+(?P<value>\d+)\s*(?:s\b|seconds?\b)", re.I)

_NUMBER = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"

# Any figure in seconds, however spelled.
_IN_SECONDS = re.compile(rf"(?<![\w-])(?P<value>{_NUMBER})\s*(?:s\b|seconds?\b)", re.I)

# Any figure in minutes or hours, which an occupancy sentence has no business carrying.
_COARSER = re.compile(
    rf"(?<![\w-])(?P<value>{_NUMBER})\s*(?P<unit>minutes?\b|mins?\b|hours?\b|hrs?\b)",
    re.I,
)

_AS_A_NUMBER = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _value(text: str) -> int:
    return int(text) if text.isdigit() else _AS_A_NUMBER[text.lower()]


def _rationale() -> str:
    """The header comment, which is every comment line above the first key.

    Read as text rather than through yaml: a comment is exactly what a parser drops, and
    the comment is the whole subject here.
    """
    lines = []
    for line in WORKFLOW.read_text(encoding = "utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            lines.append(stripped.lstrip("#").strip())
        elif stripped.startswith("concurrency:"):
            break
    return " ".join(lines)


def _timeout_minutes() -> int:
    found = _TIMEOUT.findall(WORKFLOW.read_text(encoding = "utf-8"))
    assert found, f"{WORKFLOW.name} no longer sets timeout-minutes; this guard is stale"
    return int(found[0])


def test_the_rationale_and_the_timeout_are_both_still_there():
    """A guard that found neither would pass every check below for the wrong reason."""
    rationale = _rationale()
    assert len(rationale) > 200, (
        f"{WORKFLOW.name} has no header comment left to check; either the rationale went "
        f"away, in which case the reason for the concurrency block went with it, or this "
        f"guard is reading the wrong thing"
    )
    assert _timeout_minutes() > 0


def test_the_timeout_value_is_never_written_as_a_prose_duration():
    """The rule with no exemptions, because every exemption turned out to be a way through.

    "for up to ten minutes" was the original wrong cost. "although the timeout is the
    cutoff for a hang, a superseded matrix holds its runners for ten minutes" was the
    next. "a non-hung cell holds its runner for ten minutes" was the one after that.
    They differ only as English; as a claim about what a working cell costs they are the
    same sentence, and what they have in common is writing the timeout's value out in
    prose.
    """
    minutes = _timeout_minutes()
    spellings = [str(minutes)]
    if minutes in _AS_A_WORD:
        spellings.append(_AS_A_WORD[minutes])
    written = re.compile(r"(?<![\w-])(?:" + "|".join(spellings) + r")\s+(?:minutes?|mins?)\b", re.I)
    found = [match.group(0) for match in written.finditer(_rationale())]
    assert not found, (
        f"{WORKFLOW.name} writes the timeout value as a duration: {found}. That value is "
        f"the cutoff for a cell that hangs, not what a working cell costs, and every "
        f"wrong cost this comment has carried was written exactly this way. Refer to "
        f"`timeout-minutes` instead, or say 'until the timeout'"
    )


def test_the_occupancy_figures_agree_with_the_measurement():
    """Read off the sentence that reports the median, not from anywhere near it."""
    rationale = _rationale()
    median = _MEDIAN.search(rationale)
    assert median, (
        f"{WORKFLOW.name} reports no median. Without a measurement in the comment the "
        f"timeout is the only number in reach, which is how the wrong one got quoted in "
        f"the first place"
    )
    opened = rationale.rfind(".", 0, median.start())
    closed = rationale.find(".", median.end())
    sentence = rationale[opened + 1 : closed if closed != -1 else len(rationale)].strip()
    observed = _value(median.group("value"))

    coarse = [match.group(0) for match in _COARSER.finditer(sentence)]
    assert not coarse, (
        f"the sentence reporting the median also gives a duration in minutes or hours: "
        f"{coarse} in {sentence!r}. A cell runs one echo; a coarser unit here is the "
        f"overstatement this guard exists to catch"
    )
    disagreeing = [
        match.group(0)
        for match in _IN_SECONDS.finditer(sentence)
        if not (
            observed <= _value(match.group("value")) * 3
            and _value(match.group("value")) <= observed * 3
        )
    ]
    assert not disagreeing, (
        f"the sentence reporting median {observed}s also states {disagreeing}, which "
        f"disagrees with it by more than 3x: {sentence!r}. An overstatement of the same "
        f"shape as the original fits comfortably inside the right unit"
    )


def test_the_timeout_is_still_explained_as_the_hung_cell_bound():
    """Rule 1 keeps the value out of prose; this keeps the reader told what it is for."""
    rationale = _rationale()
    assert "timeout-minutes" in rationale, (
        f"{WORKFLOW.name} no longer says what timeout-minutes is for, so the next reader "
        f"has nothing to stop them reading it as the expected cost again"
    )
    # After the key itself, and only to the end of its clause: `timeout-minutes` contains
    # "timeout", so reading from the start let the name of the thing stand in for the
    # explanation of it, and a wider window found negations belonging to other clauses.
    start = rationale.index("timeout-minutes") + len("timeout-minutes")
    stop = min(
        (offset for offset in (rationale.find(mark, start) for mark in (".", ";")) if offset != -1),
        default = len(rationale),
    )
    clause = rationale[start:stop]
    assert re.search(r"(?<![\w-])(?:hangs?|hanging|hung|cutoff)(?![\w-])", clause, re.I), (
        f"{WORKFLOW.name} mentions timeout-minutes without saying it bounds a cell that "
        f"hangs: {clause!r}"
    )
    assert not re.search(r"(?<![\w-])(?:not|never|non|no|without)(?![\w-])|n't", clause, re.I), (
        f"{WORKFLOW.name} denies that timeout-minutes bounds a hung cell: {clause!r}. "
        f"That is the invariant this test exists to keep, not one to talk out of"
    )
