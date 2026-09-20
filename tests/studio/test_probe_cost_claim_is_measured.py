# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a superseded probe matrix costs has to be measured, not read off the timeout.

`runner-pool-probe.yml` carries a rationale for cancelling superseded runs, and that
rationale is the thing anyone deciding whether to keep the `concurrency:` block will act
on. It first justified the cancel with billed minutes, which this repository does not pay,
and then with "holding four of those five for up to ten minutes", which is the value of
`timeout-minutes:`. That number is the cutoff for a cell that hangs. A cell that works
holds its runner for about five seconds, because the whole job is one echo, so quoting the
timeout overstated the normal case by two orders of magnitude.

Both mistakes have the same shape: a plausible number that was never measured. So what is
pinned here is not a particular wording but that the claim stays grounded -- the timeout is
only ever cited as the hung-cell bound, and the occupancy figure is a measured one.

This repo already treats an untrue workflow comment as a defect worth a test; see
`test_no_workflow_claims_a_main_protection_it_does_not_have`.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "runner-pool-probe.yml"

_TIMEOUT = re.compile(r"^\s*timeout-minutes:\s*(\d+)\s*$", re.M)

# Spelled out as well as in digits, because the claim that motivated this was in words.
_AS_A_WORD = {
    5: "five",
    10: "ten",
    15: "fifteen",
    20: "twenty",
    30: "thirty",
    60: "sixty",
}


def _rationale() -> str:
    """The header comment, which is everything above the first key that is not a comment.

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


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.:]) ", text) if part.strip()]


def test_the_rationale_and_the_timeout_are_both_still_there():
    """A guard that found neither would pass every check below for the wrong reason."""
    rationale = _rationale()
    assert len(rationale) > 200, (
        f"{WORKFLOW.name} has no header comment left to check; either the rationale went "
        f"away, in which case the reason for the concurrency block went with it, or this "
        f"guard is reading the wrong thing"
    )
    assert _timeout_minutes() > 0


def test_the_timeout_is_never_offered_as_what_a_superseded_matrix_holds():
    """The defect this replaces, in the exact words it had: "for up to ten minutes"."""
    minutes = _timeout_minutes()
    spellings = [str(minutes)]
    if minutes in _AS_A_WORD:
        spellings.append(_AS_A_WORD[minutes])
    duration = re.compile(r"\b(" + "|".join(spellings) + r")\s+minutes?\b", re.I)
    offenders = []
    for sentence in _sentences(_rationale()):
        if not duration.search(sentence):
            continue
        # Cited as the cutoff it is, rather than as an occupancy, is fine.
        if re.search(r"timeout|hang|hung|cutoff", sentence, re.I):
            continue
        offenders.append(sentence)
    assert not offenders, (
        f"{WORKFLOW.name} states {minutes} minutes as a duration without saying it is the "
        f"timeout: {offenders}. That value is the cutoff for a cell that hangs, not what a "
        f"working cell holds, and quoting it as the cost is what this guard exists to stop"
    )


def test_the_timeout_is_still_explained_as_the_hung_cell_bound():
    """Deleting the sentence would pass the check above by saying nothing at all."""
    rationale = _rationale()
    assert "timeout-minutes" in rationale, (
        f"{WORKFLOW.name} no longer says what timeout-minutes is for, so the next reader "
        f"has nothing to stop them reading it as the expected cost again"
    )
    window = rationale[rationale.index("timeout-minutes") :][:400]
    assert re.search(r"hang|hung|cutoff", window, re.I), (
        f"{WORKFLOW.name} mentions timeout-minutes without saying it bounds a cell that "
        f"hangs: {window!r}"
    )


def test_the_occupancy_claim_is_a_measured_one():
    """Seconds, and said to be measured, in the sentence that makes the claim.

    Read from where the rationale says what a cell HOLDS, not from the comment as a
    whole: the queue figures further down are also measured and also in seconds, so a
    check over the whole text passes with the occupancy claim removed entirely. That is
    how this test first passed its own sabotage.
    """
    rationale = _rationale()
    holds = re.search(r"\bhold(s|ing)?\b", rationale, re.I)
    assert holds, (
        f"{WORKFLOW.name} no longer says how long a cell holds its runner. Without that "
        f"the timeout is the only number in reach, which is how the wrong one got quoted "
        f"in the first place"
    )
    claim = rationale[holds.start() : holds.start() + 300]
    assert re.search(r"\b(\d+\s*s\b|\d+\s*seconds?\b|five seconds)", claim, re.I), (
        f"the occupancy claim gives no seconds-scale figure: {claim!r}. A cell runs one "
        f"echo, so this is the number that keeps the timeout from being read as the cost"
    )
    assert re.search(r"measured|median", claim, re.I), (
        f"the occupancy claim has no sign it was observed: {claim!r}. Both costs this "
        f"comment gave before were plausible numbers nobody had measured"
    )
