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
    """The defect this replaces, in the exact words it had: "for up to ten minutes".

    Judged per OCCURRENCE and on what the occurrence is attached to, not per sentence.
    A sentence is the wrong unit twice over: "although the timeout is the cutoff for a
    hang, a superseded matrix holds its runners for ten minutes" contains the exempting
    words and reinstates the false claim anyway, and the words can sit far enough away to
    mean nothing about the number they excuse. So an occupancy verb next to the duration
    condemns it outright, and the cutoff reading only excuses a duration it is adjacent
    to.
    """
    minutes = _timeout_minutes()
    spellings = [str(minutes)]
    if minutes in _AS_A_WORD:
        spellings.append(_AS_A_WORD[minutes])
    duration = re.compile(r"\b(" + "|".join(spellings) + r")\s+minutes?\b", re.I)
    rationale = _rationale()
    offenders = []
    for match in duration.finditer(rationale):
        window = rationale[max(0, match.start() - 90) : match.end() + 40]
        # Attached to a verb of occupancy: that is the false claim, whatever else the
        # sentence concedes elsewhere.
        if re.search(r"\b(hold|holds|holding|held|occup\w*|tie[sd]? up)\b", window, re.I):
            offenders.append(window)
            continue
        # Otherwise it passes only if named, right here, as the cutoff it is.
        if not re.search(r"timeout|hang\w*|hung|cutoff", window, re.I):
            offenders.append(window)
    assert not offenders, (
        f"{WORKFLOW.name} gives {minutes} minutes as something a cell occupies, or as a "
        f"bare duration with no sign it is the timeout: {offenders}. That value is the "
        f"cutoff for a cell that hangs, not what a working cell holds, and quoting it as "
        f"the cost is what this guard exists to stop"
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


# "holds it for about five seconds": the verb, then the duration it governs. Parsed
# rather than searched for, because the unit ON THIS duration is the whole claim. A
# nearby seconds figure is not the same thing: "holds it for about five MINUTES ...
# median 4s" reinstates the overstatement while leaving every loose match satisfied.
_HOLD_CLAIM = re.compile(
    r"\bhold\w*\b[^.]{0,80}?\bfor\s+(?:about\s+|roughly\s+|around\s+|~\s*)?"
    r"(?P<value>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?P<unit>seconds?|s\b|minutes?|mins?\b|hours?|hrs?\b)",
    re.I,
)

_SECONDS = re.compile(r"^(seconds?|s)$", re.I)


def test_the_occupancy_claim_is_a_measured_one():
    """Seconds, and said to be measured, both read off the claim itself.

    Neither is checked over a window. The queue figures further down are also measured
    and also in seconds, so a window check passes with the occupancy claim removed
    entirely, and it passes with the claim changed to minutes while a stale `median 4s`
    sits behind it. Both of those were found by sabotaging this test, in that order.
    """
    rationale = _rationale()
    claim = _HOLD_CLAIM.search(rationale)
    assert claim, (
        f"{WORKFLOW.name} no longer says how long a cell holds its runner, in the form "
        f"'holds it for <duration>'. Without that the timeout is the only number in "
        f"reach, which is how the wrong one got quoted in the first place"
    )
    unit = claim.group("unit")
    assert _SECONDS.match(unit), (
        f"the occupancy claim is {claim.group('value')} {unit}: {claim.group(0)!r}. A "
        f"cell runs one echo and was measured at four, so anything but seconds here is "
        f"the overstatement this guard exists to catch"
    )
    # The provenance has to belong to this claim, so read to the end of its sentence.
    sentence_end = rationale.find(".", claim.end())
    sentence = rationale[claim.start() : sentence_end if sentence_end != -1 else len(rationale)]
    assert re.search(r"measured|median", sentence, re.I), (
        f"the occupancy claim has no sign it was observed: {sentence!r}. Both costs this "
        f"comment gave before were plausible numbers nobody had measured"
    )
