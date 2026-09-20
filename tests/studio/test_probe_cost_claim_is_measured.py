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


# "holds it for about five seconds": an occupancy verb, then the duration it GOVERNS.
# Parsed rather than searched for, and used by both tests below, because proximity
# answers neither question they ask. The unit on this duration is the whole occupancy
# claim, and "holds" sitting near a number does not mean it governs that number: in
# "a cell holds a runner for five seconds; timeout-minutes is a cutoff at ten minutes if
# hung" the verb governs the five, and rejecting the ten would fail CI on a true comment.
#
# The value is matched as ANY word, not against a list of numbers this file knows. A
# list is a spelling contest the comment wins: "for fifteen minutes" walked straight past
# a pattern that stopped at ten, while the test claimed to validate every claim. So the
# FORM is recognised here and the number is parsed separately, and a duration that cannot
# be parsed fails rather than going unseen.
_HOLD_CLAIM = re.compile(
    r"\b(?:hold\w*|held|occup\w*|tie[sd]?\s+up)\b[^.;]{0,80}?\bfor\s+"
    r"(?:about\s+|roughly\s+|around\s+|nearly\s+|almost\s+|approximately\s+|"
    r"up\s+to\s+|at\s+least\s+|at\s+most\s+|no\s+more\s+than\s+|over\s+|under\s+|"
    r"the\s+|full\s+|entire\s+|whole\s+|"
    r"as\s+(?:much|long)\s+as\s+|~\s*)*"
    r"(?P<value>[\w-]+(?:\s+[\w-]+){0,2}?)\s*"
    r"(?P<unit>seconds?|s\b|minutes?|mins?\b|hours?|hrs?\b)",
    re.I,
)

_SECONDS = re.compile(r"^(seconds?|s)$", re.I)

# What the comment reports having observed, which is what the claim has to agree with.
_MEASURED = re.compile(r"\bmedian\s+(?P<value>\d+)\s*(?:s\b|seconds?\b)", re.I)

_AS_A_NUMBER = {
    "a": 1,
    "an": 1,
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
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "ninety": 90,
}


# "six hundred seconds" is one number in three words, and a value matched as a single
# token produced no claim at all rather than the unreadable-value failure the guard
# promises. So the value is several tokens and they are summed here.
_MULTIPLIERS = {"hundred": 100, "thousand": 1000}


def _value(text: str) -> int | None:
    """None where the number cannot be read, which every caller treats as a failure."""
    total = 0
    read = False
    for token in re.split(r"[\s-]+", text.strip().lower()):
        if not token:
            continue
        if token.isdigit():
            total += int(token)
        elif token in _AS_A_NUMBER:
            total += _AS_A_NUMBER[token]
        elif token in _MULTIPLIERS:
            total = (total or 1) * _MULTIPLIERS[token]
        else:
            return None
        read = True
    return total if read else None


def _clause_before(text: str, position: int) -> str:
    """What sits between the last clause boundary and `position`.

    The subject of the verb, in other words, and the only place a hung-cell
    qualification can legitimately be. "A cell that hangs holds its runner for ten
    minutes" is true and has to stay legal; "although the timeout is the cutoff for a
    hang, a superseded matrix holds its runners for ten minutes" is the same false claim
    it always was, and the difference between them is exactly which clause the
    qualification is in.
    """
    boundary = max(text.rfind(mark, 0, position) for mark in (".", ";", ",", ":"))
    return text[boundary + 1 : position]


# A POSITIVE hanging qualifier. Hyphens and the surrounding words both matter: "a
# non-hung cell holds its runner for ten minutes" contains the substring `hung` and is
# the exact claim this guard exists to reject, so the boundary excludes a preceding
# hyphen and an explicit negation disqualifies the clause outright.
_HANGS = re.compile(r"(?<![\w-])(?:hangs?|hanging|hung)(?![\w-])", re.I)
_NEGATED = re.compile(r"(?<![\w-])(?:not|never|non|no|without)(?![\w-])|n't", re.I)

# A second holder in the same subject. "One hanging cell and one working cell each hold a
# runner for ten minutes" says the false thing about the working one, and a qualifier that
# covers only part of a subject cannot excuse the whole claim.
_ANOTHER_HOLDER = re.compile(
    r"(?<![\w-])(?:and|or|each|both|either|working|healthy|normal|successful)(?![\w-])",
    re.I,
)


def _about_a_hung_cell(text: str, position: int) -> bool:
    """Is the thing doing the holding a cell that hangs, said positively?"""
    clause = _clause_before(text, position)
    hangs = _HANGS.search(clause)
    if not hangs:
        return False
    if _NEGATED.search(clause[: hangs.start()]):
        return False
    # The qualifier has to cover the whole subject, not one member of a list.
    return not _ANOTHER_HOLDER.search(clause)


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

    Judged per OCCURRENCE, on what GOVERNS it. A sentence is the wrong unit: "although
    the timeout is the cutoff for a hang, a superseded matrix holds its runners for ten
    minutes" contains the exempting words and reinstates the false claim anyway. But
    proximity is the wrong unit too, in the other direction: in "a cell holds a runner
    for five seconds; timeout-minutes is a cutoff at ten minutes if hung" the verb
    governs the five, and condemning the ten because "holds" is nearby would fail CI on
    a comment that is entirely true.

    So the durations an occupancy verb actually governs are parsed out, and only those
    are condemned. Any other mention of the timeout value still has to be named as the
    cutoff where it stands.
    """
    minutes = _timeout_minutes()
    spellings = [str(minutes)]
    if minutes in _AS_A_WORD:
        spellings.append(_AS_A_WORD[minutes])
    duration = re.compile(r"\b(" + "|".join(spellings) + r")\s+minutes?\b", re.I)
    rationale = _rationale()
    # The durations an occupancy verb actually governs, by where their number starts.
    occupied = {claim.start("value"): claim.start() for claim in _HOLD_CLAIM.finditer(rationale)}
    offenders = []
    for match in duration.finditer(rationale):
        window = rationale[max(0, match.start() - 90) : match.end() + 40]
        # Governed by a verb of occupancy: that is the false claim, whatever else the
        # sentence concedes elsewhere. Unless the thing doing the holding is a cell that
        # HANGS, which really does hold its runner until the timeout.
        if match.start() in occupied:
            if not _about_a_hung_cell(rationale, occupied[match.start()]):
                offenders.append(window)
                continue
        # Otherwise it passes only if ITS OWN clause names it as the cutoff. A window
        # lets a bare claim borrow the qualification from the clause before it: "A
        # superseded matrix costs ten minutes" placed after the accurate timeout
        # sentence is not an occupancy verb, so it lands here, and it read as excused.
        if not re.search(
            r"timeout|hang\w*|hung|cutoff", _clause_before(rationale, match.start()), re.I
        ):
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


def _sentence_around(text: str, start: int, end: int) -> str:
    """The claim's own sentence, so provenance cannot be borrowed from a neighbour."""
    opened = text.rfind(".", 0, start)
    closed = text.find(".", end)
    return text[opened + 1 : closed if closed != -1 else len(text)].strip()


def test_every_occupancy_claim_is_a_measured_one():
    """Seconds, and consistent with the measurement, for EVERY claim rather than the first.

    Checking only the first lets a correct opening sentence mask anything added later:
    "A superseded matrix holds its runners for about 600 seconds" further down passed
    while the measured sentence above it stayed intact, and the timeout test does not see
    it either, because 600 seconds is not the configured ten minutes.

    Nothing here is checked over a window. The queue figures further down are also
    measured and also in seconds, so a window check passes with the occupancy claim
    removed entirely, and it passes with the claim changed to minutes while a stale
    `median 4s` sits behind it. Seconds alone is not enough either: "about 600 seconds"
    is the original overstatement in the right unit.

    So the comment has to report a median, at least one claim has to carry that median in
    its own sentence, and every claim has to agree with it within 3x. Restating the figure
    elsewhere without repeating the provenance stays legal, which it should: what is being
    guarded against is an unmeasured number, not a second mention of a measured one.

    A claim about the HUNG case is exempt, since "a cell that hangs holds its runner for
    ten minutes" is true and is the timeout test's subject rather than this one's.
    """
    rationale = _rationale()
    claims = [
        claim
        for claim in _HOLD_CLAIM.finditer(rationale)
        if not _about_a_hung_cell(rationale, claim.start())
    ]
    assert claims, (
        f"{WORKFLOW.name} no longer says how long a WORKING cell holds its runner, in the "
        f"form 'holds it for <duration>'. Without that the timeout is the only occupancy "
        f"figure in the comment, which is the reading this whole guard exists to prevent"
    )
    # The median has to come from a sentence that MAKES a claim, not from anywhere in the
    # comment. A queue figure reading "median 500s" elsewhere would otherwise become the
    # yardstick, and a 500-second occupancy claim sitting beside its own "median 4s" would
    # measure as consistent.
    grounded = [
        (claim, _MEASURED.search(_sentence_around(rationale, claim.start(), claim.end())))
        for claim in claims
    ]
    grounded = [(claim, found) for claim, found in grounded if found]
    assert grounded, (
        f"no occupancy claim carries a median in its own sentence: "
        f"{[claim.group(0) for claim in claims]}. Saying a figure was measured without "
        f"giving the measurement beside it leaves nothing for the claim to be checked "
        f"against, which is how 'about five seconds' could have read 600 and still passed"
    )
    measured = grounded[0][1]
    observed = _value(measured.group("value"))
    assert observed is not None, f"unreadable median: {measured.group(0)!r}"
    for claim in claims:
        claimed = _value(claim.group("value"))
        assert claimed is not None, (
            f"this occupancy claim states a duration this guard cannot read: "
            f"{claim.group(0)!r}. Spelling the number out is how 'for fifteen minutes' "
            f"once walked past a list that stopped at ten, so an unreadable duration "
            f"fails here rather than being skipped. Use digits, or add the word to "
            f"_AS_A_NUMBER"
        )
        unit = claim.group("unit")
        assert _SECONDS.match(unit), (
            f"an occupancy claim is {claim.group('value')} {unit}: {claim.group(0)!r}. A "
            f"cell runs one echo and was measured at {observed}s, so anything but seconds "
            f"here is the overstatement this guard exists to catch"
        )
        assert observed <= claimed * 3 and claimed <= observed * 3, (
            f"an occupancy claim says {claimed}s where the comment reports measuring "
            f"{observed}s: {claim.group(0)!r}. Seconds is not enough on its own -- an "
            f"overstatement of the same shape as the original fits inside the unit"
        )
