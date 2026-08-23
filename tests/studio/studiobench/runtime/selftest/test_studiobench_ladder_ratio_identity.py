# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One payload, one chars-per-token ratio.

A rung is NAMED in tokens and BUILT in characters, and the ratio is the only thing that makes
those the same claim. `session.ladder_chars_per_token` measures it afresh on every invocation,
`--resume` included, and falls back to `PROVISIONAL_CHARS_PER_TOKEN` whenever no real tokeniser
answers -- tiktoken is not a dependency of this harness, and where it is installed `get_encoding`
fetches `cl100k_base` over the network on first use, so "no tokeniser answered" is one absent
package or one unlucky minute.

So a run on a machine with no tokeniser sizes every rung at 4.0 and dies at 100K, and the resume
finds tiktoken and sizes at 3.336. `_resume_set` skips by `cell_id`, which is
`r{rung}.{arm}.rep{rep}` and carries no ratio: 1K and 10K stay at 4,000 and 40,000 characters
while 100K and 1M are built at 333,600 and 3,336,000. Nothing downstream can see the mixture --
`score_payload` keys by rung, the report prints one ladder, and ONSET RUNG names a token label
standing over two different amounts of work in the same table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import (  # noqa: E402
    LADDER_RATIO_AXIS,
    LADDER_RATIO_TOLERANCE,
    _resume_set,
    ladder_ratio_problems,
    recorded_identities,
)
from studiobench.runtime.types import Paths, Recorder  # noqa: E402

CORPUS = "c0ffee"


def _ratio(value: float, source: str, provisional: bool) -> dict:
    """A `ladder_chars_per_token` block exactly as `session.build_cells` writes one."""
    return {
        "chars_per_token": value,
        "measured": None if provisional else value,
        "source": source,
        "provisional": provisional,
        "reason": "no tokeniser answered" if provisional else None,
    }


def _payload(tmp_path, cells: list, session: str = "sess-1") -> Paths:
    """A payload holding `cells`, written through the REAL recorder."""
    paths = Paths.under(tmp_path / "out")
    recorder = Recorder(paths.payload_jsonl, session)
    recorder.emit(
        {
            "row_type": "run_meta",
            "tier": "standard",
            "tool_version": "0.1.0",
            "corpus_hash": CORPUS,
            "studio_ref": "main",
            "bundle": {"production": True},
            "platform": {"system": "Linux"},
            "started_at": "2026-01-01T00:00:00",
            "cadence": "field",
            "rungs": ["1K", "10K", "100K"],
            "reps": 1,
            "instrument_level": 0,
        }
    )
    for row in cells:
        recorder.emit(dict(row))
    recorder.close()
    return paths


def _cell(rung: str, meta: dict) -> dict:
    return {
        "row_type": "cell",
        "cell_id": f"r{rung}.A0.rep0",
        "cell": {"arm": "A0", "rep": 0, "tier": "standard", "meta": meta},
        "completed": True,
        "fidelity": "ok",
    }


def _problems(paths: Paths, measured: float) -> list:
    """What `run` asks once `build_cells` has measured this invocation's ratio."""
    out: list = []
    for recorded in recorded_identities(paths.payload_jsonl):
        for problem in ladder_ratio_problems(recorded, measured):
            if problem not in out:
                out.append(problem)
    return out


def test_a_resume_that_sizes_the_ladder_differently_is_refused(tmp_path):
    """The bug, as the numbers it produced.

    The recorded run had no tokeniser and planned 1K at 4,000 characters; this one reads the corpus
    at 3.336 and would plan 100K at 333,600 rather than 400,000. `_resume_set` skips the first two
    rungs regardless, because the ratio is nowhere in a cell id.
    """

    paths = _payload(
        tmp_path,
        [
            _cell(rung, {"ladder_chars_per_token": _ratio(4.0, "whitespace estimate", True)})
            for rung in ("1K", "10K")
        ],
    )
    assert _resume_set(paths) == {"r1K.A0.rep0", "r10K.A0.rep0"}

    problems = _problems(paths, 3.336)
    assert len(problems) == 1, problems
    assert LADDER_RATIO_AXIS in problems[0]
    assert "4.0" in problems[0] and "3.336" in problems[0]


def test_the_ratio_is_read_off_the_cells_a_session_actually_wrote(tmp_path):
    paths = _payload(
        tmp_path,
        [_cell("1K", {"ladder_chars_per_token": _ratio(3.336, "tiktoken/cl100k", False)})],
    )
    (recorded,) = recorded_identities(paths.payload_jsonl)
    assert recorded[LADDER_RATIO_AXIS] == 3.336


def test_the_same_ratio_still_resumes(tmp_path):
    """The control. A tokeniser that answers the same way twice is a continuation, not a change."""

    paths = _payload(
        tmp_path,
        [_cell("1K", {"ladder_chars_per_token": _ratio(3.336, "tiktoken/cl100k", False)})],
    )
    assert _problems(paths, 3.336) == []


def test_a_float_that_only_differs_in_the_noise_still_resumes(tmp_path):
    """5.0 against 5.0000001 is one measurement read twice, not two measurements.

    `measure_chars_per_token` rounds to three decimals and `PROVISIONAL_CHARS_PER_TOKEN` is exact,
    so every ratio that reaches a payload sits on a 0.001 grid and `LADDER_RATIO_TOLERANCE` is half
    a grid step: nothing a float's own representation can do to a value crosses it, and nothing two
    real measurements can differ by fails to.
    """

    paths = _payload(
        tmp_path,
        [_cell("1K", {"ladder_chars_per_token": _ratio(5.0, "tiktoken/cl100k", False)})],
    )
    assert _problems(paths, 5.0 + 1e-7) == []
    assert _problems(paths, 5.0 - 1e-7) == []
    # And the smallest difference two recorded ratios can actually have is still refused.
    assert _problems(paths, 5.001) != []
    assert LADDER_RATIO_TOLERANCE < 0.001


def test_a_payload_that_never_recorded_the_ratio_still_resumes(tmp_path):
    """A payload written before the ratio travelled on `meta` declares no ratio.

    The same rule `recorded_identities` applies to every other axis: an axis a row never declared
    cannot be a difference, so an older output resumes on the axes it did record.
    """

    paths = _payload(tmp_path, [_cell("1K", {})])
    (recorded,) = recorded_identities(paths.payload_jsonl)
    assert LADDER_RATIO_AXIS not in recorded
    assert _problems(paths, 3.336) == []
    assert ladder_ratio_problems({}, 3.336) == []


def test_a_session_that_recorded_no_cell_declares_no_ratio(tmp_path):
    """A run that died before its first cell measured nothing, so it pins nothing."""

    paths = _payload(tmp_path, [])
    assert _problems(paths, 3.336) == []


@pytest.mark.parametrize("measured", [None])
def test_an_unmeasurable_ratio_is_not_a_refusal(tmp_path, measured):
    """`chars_per_token` is null only when there was no text to measure. Nothing to compare."""

    paths = _payload(
        tmp_path,
        [_cell("1K", {"ladder_chars_per_token": _ratio(4.0, "whitespace estimate", True)})],
    )
    assert _problems(paths, measured) == []
