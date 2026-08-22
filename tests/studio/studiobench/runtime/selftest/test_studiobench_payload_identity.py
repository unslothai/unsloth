# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One `--out` holds one run, and `--resume` continues the run that is in it.

A cell id is `r{rung}.{arm}.rep{rep}` and nothing more, so the tier that picked the film, the
cadence the reply streamed at, the instrument level, the corpus and both refs are all invisible to
the two places that decide what a payload already contains. Two ways that produced a report about
something nobody ran:

  A RESUME UNDER A DIFFERENT CONFIGURATION SKIPPED EVERYTHING. `--branch main --ab other --resume`
  into a directory holding a finished `main -> fix` run installs and launches two Studios, finds
  every `cell_id` already complete, exits 0, and -- because `_render_ab` keeps the previous table
  when this session measured nothing -- leaves the OLD comparison standing in `ab.md` to be read
  as the answer for `other`.

  A FRESH RUN APPENDED TO THE PREVIOUS ONE. `Recorder` opens the payload with `"a"`. The README's
  own two invocations wrote into one `--out`, and when the second was interrupted the file held a
  fast-tier 100K cell from an attached Studio next to standard-tier 1K and 10K cells from `main`.
  `--report` scored that as one ladder, took its header from the FIRST `run_meta`, and said
  `complete: true`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import (  # noqa: E402
    _resume_set,
    archive_payload,
    parse_args,
    prepare_payload,
    requested_identity,
)
from studiobench.report.build import build_report  # noqa: E402
from studiobench.runtime.types import Paths, Recorder  # noqa: E402
from studiobench.scoring.from_payload import (  # noqa: E402
    latest_attempt_rows,
    measures_from_records,
)

CORPUS = "c0ffee"


def _run_meta(tier, ref, rungs, **extra):
    row = {
        "row_type": "run_meta",
        "tier": tier,
        "tool_version": "0.1.0",
        "corpus_hash": CORPUS,
        "studio_ref": ref,
        "bundle": {"production": True},
        "platform": {"system": "Linux"},
        "started_at": "2026-01-01T00:00:00",
        "cadence": "field",
        "rungs": rungs,
        "reps": 1,
        "instrument_level": 0,
    }
    row.update(extra)
    return row


def _cell(
    cell_id,
    tokens,
    arm = "A0",
    rep = 0,
    tier = "standard",
):
    return {
        "row_type": "cell",
        "cell_id": cell_id,
        "cell": {"arm": arm, "rep": rep, "tier": tier},
        "completed": True,
        "fidelity": "ok",
        "target_tokens": tokens,
    }


def _keystroke(cell_id, p95):
    return {
        "row_type": "action",
        "cell_id": cell_id,
        "action": "keystroke",
        "window": "film",
        "ran": True,
        "expect_ok": True,
        "expect": {},
        "timings": {"p95_ms": p95},
        "slot_missed": False,
    }


def _record(paths, session, rows):
    """Write `rows` through the REAL recorder, which is what appends in a live run."""
    recorder = Recorder(paths.payload_jsonl, session)
    for row in rows:
        recorder.emit(dict(row))
    recorder.close()


def _finished_ab(
    tmp_path,
    *,
    tier = "standard",
    base = "main",
    treatment = "fix",
    treatment_url = "",
) -> Paths:
    paths = Paths.under(tmp_path / "out")
    _record(
        paths,
        "sess-1",
        [
            _run_meta(tier, base, ["10K"]),
            {
                "row_type": "ab_plan",
                "base_ref": base,
                "treatment_ref": treatment,
                # Empty for a treatment the run installed itself, exactly as `run()` records it.
                "treatment_url": treatment_url,
                "balanced": False,
                "order": ["r10K.base.rep0", "r10K.treatment.rep0"],
            },
            _cell("r10K.base.rep0", 10_000, arm = "base", tier = tier),
            _keystroke("r10K.base.rep0", 100.0),
            _cell("r10K.treatment.rep0", 10_000, arm = "treatment", tier = tier),
            _keystroke("r10K.treatment.rep0", 50.0),
        ],
    )
    return paths


# ── a resume must be a resume OF THIS RUN ───────────────────────────────────────────────────


def test_resuming_after_changing_the_treatment_ref_is_refused(tmp_path):
    paths = _finished_ab(tmp_path)
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "other", "--resume"])

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, "other", CORPUS), resume = True, log = lambda *_a: None
        )

    message = str(excinfo.value)
    assert "treatment_ref" in message
    assert "'fix'" in message and "'other'" in message


def test_resuming_a_self_managed_treatment_against_an_attached_one_is_refused(tmp_path):
    """The label after `--ab` says nothing about where the build came from.

    The payload's treatment was cloned and built by that run; this one points the same label at a
    server the caller is holding open. Two different builds, one `--ab fix`.
    """

    paths = _finished_ab(tmp_path)
    args = parse_args(
        ["--tier", "standard", "--branch", "main", "--ab", "fix"]
        + ["--attach-b", "http://127.0.0.1:5311", "--resume"]
    )

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )

    assert "treatment_url" in str(excinfo.value)


def test_resuming_after_changing_the_branch_is_refused(tmp_path):
    paths = _finished_ab(tmp_path)
    args = parse_args(["--tier", "standard", "--branch", "release", "--ab", "fix", "--resume"])

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )

    assert "studio_ref" in str(excinfo.value)


def test_resuming_a_fast_tier_payload_under_the_standard_tier_is_refused(tmp_path):
    """The films differ: 243 s of scene against 47 s, with different budgets and headroom."""

    paths = _finished_ab(tmp_path, tier = "fast")
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "fix", "--resume"])

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )

    assert "tier" in str(excinfo.value)


def test_resuming_after_changing_the_cadence_or_the_instrument_level_is_refused(tmp_path):
    paths = _finished_ab(tmp_path)
    args = parse_args(
        ["--tier", "standard", "--branch", "main", "--ab", "fix", "--cadence", "fast"]
        + ["--instrument-level", "3", "--resume"]
    )

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )

    message = str(excinfo.value)
    assert "cadence" in message
    assert "instrument_level" in message


def test_the_refusal_happens_before_anything_is_installed_or_recorded(tmp_path):
    """A refusal that costs a clone and a build has said nothing it could not have said at once.

    And it must not leave its own `run_meta` behind: a rejected run's header in the payload it
    refused to touch would then reject the correctly configured resume that followed it.
    """

    paths = _finished_ab(tmp_path)
    before = paths.payload_jsonl.read_text(encoding = "utf-8")
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "other", "--resume"])

    with pytest.raises(SystemExit):
        prepare_payload(
            paths, requested_identity(args, "other", CORPUS), resume = True, log = lambda *_a: None
        )

    assert paths.payload_jsonl.read_text(encoding = "utf-8") == before
    assert sorted(p.name for p in paths.out.glob("payload*.jsonl")) == ["payload.jsonl"]


# ── the controls: a legitimate resume still resumes ──────────────────────────────────────────


def test_the_same_configuration_still_resumes(tmp_path):
    paths = _finished_ab(tmp_path)
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "fix", "--resume"])

    assert (
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.base.rep0", "r10K.treatment.rep0"}


def test_resuming_with_more_reps_and_another_rung_is_still_a_resume(tmp_path):
    """`--reps` and `--rungs` ADD cells; they do not reinterpret the ones already recorded."""

    paths = _finished_ab(tmp_path)
    args = parse_args(
        ["--tier", "standard", "--branch", "main", "--ab", "fix", "--reps", "4"]
        + ["--rungs", "10K,100K", "--resume"]
    )

    assert (
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )


def test_a_payload_that_never_recorded_an_axis_still_resumes(tmp_path):
    """The legacy control. An older payload carries only the required `run_meta` keys, and an axis
    it never declared cannot be a difference."""

    paths = Paths.under(tmp_path / "out")
    legacy = {
        "row_type": "run_meta",
        "tier": "standard",
        "tool_version": "0.0.9",
        "corpus_hash": CORPUS,
        "studio_ref": "main",
        "bundle": {"production": True},
        "platform": {"system": "Linux"},
        "started_at": "2026-01-01T00:00:00",
    }
    _record(
        paths, "sess-old", [legacy, _cell("r10K.A0.rep0", 10_000), _keystroke("r10K.A0.rep0", 9)]
    )
    args = parse_args(
        ["--tier", "standard", "--branch", "main", "--cadence", "fast"]
        + ["--instrument-level", "2", "--resume"]
    )

    assert (
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.A0.rep0"}


def test_an_attached_ab_payload_still_resumes_under_a_run_that_is_not_an_ab(tmp_path):
    """The other direction of the same control. A payload recorded as an A/B against a URL, and a
    single-sided run over the same output: the arm in the cell id keeps `A0` off `base` and
    `treatment` without either treatment axis inventing a refusal."""

    paths = _finished_ab(tmp_path, treatment_url = "http://127.0.0.1:5311")
    args = parse_args(["--tier", "standard", "--branch", "main", "--resume"])

    assert (
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )


def test_an_ab_payload_that_never_recorded_a_treatment_url_still_resumes(tmp_path):
    """The legacy control for the newest axis: an `ab_plan` written before the URL was recorded
    declares nothing about it, and an axis a row never declared cannot be a difference."""

    paths = Paths.under(tmp_path / "out")
    _record(
        paths,
        "sess-1",
        [
            _run_meta("standard", "attached:http://127.0.0.1:5310", ["10K"]),
            {
                "row_type": "ab_plan",
                "base_ref": "main",
                "treatment_ref": "fix",
                "balanced": False,
                "order": ["r10K.base.rep0", "r10K.treatment.rep0"],
            },
            _cell("r10K.base.rep0", 10_000, arm = "base"),
        ],
    )
    args = parse_args(
        ["--tier", "standard", "--attach", "http://127.0.0.1:5310", "--ab", "fix"]
        + ["--attach-b", "http://127.0.0.1:5311", "--resume"]
    )

    assert (
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )


def test_a_non_ab_payload_and_an_ab_run_do_not_collide(tmp_path):
    """`A0` against `base`/`treatment`: the arm in the cell id already keeps these apart, so the
    identity check must not invent a refusal on top of it."""

    paths = Paths.under(tmp_path / "out")
    _record(
        paths,
        "sess-1",
        [_run_meta("standard", "main", ["10K"]), _cell("r10K.A0.rep0", 10_000)],
    )
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "fix", "--resume"])

    assert (
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )


# ── a fresh run does not append to the run before it ─────────────────────────────────────────


def test_a_fresh_run_moves_the_previous_payload_aside(tmp_path):
    paths = _finished_ab(tmp_path)
    before = paths.payload_jsonl.read_text(encoding = "utf-8")

    archived = prepare_payload(
        paths,
        requested_identity(parse_args(["--tier", "standard"]), None, CORPUS),
        resume = False,
        log = lambda *_a: None,
    )

    assert archived is not None and archived.exists()
    assert archived.read_text(encoding = "utf-8") == before
    assert not paths.payload_jsonl.exists()


def test_the_readme_sequence_no_longer_scores_two_runs_as_one_ladder(tmp_path):
    """`--tier fast --attach ... --out X` then `--tier standard --branch main --out X`, the second
    killed before its top rung. The 100K reading must not be the fast run's."""

    paths = Paths.under(tmp_path / "out")
    _record(
        paths,
        "sess-fast",
        [
            _run_meta("fast", "attached:http://127.0.0.1:5401", ["100K"]),
            _cell("r100K.A0.rep0", 100_000, tier = "fast"),
            _keystroke("r100K.A0.rep0", 900.0),
        ],
    )

    prepare_payload(
        paths,
        requested_identity(parse_args(["--tier", "standard", "--branch", "main"]), None, CORPUS),
        resume = False,
        log = lambda *_a: None,
    )
    # The standard run, killed by the watchdog after 10K.
    _record(
        paths,
        "sess-standard",
        [
            _run_meta("standard", "main", ["1K", "10K", "100K"]),
            _cell("r1K.A0.rep0", 1_000),
            _keystroke("r1K.A0.rep0", 40.0),
            _cell("r10K.A0.rep0", 10_000),
            _keystroke("r10K.A0.rep0", 55.0),
        ],
    )

    rows = [
        json.loads(line)
        for line in paths.payload_jsonl.read_text(encoding = "utf-8").splitlines()
        if line.strip()
    ]
    measures = measures_from_records(latest_attempt_rows(rows))
    assert 100_000 not in measures, "the fast tier's 100K cell is in the standard run's ladder"
    assert {1_000, 10_000} <= set(measures)

    _text, _ladder, payload = build_report(paths.payload_jsonl, [1_000, 10_000, 100_000])
    assert payload["header"]["tier"] == "standard"
    assert payload["header"]["studio_ref"] == "main"


def test_a_fresh_run_into_a_new_directory_archives_nothing(tmp_path):
    paths = Paths.under(tmp_path / "out")
    assert archive_payload(paths, log = lambda *_a: None) is None


def test_an_empty_payload_is_not_worth_archiving(tmp_path):
    paths = Paths.under(tmp_path / "out")
    paths.payload_jsonl.write_text("", encoding = "utf-8")
    assert archive_payload(paths, log = lambda *_a: None) is None
    assert paths.payload_jsonl.exists()


def test_a_resume_never_archives(tmp_path):
    paths = _finished_ab(tmp_path)
    before = paths.payload_jsonl.read_text(encoding = "utf-8")
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "fix", "--resume"])

    prepare_payload(
        paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
    )

    assert paths.payload_jsonl.read_text(encoding = "utf-8") == before
    assert sorted(p.name for p in paths.out.glob("payload*.jsonl")) == ["payload.jsonl"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
