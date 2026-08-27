# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One `--out` holds one run, and `--resume` continues the run that is in it.

A cell id is `r{rung}.{arm}.rep{rep}` and nothing more, so the tier that picked the film, the
cadence the reply streamed at, the instrument level, the corpus and both refs are all invisible to
the two places that decide what a payload already contains. Two ways that produced a report about
something nobody ran:

  A RESUME UNDER A DIFFERENT CONFIGURATION SKIPPED EVERYTHING. `--branch main --ab other --resume`
  into a directory holding a finished `main -> fix` run installs and launches two Unsloth instances, finds
  every `cell_id` already complete, exits 0, and -- because `_render_ab` keeps the previous table
  when this session measured nothing -- leaves the OLD comparison standing in `ab.md` to be read
  as the answer for `other`.

  A FRESH RUN APPENDED TO THE PREVIOUS ONE. `Recorder` opens the payload with `"a"`. The README's
  own two invocations wrote into one `--out`, and when the second was interrupted the file held a
  fast-tier 100K cell from an attached Unsloth next to standard-tier 1K and 10K cells from `main`.
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
    TOOL_VERSION,
    _resume_set,
    archive_payload,
    parse_args,
    prepare_payload,
    requested_identity,
)
from studiobench.report.build import build_report, score_payload  # noqa: E402
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
        "tool_version": TOOL_VERSION,
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
        "tool_version": TOOL_VERSION,
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


def test_resuming_an_ab_payload_as_a_single_build_run_is_refused(tmp_path):
    """The other direction of the mode transition, refused for the same reason.

    The A/B's `base` cells stay in the payload and stay first at their rung, so the ladder keeps
    reporting them and never the `A0` cells this run is about to measure."""

    paths = _finished_ab(tmp_path, treatment_url = "http://127.0.0.1:5311")
    args = parse_args(["--tier", "standard", "--branch", "main", "--resume"])

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )

    assert "mode" in str(excinfo.value)


def test_resuming_a_payload_measured_by_another_instrument_version_is_refused(tmp_path):
    """A harness upgrade mid-campaign is the one identity change nobody has to type.

    `TOOL_VERSION` is bumped when what an instrument MEASURES changes and for no other reason: at
    0.2.0 `reasoning_toggle.open_ms` terminates on a settled mount rather than on the `data-state`
    flip, which is a different quantity under the same name. None of that moves a cell id, so
    `--resume` into a half-finished 0.1.0 payload from an upgraded tree kept the old completed
    cells and appended new ones measured by the new instruments. `merged_run_meta` names that
    mixture, but plain `--report` never calls it -- it reads the FIRST header and pools both.
    """

    paths = Paths.under(tmp_path / "upgraded")
    _record(
        paths,
        "sess-old",
        [
            {**_run_meta("standard", "main", ["10K"]), "tool_version": "0.1.0"},
            _cell("r10K.A0.rep0", 10_000),
            _keystroke("r10K.A0.rep0", 9),
        ],
    )
    args = parse_args(["--tier", "standard", "--branch", "main", "--rungs", "10K,100K", "--resume"])

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )

    assert "tool_version" in str(excinfo.value)
    assert TOOL_VERSION in str(excinfo.value)


def test_a_payload_recorded_by_this_version_still_resumes(tmp_path):
    """The control: the axis must refuse an upgrade, not every resume."""

    paths = Paths.under(tmp_path / "same")
    _record(
        paths,
        "sess-1",
        [_run_meta("standard", "main", ["10K"]), _cell("r10K.A0.rep0", 10_000)],
    )
    args = parse_args(["--tier", "standard", "--branch", "main", "--rungs", "10K,100K", "--resume"])

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


def test_resuming_a_single_build_payload_as_an_ab_is_refused(tmp_path):
    """`A0` against `base`/`treatment` keeps the new cells from being SKIPPED. It does not keep
    them out of the REPORT, which is a different question and the one that decides this.

    `score_payload` hands every cell to `measures_from_records`, which keys by rung and keeps the
    first reading, and `_completion_by_rung` keeps a failure over a success at the same rung. So
    the payload below -- a single-build run that died at 10K, then an A/B that measured both arms
    of 10K without a fault -- reports `INCOMPLETE: timeout: the base died at 10K`, scores the rung
    0 and prints ONSET RUNG: none, over a rung two arms had just measured. The same crash resumed
    WITHOUT `--ab` reuses the cell id, `latest_attempt_rows` supersedes the dead attempt, and the
    rung scores. The mode is part of what the payload measured; changing it is not a resume.
    """

    paths = Paths.under(tmp_path / "out")
    _record(
        paths,
        "sess-1",
        [_run_meta("standard", "main", ["10K"]), _cell("r10K.A0.rep0", 10_000)],
    )
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "fix", "--resume"])

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )

    assert "mode" in str(excinfo.value)


def test_a_session_that_died_before_its_first_cell_declares_no_mode(tmp_path):
    """The control for the refusal above: it may not fire on a run that measured nothing.

    `run_meta` is written after both installs and `ab_plan` after the corpus and the gates, so an
    A/B killed in between leaves a header and no cells. There is nothing there for a later report
    to keep in preference to this one's readings, so resuming it is a resume.
    """

    paths = Paths.under(tmp_path / "out")
    _record(paths, "sess-1", [_run_meta("standard", "main", ["10K"])])
    args = parse_args(["--tier", "standard", "--branch", "main", "--ab", "fix", "--resume"])

    assert (
        prepare_payload(
            paths, requested_identity(args, "fix", CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )


# ── the two fixture axes ────────────────────────────────────────────────────────────────────
#
# `--stream-tail-chars` and `--corpus-dollars` change what the last turn STREAMS, and neither of
# them moves `corpus_hash`, which covers the frozen units on disk and the generator's parameters.
# Nor does either move a `cell_id`. They are on `IDENTITY_AXES` for the same reason the tier is:
# without that, a resume under a changed fixture skips cells measured against a different film and
# the payload silently becomes one ladder built from two.


def _fixture_payload(tmp_path, name, session, **fixture):
    paths = Paths.under(tmp_path / name)
    _record(
        paths,
        session,
        [
            _run_meta("standard", "main", ["10K"], **fixture),
            _cell("r10K.A0.rep0", 10_000),
            _keystroke("r10K.A0.rep0", 55.0),
        ],
    )
    return paths


def _one_engine_rung(
    tmp_path,
    engine,
    name = "out",
) -> Paths:
    """A payload holding one measured rung, recorded by a session that launched `engine`."""

    paths = Paths.under(tmp_path / name)
    _record(
        paths,
        "sess-1",
        [
            _run_meta(
                "standard",
                "main",
                ["1K", "10K"],
                platform = {"system": "Linux", "engine": engine, "engine_note": "for this test"},
            ),
            _cell("r1K.A0.rep0", 1_000),
            _keystroke("r1K.A0.rep0", 40.0),
        ],
    )
    return paths


def _resume_args(*flags):
    return parse_args(["--tier", "standard", "--branch", "main", "--resume", *flags])


def test_resuming_under_a_changed_stream_tail_is_refused(tmp_path):
    paths = _fixture_payload(tmp_path, "tail", "sess-1", stream_tail_chars = 24_000)

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_resume_args("--stream-tail-chars", "96000"), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )

    message = str(excinfo.value)
    assert "stream_tail_chars" in message
    assert "24000" in message and "96000" in message


def test_resuming_under_a_changed_dollar_setting_is_refused(tmp_path):
    """Both directions. Dropping the flag on the resume is the likelier of the two."""

    on = _fixture_payload(tmp_path, "on", "sess-on", corpus_dollars = True)
    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            on,
            requested_identity(_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
    assert "corpus_dollars" in str(excinfo.value)

    off = _fixture_payload(tmp_path, "off", "sess-off", corpus_dollars = False)
    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            off,
            requested_identity(_resume_args("--corpus-dollars"), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
    assert "corpus_dollars" in str(excinfo.value)


def test_a_payload_that_already_holds_two_fixtures_is_refused_either_way(tmp_path):
    """Two sessions, two films, one file: whichever of them you ask for, the other one is still
    in there and its cells would still be skipped."""

    paths = _fixture_payload(tmp_path, "mixed", "sess-1", stream_tail_chars = 6_000)
    _record(
        paths,
        "sess-2",
        [
            _run_meta("standard", "main", ["10K"], stream_tail_chars = 96_000),
            _cell("r100K.A0.rep0", 100_000),
        ],
    )

    for asked in ("6000", "96000"):
        with pytest.raises(SystemExit) as excinfo:
            prepare_payload(
                paths,
                requested_identity(_resume_args("--stream-tail-chars", asked), None, CORPUS),
                resume = True,
                log = lambda *_a: None,
            )
        assert "stream_tail_chars" in str(excinfo.value)


# ── the controls: neither axis may swallow a resume that is legitimate ───────────────────────


def test_an_unchanged_fixture_resumes_and_returns_its_completed_cells(tmp_path):
    paths = _fixture_payload(
        tmp_path, "same", "sess-1", stream_tail_chars = 24_000, corpus_dollars = True
    )
    args = _resume_args("--stream-tail-chars", "24000", "--corpus-dollars")

    assert (
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.A0.rep0"}


# ── the browser engine that rendered every number ────────────────────────


def test_resuming_under_a_different_browser_engine_is_refused(tmp_path):
    """The engine RENDERED every number in the payload, and the cell id does not name it.

    A run that measured 1K under Chromium, resumed under WebKit to add 10K, skips the completed
    Chromium cell and measures the new rung under a different renderer -- and the two land in one
    ladder, which `score_payload` reads as one build getting slower with context. The engine is
    part of what a payload measured, exactly like the tier and the cadence.
    """

    paths = _one_engine_rung(tmp_path, "chromium")
    args = parse_args(
        ["--tier", "standard", "--branch", "main", "--engine", "webkit"]
        + ["--rungs", "1K,10K", "--resume"]
    )

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )

    message = str(excinfo.value)
    assert "engine" in message
    assert "'chromium'" in message and "'webkit'" in message


def test_the_engine_compared_is_the_one_that_will_launch_not_the_flag(tmp_path):
    """`--engine` defaults to nothing and `browser.launch` resolves the platform's webview family.

    So the commonest engine change of all is the one where the second invocation names no engine
    at all, and comparing the flags rather than the resolved engines would wave it through.
    """

    from studiobench.runtime.browser import default_engine

    other = "chromium" if default_engine()[0] != "chromium" else "webkit"
    paths = _one_engine_rung(tmp_path, other)
    args = parse_args(["--tier", "standard", "--branch", "main", "--rungs", "1K,10K", "--resume"])

    assert requested_identity(args, None, CORPUS)["engine"] == default_engine()[0]
    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )

    assert "engine" in str(excinfo.value)


def test_the_same_engine_still_resumes(tmp_path):
    """The control: naming the engine the payload was recorded under is a resume."""

    paths = _one_engine_rung(tmp_path, "chromium")
    args = parse_args(
        ["--tier", "standard", "--branch", "main", "--engine", "chromium"]
        + ["--rungs", "1K,10K", "--resume"]
    )

    assert (
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )
    assert _resume_set(paths) == {"r1K.A0.rep0"}


def test_a_payload_from_before_the_fixture_axes_existed_resumes_exactly_as_it_did(tmp_path):
    """No `stream_tail_chars` and no `corpus_dollars` key at all, which is every payload written
    before this branch. Resumed UNDER THE DEFAULTS it is asking for the film it already ran, so it
    resumes and its completed cells are skipped, exactly as they were before either flag existed.

    The other half of that reading -- a resume that does NOT use the defaults -- is the test below.
    """

    paths = _fixture_payload(tmp_path, "old", "sess-old")
    assert "stream_tail_chars" not in paths.payload_jsonl.read_text(encoding = "utf-8")

    assert (
        prepare_payload(
            paths,
            requested_identity(_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.A0.rep0"}


def test_a_payload_from_before_the_fixture_axes_is_refused_under_a_non_default_fixture(tmp_path):
    """REGRESSION. Absence PROVES the default here, and skipping the axis threw that proof away.

    An axis a payload never declared is normally skipped: it declined to say, so there is nothing
    to disagree with. These two axes arrived WITH the flags that set them, so a payload written
    before them could not have run under anything but `stream_tail_chars = None` and
    `corpus_dollars = False`. Skipping them accepted `--resume --stream-tail-chars 24000` against
    such a payload, skipped every cell it had completed, and recorded the remaining cells under a
    different streamed fixture beneath the same cell ids -- the mixed ladder the refusal exists to
    prevent, arrived at through the one door left open.
    """

    for name, flags in (
        ("tail", ["--stream-tail-chars", "24000"]),
        ("dollars", ["--corpus-dollars"]),
    ):
        paths = _fixture_payload(tmp_path, f"legacy-{name}", "sess-old")
        with pytest.raises(SystemExit) as excinfo:
            prepare_payload(
                paths,
                requested_identity(_resume_args(*flags), None, CORPUS),
                resume = True,
                log = lambda *_a: None,
            )

        message = str(excinfo.value)
        assert ("stream_tail_chars" if "--stream-tail-chars" in flags else "corpus_dollars") in (
            message
        )
        # The refusal has to say WHY an axis the payload never mentions is a difference, or the
        # reader's next move is to go looking for a key that was never going to be there.
        assert "predates this axis" in message


# ── the measurement-mode axis ───────────────────────────────────────────────────────────────
#
# `--click-probe` runs a full `page.click`, a real mouse click, a dispatch, a focus and a hover
# over the thread BEFORE the film starts, and the tool's own help text for the flag says it "makes
# the cell's timings incomparable with a cell that did not run it". The cell then carries a
# `composer_click_ms` measured on a composer all of those paths have already been through, and a
# `click_attribution` block a cell without the flag does not have at all. None of that moves the
# cell id, so it is on `IDENTITY_AXES` for the same reason the tier is.


def test_resuming_after_toggling_the_click_probe_is_refused(tmp_path):
    """REGRESSION. Both directions, because either one produces the same mixed ladder.

    Without this the flag was invisible to the check: `click_probe` was in neither `run_meta` nor
    `requested_identity`, so `--resume --click-probe` against a finished plain payload skipped
    every completed cell, appended the remaining rungs measured the other way, and reported the
    two halves as one ladder.
    """

    probed = _fixture_payload(tmp_path, "probed", "sess-probed", click_probe = True)
    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            probed,
            requested_identity(_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
    message = str(excinfo.value)
    assert "click_probe" in message
    assert "True" in message and "False" in message

    plain = _fixture_payload(tmp_path, "plain", "sess-plain", click_probe = False)
    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            plain,
            requested_identity(_resume_args("--click-probe"), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
    assert "click_probe" in str(excinfo.value)


def test_a_payload_from_before_the_probe_flag_is_refused_under_a_probed_resume(tmp_path):
    """Absence proves the default here too. A payload with no `click_probe` key was written when
    there was no way to ask for the probe, so it ran without it."""

    paths = _fixture_payload(tmp_path, "legacy-probe", "sess-old")
    assert "click_probe" not in paths.payload_jsonl.read_text(encoding = "utf-8")

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_resume_args("--click-probe"), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )

    message = str(excinfo.value)
    assert "click_probe" in message
    assert "predates this axis" in message


def test_an_unchanged_probe_setting_resumes_and_returns_its_completed_cells(tmp_path):
    """The control. The axis may not swallow a resume that is asking for the run it already has,
    on either setting."""

    probed = _fixture_payload(tmp_path, "probed-same", "sess-probed", click_probe = True)
    assert (
        prepare_payload(
            probed,
            requested_identity(_resume_args("--click-probe"), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )
    assert _resume_set(probed) == {"r10K.A0.rep0"}

    plain = _fixture_payload(tmp_path, "plain-same", "sess-plain")
    assert (
        prepare_payload(
            plain,
            requested_identity(_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )
    assert _resume_set(plain) == {"r10K.A0.rep0"}


def test_the_probe_flag_reaches_the_identity_this_invocation_asks_for():
    """The check can only refuse a difference it was told about, so the flag has to be requested.

    What the payload has to carry for it to be compared against is asserted end to end, over a
    `run_meta` a real run wrote, in `test_studiobench_run_acquisition`.
    """

    for flags, expected in ((("--click-probe",), True), ((), False)):
        args = parse_args(["--tier", "standard", "--branch", "main", *flags])
        assert requested_identity(args, None, CORPUS)["click_probe"] is expected


def test_the_skip_rule_still_holds_for_an_axis_that_really_did_decline_to_say(tmp_path):
    """The other axes keep the general rule: `run_meta` has always carried them, so a payload that
    omits one omitted it for its own reasons and this check has nothing to say about it. Only the
    axes that arrived with a flag of their own read absence as a value."""

    paths = Paths.under(tmp_path / "quiet")
    _record(
        paths,
        "sess-quiet",
        [
            {
                "row_type": "run_meta",
                "tier": "standard",
                "tool_version": TOOL_VERSION,
                "corpus_hash": CORPUS,
                "studio_ref": "main",
                "bundle": {"production": True},
                "platform": {"system": "Linux"},
                "started_at": "2026-01-01T00:00:00",
                "stream_tail_chars": 24_000,
                "corpus_dollars": True,
            },
            _cell("r10K.A0.rep0", 10_000),
        ],
    )
    args = _resume_args("--stream-tail-chars", "24000", "--corpus-dollars", "--cadence", "fast")

    assert (
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )


def test_a_dead_cell_is_still_re_run_and_a_missing_payload_is_still_empty(tmp_path):
    """The refusal decides WHETHER to resume; this is what a resume then skips. A cell that died
    is re-run, because its failure may have been the machine and not the build."""

    paths = Paths.under(tmp_path / "died")
    _record(
        paths,
        "sess-1",
        [
            _run_meta("standard", "main", ["10K"], stream_tail_chars = None),
            {**_cell("r10K.A0.rep0", 10_000), "completed": False, "fidelity": "died"},
        ],
    )
    assert _resume_set(paths) == set()
    assert _resume_set(Paths.under(tmp_path / "nothing-here")) == set()


# ── the engine controls ───────────────────────────────────────────────────


def test_a_payload_that_never_recorded_an_engine_still_resumes(tmp_path):
    """The legacy control for this axis, and the one for a session that never got a browser up.

    `run_meta` is emitted after `browser.launch` returns, so a session with no engine in its
    header either predates the field or never launched anything. Neither declares an engine, and
    an axis a row never declared cannot be a difference.
    """

    paths = Paths.under(tmp_path / "out")
    _record(
        paths,
        "sess-old",
        [
            _run_meta("standard", "main", ["1K", "10K"], platform = {"system": "Linux"}),
            _cell("r1K.A0.rep0", 1_000),
            _keystroke("r1K.A0.rep0", 40.0),
        ],
    )
    args = parse_args(
        ["--tier", "standard", "--branch", "main", "--engine", "firefox"]
        + ["--rungs", "1K,10K", "--resume"]
    )

    assert (
        prepare_payload(
            paths, requested_identity(args, None, CORPUS), resume = True, log = lambda *_a: None
        )
        is None
    )
    assert _resume_set(paths) == {"r1K.A0.rep0"}


def test_the_report_a_mode_change_would_have_produced(tmp_path):
    """WHY the refusal above exists, as a number rather than as an argument.

    Both payloads hold the same facts: 10K died under one run and was measured clean under the
    next. The only difference is whether the repair ran under `--ab`, which changes the arm and
    so the cell id -- and with it whether the dead attempt is superseded or kept.
    """

    def _payload(name, repair_arm):
        paths = Paths.under(tmp_path / name)
        dead = _cell("r10K.A0.rep0", 10_000)
        dead["completed"] = False
        dead["failure"] = {"kind": "timeout", "message": "the base died at 10K"}
        _record(paths, "sess-1", [_run_meta("standard", "main", ["10K"]), dead])
        cell_id = f"r10K.{repair_arm}.rep0"
        _record(
            paths,
            "sess-2",
            [
                _run_meta("standard", "main", ["10K"]),
                _cell(cell_id, 10_000, arm = repair_arm),
                _keystroke(cell_id, 100.0),
            ],
        )
        return paths

    same_mode = score_payload(_payload("same", "A0").payload_jsonl, [10_000])
    changed_mode = score_payload(_payload("changed", "base").payload_jsonl, [10_000])

    # Same mode: the retry reuses the cell id, so the crash is superseded and never reaches the
    # ladder. Change the mode and the same crash is a cell of its own, first at its rung, and it
    # is what the report prints -- over a rung the run being reported measured cleanly.
    assert "died at 10K" not in (same_mode.rungs[0].incomplete_reason or "")
    assert "died at 10K" in (changed_mode.rungs[0].incomplete_reason or "")


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


# ── the injection axis ──────────────────────────────────────────────────────────────────────
#
# `--inject-stream-cost-ms` burns a known amount of main-thread time per SSE chunk on the TREATMENT
# arm, so a treatment cell recorded with it is not the same reading as one recorded without it --
# and `cell_id` carries the rung, the arm and the repetition and nothing that could tell. It is on
# `IDENTITY_AXES` for the same reason the two fixture axes above are, and for a larger perturbation
# than either of them makes.


def _injected_payload(tmp_path, name, session, **fixture):
    paths = Paths.under(tmp_path / name)
    _record(
        paths,
        session,
        [
            _run_meta("standard", "main", ["10K"], **fixture),
            _cell("r10K.base.rep0", 10_000, arm = "base"),
            _cell("r10K.treatment.rep0", 10_000, arm = "treatment"),
        ],
    )
    return paths


def _ab_resume_args(*flags):
    return parse_args(
        ["--tier", "standard", "--branch", "main", "--ab", "main", "--resume", *flags]
    )


def test_resuming_a_finished_uninjected_run_with_the_injection_on_is_refused(tmp_path):
    """REGRESSION, and the expensive half of it.

    Every pair in this payload is complete, so `skippable_cells` skips all of them, `rows` comes
    back empty with `resumed` non-zero and `completion_exit_code` returns 0. The run pays for two
    installs, measures nothing, and the recovery gate is then answered out of cells that were never
    injected -- a calibration reporting that the metric cannot see a cost nobody put there.
    """

    paths = _injected_payload(tmp_path, "clean", "sess-1")

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_ab_resume_args("--inject-stream-cost-ms", "3"), "main", CORPUS),
            resume = True,
            log = lambda *_a: None,
        )

    message = str(excinfo.value)
    assert "inject_stream_cost_ms" in message
    assert "3.0" in message


def test_resuming_an_injected_run_without_the_injection_is_refused(tmp_path):
    """The other direction, and the likelier one: an injected A/B dies part way up the ladder and
    the retry is typed without the flag. The rungs already done stay injected, the rungs still to
    come are not, and the ladder the recovery fraction is computed from is half of each."""

    paths = _injected_payload(tmp_path, "injected", "sess-1", inject_stream_cost_ms = 3.0)

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_ab_resume_args(), "main", CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
    assert "inject_stream_cost_ms" in str(excinfo.value)


def test_resuming_under_a_changed_injection_amount_is_refused(tmp_path):
    """The amount is the measurement. A ladder half of which burned 3 ms per chunk and half 40 ms
    has no recovery fraction at all."""

    paths = _injected_payload(tmp_path, "amount", "sess-1", inject_stream_cost_ms = 3.0)

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_ab_resume_args("--inject-stream-cost-ms", "40"), "main", CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
    assert "inject_stream_cost_ms" in str(excinfo.value)


def test_a_payload_from_before_the_injection_flag_resumes_under_the_default(tmp_path):
    """THE CONTROL, and the one `HISTORICAL_DEFAULTS` exists for. A payload written before the flag
    could not have been injected, so absence reads as off and an ordinary resume still works."""

    paths = _injected_payload(tmp_path, "old", "sess-old")
    assert "inject_stream_cost_ms" not in paths.payload_jsonl.read_text(encoding = "utf-8")

    assert (
        prepare_payload(
            paths,
            requested_identity(_ab_resume_args(), "main", CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.base.rep0", "r10K.treatment.rep0"}


def test_an_unchanged_injection_resumes(tmp_path):
    """THE SECOND CONTROL. An injected run that died is meant to be resumable AS an injected run;
    a refusal on every value would make the axis useless rather than safe."""

    paths = _injected_payload(tmp_path, "same", "sess-1", inject_stream_cost_ms = 3.0)

    assert (
        prepare_payload(
            paths,
            requested_identity(_ab_resume_args("--inject-stream-cost-ms", "3"), "main", CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.base.rep0", "r10K.treatment.rep0"}


# ── the external probe axis ─────────────────────────────────────────────────────────────────
#
# `SBENCH_EXTRA_INIT_SCRIPT` installs a caller's own init script into the page, and the run says so
# in its log: "this run carries an external probe and is NOT a clean measurement of the build".
# What it cannot end in is a wrong number, because `refuse_if_probed` reads EVERY `run_meta` in the
# file and every scoring entry point calls it. It is on `IDENTITY_AXES` for what that refusal
# costs: the refusal is whole-file and the payload is append-only, so a resume that toggles the
# probe takes the cells recorded before it down with it, and nothing here can put them back.
#
# It is also the axis most easily toggled by accident. The other four are typed on the command
# line every time; this one is an environment variable, and a variable that is still set is
# indistinguishable from one that was never set until the run is over.

PROBE = "probes/paint_counter.js"


def _probe_resume_args():
    return parse_args(["--tier", "standard", "--branch", "main", "--resume"])


def test_resuming_a_clean_payload_with_the_probe_variable_still_set_is_refused(
    tmp_path, monkeypatch
):
    """REGRESSION, and the destructive direction.

    Half a clean ladder is on disk, the variable from an earlier probe experiment is still set in
    the shell, and the resume was not asking for a probe run at all. Without the axis this
    installed both sides, ran every rung still owed with the probe in the page, and appended a
    probed `run_meta` -- after which `refuse_if_probed` refuses the file, INCLUDING the cells that
    were recorded cleanly hours earlier.
    """

    paths = _fixture_payload(tmp_path, "clean", "sess-clean")
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", PROBE)

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_probe_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )

    message = str(excinfo.value)
    assert "probe_init_script" in message
    assert PROBE in message
    assert "predates this axis" in message


def test_resuming_a_probed_payload_with_the_variable_unset_is_refused(tmp_path, monkeypatch):
    """The other direction. It cannot poison anything -- the payload is already unscorable -- but
    it is a full wave of installs and rungs spent on a file no reader will ever accept."""

    paths = _fixture_payload(tmp_path, "probed", "sess-probed", probe_init_script = PROBE)
    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_probe_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )

    message = str(excinfo.value)
    assert "probe_init_script" in message
    assert PROBE in message


def test_resuming_a_probed_payload_under_a_different_probe_is_refused(tmp_path, monkeypatch):
    """Which probe is the experiment. Two scripts sample different things on different schedules,
    so half a ladder under each is not one arm however unscorable both halves already are."""

    paths = _fixture_payload(tmp_path, "other", "sess-probed", probe_init_script = PROBE)
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", "probes/layout_counter.js")

    with pytest.raises(SystemExit) as excinfo:
        prepare_payload(
            paths,
            requested_identity(_probe_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
    assert "probe_init_script" in str(excinfo.value)


def test_a_payload_from_before_the_probe_hook_resumes_under_the_default(tmp_path, monkeypatch):
    """THE CONTROL, and the one `HISTORICAL_DEFAULTS` exists for. A payload with no
    `probe_init_script` key was written when there was no way to install a probe, so absence reads
    as clean and an ordinary resume is still an ordinary resume."""

    paths = _fixture_payload(tmp_path, "old", "sess-old")
    assert "probe_init_script" not in paths.payload_jsonl.read_text(encoding = "utf-8")
    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)

    assert (
        prepare_payload(
            paths,
            requested_identity(_probe_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.A0.rep0"}


def test_an_unchanged_probe_resumes(tmp_path, monkeypatch):
    """THE SECOND CONTROL. A probe ladder that died is meant to be resumable AS a probe ladder.
    The payload is not scorable either way, but a potency run is a run, and refusing every value
    would make the axis useless rather than safe."""

    paths = _fixture_payload(tmp_path, "same", "sess-probed", probe_init_script = PROBE)
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", PROBE)

    assert (
        prepare_payload(
            paths,
            requested_identity(_probe_resume_args(), None, CORPUS),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )
    assert _resume_set(paths) == {"r10K.A0.rep0"}


def test_the_probe_variable_reaches_the_identity_this_invocation_asks_for(monkeypatch):
    """The check can only refuse a difference it was told about, and this one is not on `args`:
    the hook is an environment variable, so `requested_identity` reads it where it is set."""

    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", PROBE)
    assert requested_identity(_probe_resume_args(), None, CORPUS)["probe_init_script"] == PROBE

    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", "")
    assert requested_identity(_probe_resume_args(), None, CORPUS)["probe_init_script"] is None

    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)
    assert requested_identity(_probe_resume_args(), None, CORPUS)["probe_init_script"] is None


def test_one_probed_session_makes_the_clean_cells_beside_it_unscorable(tmp_path):
    """WHAT THE REFUSAL COSTS, which is the whole reason the axis is worth an entry.

    The mixture is never scored as a result -- that part is already safe, whole-file and with no
    override -- and this is the price of it being safe. The clean session below scores; the same
    file scores nothing at all once one probed `run_meta` has been appended after it. A payload is
    append-only, so this is not recoverable by re-running: it is recoverable only by not doing it.

    Passes on the unfixed code, because it is a property of `refuse_if_probed` rather than of the
    identity check.
    """

    paths = _fixture_payload(tmp_path, "mixed", "sess-clean")
    text, _ladder, _payload = build_report(paths.payload_jsonl, [10_000])
    assert text

    _record(
        paths,
        "sess-probed",
        [
            _run_meta("standard", "main", ["10K"], probe_init_script = PROBE),
            _cell("r100K.A0.rep0", 100_000),
            _keystroke("r100K.A0.rep0", 61.0),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        build_report(paths.payload_jsonl, [10_000, 100_000])
    assert PROBE in str(excinfo.value)
    assert "refusing to score" in str(excinfo.value)


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
