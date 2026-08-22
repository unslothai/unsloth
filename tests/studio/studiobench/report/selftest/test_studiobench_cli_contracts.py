# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CLI decisions that decide what a run MEANS, taken out of the driver so they can be asserted.

Each of these was a one-line condition whose failure was silent:

  a null control that read as an ordinary A/B, because the two identical builds it had just
  installed were on different ports;

  two A/B sides installing into one home, so the treatment overwrote the base while the base was
  running and both arms measured the same binaries;

  a doctor that reported PASS with no browser engine downloaded, which is what `pip install
  playwright` alone leaves behind;

  a report scoring a standard payload against the DEFAULT tier's shorter ladder, so a rung the run
  promised and never reached was dropped instead of scored INCOMPLETE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import (  # noqa: E402
    completion_exit_code,
    engines_installed,
    is_null_control,
    main,
    parse_args,
    recorded_ladder,
    side_home,
    side_specs,
    stream_cost_injection_problem,
)


def _side(label, ref, url, owns):
    return {"label": label, "ref": ref, "base_url": url, "owns": owns}


# ── the null control ────────────────────────────────────────────────────────────────────────


def test_two_self_installed_copies_of_one_ref_are_a_null_control():
    """`--branch main --ab main`: same build, two ports, and it is the calibration run."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True),
        _side("treatment", "main", "http://127.0.0.1:5400", True),
    ]
    assert is_null_control(sides) is True


def test_two_refs_are_never_a_null_control():
    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True),
        _side("treatment", "pr-9296", "http://127.0.0.1:5400", True),
    ]
    assert is_null_control(sides) is False


def test_two_attached_studios_at_different_urls_are_not_a_null_control():
    """Nothing here can see what is deployed at either URL, so equal refs prove nothing."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5401", False),
        _side("treatment", "main", "http://127.0.0.1:5402", False),
    ]
    assert is_null_control(sides) is False


def test_one_attached_studio_driven_twice_is_a_null_control():
    sides = [
        _side("base", "main", "http://127.0.0.1:5401", False),
        _side("treatment", "main", "http://127.0.0.1:5401", False),
    ]
    assert is_null_control(sides) is True


# ── the stream-cost injection needs two origins ─────────────────────────────────────────────
#
# The injection is a CONTEXT init script gated on `window.location.origin`, and both arms are
# driven by one browser context and one page, so the origin is the only thing that can tell them
# apart. Two arms on one origin therefore BOTH burn the injected cost, the difference between them
# is zero, and `evaluate_stream_cost_recovery_gate` -- which reads back
# `(injected_rate - base_rate) * chars` -- reports a recovery of nothing and blames the
# accumulator. The one flag whose job is to separate "the change did nothing" from "the metric is
# not watching" would answer the second when the truth is neither.
#
# One attached Studio driven twice is not a mistake in general: it is a null control this tool
# detects on purpose, pinned by the test directly above. It is only fatal with the injection on.


def _inject_args(attach, attach_b, *extra):
    return parse_args(["--attach", attach, "--ab", "main", "--attach-b", attach_b, *extra])


def test_injecting_stream_cost_into_two_arms_on_one_origin_is_refused():
    """REGRESSION. `--attach U --attach-b U` is the cheapest null control there is -- one Studio,
    no installs -- and it is the obvious place to check that the metric can see a known cost."""

    args = _inject_args(
        "http://127.0.0.1:5401", "http://127.0.0.1:5401", "--inject-stream-cost-ms", "3"
    )
    problem = stream_cost_injection_problem(side_specs(args, "main"), args.inject_stream_cost_ms)

    assert problem is not None
    assert "127.0.0.1:5401" in problem
    assert "origin" in problem


def test_a_trailing_slash_does_not_make_one_origin_into_two():
    """`--attach-b http://host:5401/` and `--attach http://host:5401` are the same server. The
    acquisition loop strips the slash before it ever reaches `origin_scoped`, so a check that did
    not would pass this and inject into both arms anyway."""

    args = _inject_args(
        "http://127.0.0.1:5401", "http://127.0.0.1:5401/", "--inject-stream-cost-ms", "3"
    )
    assert stream_cost_injection_problem(side_specs(args, "main"), args.inject_stream_cost_ms)


def test_an_attached_side_and_a_self_installed_one_on_the_same_port_are_refused():
    """The mixed case. A side this run installs is launched on `--port` and serves at
    `http://127.0.0.1:{port}`, so an `--attach` URL naming that port is the same origin twice
    without either flag looking like it."""

    args = parse_args(
        [
            "--attach",
            "http://127.0.0.1:5400",
            "--ab",
            "main",
            "--attach-b",
            "http://127.0.0.1:5400",
            "--port",
            "5399",
            "--inject-stream-cost-ms",
            "3",
        ]
    )
    assert stream_cost_injection_problem(side_specs(args, "main"), args.inject_stream_cost_ms)


# ── the controls: the refusal may not swallow a run that is fine ────────────────────────────


def test_two_origins_may_inject():
    args = _inject_args(
        "http://127.0.0.1:5401", "http://127.0.0.1:5402", "--inject-stream-cost-ms", "3"
    )
    assert (
        stream_cost_injection_problem(side_specs(args, "main"), args.inject_stream_cost_ms) is None
    )


def test_a_self_installed_pair_may_inject():
    """`--branch main --ab main` installs the same ref twice and launches the copies on `--port`
    and `--port + 1`, so their origins differ by construction. This is the null control the
    injection is most likely to be run against and it must keep working."""

    args = parse_args(["--branch", "main", "--ab", "main", "--inject-stream-cost-ms", "3"])
    assert (
        stream_cost_injection_problem(side_specs(args, "main"), args.inject_stream_cost_ms) is None
    )


def test_one_origin_without_the_injection_is_still_allowed():
    """THE CONTROL THAT MATTERS MOST. One attached Studio driven twice is a null control, and
    refusing it outright would remove the calibration run this tool exists to support."""

    args = _inject_args("http://127.0.0.1:5401", "http://127.0.0.1:5401")
    assert stream_cost_injection_problem(side_specs(args, "main"), None) is None


def test_a_single_side_is_never_refused():
    args = parse_args(["--branch", "main", "--inject-stream-cost-ms", "3"])
    assert stream_cost_injection_problem(side_specs(args, None), 3.0) is None


# ── one password per side ───────────────────────────────────────────────────────────────────


def test_each_attached_side_authenticates_with_its_own_password():
    args = parse_args(
        [
            "--attach",
            "http://127.0.0.1:5401",
            "--password",
            "base-secret",
            "--ab",
            "pr",
            "--attach-b",
            "http://127.0.0.1:5402",
            "--password-b",
            "treatment-secret",
        ]
    )
    specs = side_specs(args, "pr")
    assert [s[0] for s in specs] == ["base", "treatment"]
    assert specs[0][2] == "http://127.0.0.1:5401" and specs[0][4] == "base-secret"
    assert specs[1][2] == "http://127.0.0.1:5402" and specs[1][4] == "treatment-secret"


def test_the_treatment_falls_back_to_the_one_password():
    args = parse_args(
        ["--attach", "http://a", "--password", "one", "--ab", "pr", "--attach-b", "http://b"]
    )
    assert side_specs(args, "pr")[1][4] == "one"


def test_without_ab_there_is_one_side():
    args = parse_args(["--attach", "http://a", "--password", "one"])
    assert len(side_specs(args, None)) == 1


# ── one home per side ───────────────────────────────────────────────────────────────────────


def test_ab_sides_never_share_an_explicit_home():
    base = side_home("/tmp/home", "/out", "base", ab = True)
    treatment = side_home("/tmp/home", "/out", "treatment", ab = True)
    assert base != treatment
    assert base == Path("/tmp/home/base")


def test_a_single_side_still_installs_where_it_was_told():
    assert side_home("/tmp/home", "/out", "base", ab = False) == Path("/tmp/home")


def test_without_home_each_side_lands_under_the_output_directory():
    assert side_home(None, "/out", "treatment", ab = True) == Path("/out/studio_home_treatment")


# ── the doctor ──────────────────────────────────────────────────────────────────────────────


def test_an_engine_with_a_note_is_not_installed():
    assert engines_installed("chromium, webkit (not installed), firefox (unavailable)") == [
        "chromium"
    ]
    assert (
        engines_installed(
            "chromium (not installed), webkit (not installed), firefox (not installed)"
        )
        == []
    )
    assert engines_installed("chromium, webkit, firefox") == ["chromium", "webkit", "firefox"]


# ── the exit status ─────────────────────────────────────────────────────────────────────────


def test_a_fully_resumed_run_is_a_success():
    """Every cell was already complete, so the requested output exists. Exit 0."""

    assert completion_exit_code([], resumed = 4) == 0


def test_a_run_that_did_nothing_at_all_is_still_a_failure():
    assert completion_exit_code([], resumed = 0) == 1


def test_one_failed_cell_fails_the_run():
    assert completion_exit_code([{"completed": True}, {"completed": False}], resumed = 2) == 1
    assert completion_exit_code([{"completed": True}], resumed = 0) == 0


# ── the report ladder ───────────────────────────────────────────────────────────────────────


def _rows(
    rungs,
    cells,
    session = "s1",
):
    rows = [
        {
            "row_type": "run_meta",
            "tier": "standard",
            "rungs": rungs,
            "session_id": session,
            "corpus_hash": "c0ffee",
        }
    ]
    for tokens in cells:
        rows.append(
            {
                "row_type": "cell",
                "cell_id": f"r{tokens}",
                "session_id": session,
                "target_tokens": tokens,
                "completed": True,
                "cell": {"arm": "A0", "rep": 0},
            }
        )
        rows.append(
            {
                "row_type": "action",
                "cell_id": f"r{tokens}",
                "session_id": session,
                "action": "keystroke",
                "ran": True,
                "expect_ok": True,
                "timings": {"p95_ms": 20.0},
            }
        )
    return rows


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding = "utf-8")
    return path


def _payload(tmp_path, rungs, cells):
    return _write(tmp_path / "payload.jsonl", _rows(rungs, cells))


def _resumed_payload(tmp_path, first, then, cells):
    """One payload, two sessions: a run that promised `first`, and the `--resume` that promised
    `then`. A resume APPENDS its own `run_meta`, which is how a longer ladder gets into a file that
    already has one -- `rungs` is deliberately not a payload identity axis precisely so it can."""

    return _write(tmp_path / "payload.jsonl", _rows(first, cells) + _rows(then, [], session = "s2"))


def test_the_recorded_ladder_is_read_back_from_the_payload(tmp_path):
    path = _payload(tmp_path, ["1K", "10K", "100K"], [1_000, 10_000])
    assert recorded_ladder(path) == ["1K", "10K", "100K"]


def test_a_payload_without_a_run_meta_row_says_nothing(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"row_type":"cell","cell_id":"a"}\n', encoding = "utf-8")
    assert recorded_ladder(path) == []


def test_report_scores_the_rung_the_run_promised_and_never_reached(tmp_path, capsys):
    """The standard run died before 100K. Reporting it must not quietly become a quick run."""

    path = _payload(tmp_path, ["1K", "10K", "100K"], [1_000, 10_000])
    assert main(["--report", str(path)]) == 0
    summary = (tmp_path / "summary.md").read_text(encoding = "utf-8")
    assert "100,000" in summary or "100000" in summary or "100K" in summary


def test_a_rung_a_resume_added_is_still_owed_by_the_payload(tmp_path):
    """1K completed, resumed with `--rungs 1K,10K`, killed after the new header and before the cell.

    The ladder a payload owes is every rung any of its sessions promised. Reading the FIRST
    `run_meta` alone declared only 1K, and a continuation that never reached its new top rung
    scored COMPLETE -- the crash-beats-limp failure, arriving through the resume.
    """

    path = _resumed_payload(tmp_path, ["1K"], ["1K", "10K"], [1_000])
    assert recorded_ladder(path) == ["1K", "10K"]


def test_the_report_declares_the_rung_the_resume_promised_and_never_reached(tmp_path):
    path = _resumed_payload(tmp_path, ["1K"], ["1K", "10K"], [1_000])
    assert main(["--report", str(path)]) == 0
    summary = (tmp_path / "summary.md").read_text(encoding = "utf-8")
    assert "10,000" in summary
    assert "INCOMPLETE" in summary


def test_a_resume_that_promised_no_new_rung_reports_the_same_ladder(tmp_path):
    """The control: folding the headers may not invent a rung out of an ordinary continuation."""

    path = _resumed_payload(tmp_path, ["1K", "10K"], ["1K", "10K"], [1_000, 10_000])
    assert recorded_ladder(path) == ["1K", "10K"]
    assert main(["--report", str(path)]) == 0
    assert "100,000" not in (tmp_path / "summary.md").read_text(encoding = "utf-8")


def test_an_explicit_tier_still_wins(tmp_path):
    path = _payload(tmp_path, ["1K", "10K", "100K"], [1_000, 10_000])
    assert main(["--report", str(path), "--tier", "quick"]) == 0
    summary = (tmp_path / "summary.md").read_text(encoding = "utf-8")
    assert "100,000" not in summary and "100K" not in summary


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
