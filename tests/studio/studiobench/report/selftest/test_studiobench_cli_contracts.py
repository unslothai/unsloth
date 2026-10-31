# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CLI decisions that decide what a run means, each a one-line condition whose failure was silent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.__main__ import (  # noqa: E402
    completion_exit_code,
    engines_installed,
    _ab_label,
    is_null_control,
    main,
    parse_args,
    planned_rungs,
    recorded_ladder,
    side_home,
    side_specs,
    stream_cost_injection_problem,
)


def _side(
    label,
    ref,
    url,
    owns,
    commit = None,
):
    side = {"label": label, "ref": ref, "base_url": url, "owns": owns}
    if commit is not None:
        side["commit"] = commit
    return side


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


# Stream-cost injection is gated on window.location.origin, so the two arms need distinct origins.


def _inject_args(attach, attach_b, *extra):
    return parse_args(["--attach", attach, "--ab", "main", "--attach-b", attach_b, *extra])


def test_injecting_stream_cost_into_two_arms_on_one_origin_is_refused():
    """REGRESSION. `--attach U --attach-b U` is the cheapest null control there is -- one Unsloth,
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


@pytest.mark.parametrize(
    ("attach", "attach_b"),
    [
        ("http://studio", "http://studio:80"),
        ("http://studio:80", "http://studio"),
        ("https://studio.example.com", "https://studio.example.com:443"),
        ("http://studio", "http://STUDIO"),
        ("http://studio", "HTTP://studio"),
        ("http://studio", "http://studio/app"),
    ],
)
def test_one_origin_spelled_two_ways_is_still_one_origin(attach, attach_b):
    """One origin under many spellings: http://studio:80 and HTTP://STUDIO must both match http://studio."""

    args = _inject_args(attach, attach_b, "--inject-stream-cost-ms", "3")
    problem = stream_cost_injection_problem(side_specs(args, "main"), args.inject_stream_cost_ms)

    assert problem is not None, f"{attach} and {attach_b} are one origin"
    assert "origin" in problem
    assert attach in problem and attach_b in problem


def test_the_predicate_and_the_refusal_read_the_same_origin():
    """The two halves of this guard are in different files and only agree by construction. If
    `origin_scoped` gated on the URL as typed while `arm_origins` canonicalised, a pair that got
    past the refusal could still be gated onto a document that does not exist."""

    from studiobench.__main__ import arm_origins
    from studiobench.runtime.ab import origin_scoped

    args = _inject_args("http://studio:80", "https://other.example.com:443")
    for spec, origin in zip(side_specs(args, "main"), arm_origins(side_specs(args, "main"))):
        assert f'"{origin}"' in origin_scoped(spec[2], "doThing();")


def test_localhost_and_the_loopback_address_are_two_origins():
    """localhost and 127.0.0.1 are two browser origins, so canonicalisation must not fold them together."""

    args = _inject_args(
        "http://localhost:8000", "http://127.0.0.1:8000", "--inject-stream-cost-ms", "3"
    )
    assert (
        stream_cost_injection_problem(side_specs(args, "main"), args.inject_stream_cost_ms) is None
    )


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
    """THE CONTROL THAT MATTERS MOST. One attached Unsloth driven twice is a null control, and
    refusing it outright would remove the calibration run this tool exists to support."""

    args = _inject_args("http://127.0.0.1:5401", "http://127.0.0.1:5401")
    assert stream_cost_injection_problem(side_specs(args, "main"), None) is None


def test_a_single_side_is_never_refused():
    args = parse_args(["--branch", "main", "--inject-stream-cost-ms", "3"])
    assert stream_cost_injection_problem(side_specs(args, None), 3.0) is None


def test_one_attached_studio_driven_twice_under_two_labels_is_still_a_null_control():
    """One attached server under two labels is a null control: the URL, not the ref, names the build."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5401", False),
        _side("treatment", "fix", "http://127.0.0.1:5401", False),
    ]
    assert is_null_control(sides) is True


def test_an_attached_treatment_pointed_at_the_installed_base_is_a_null_control():
    """The mixed form of the same thing: `--attach-b` naming the Unsloth this run just launched."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True, commit = "a" * 40),
        _side("treatment", "fix", "http://127.0.0.1:5399", False),
    ]
    assert is_null_control(sides) is True


def test_two_owned_installs_of_different_refs_are_still_an_ordinary_ab():
    """The control for the reorder: owned sides get `port` and `port + 1`, so they never collide
    on a URL and the ref and commit comparisons below still decide them."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True, commit = "a" * 40),
        _side("treatment", "fix", "http://127.0.0.1:5400", True, commit = "b" * 40),
    ]
    assert is_null_control(sides) is False


def test_a_ref_that_moved_between_the_two_installs_is_not_a_null_control():
    """A ref that advanced between the two installs is not a null control: same ref, different builds."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True, commit = "a" * 40),
        _side("treatment", "main", "http://127.0.0.1:5400", True, commit = "b" * 40),
    ]
    assert is_null_control(sides) is False


def test_two_installs_of_one_ref_that_resolved_to_one_commit_are_a_null_control():
    """The control: the calibration run this tool exists to support still classifies."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True, commit = "a" * 40),
        _side("treatment", "main", "http://127.0.0.1:5400", True, commit = "a" * 40),
    ]
    assert is_null_control(sides) is True


def test_a_side_with_no_commit_to_declare_is_judged_as_before():
    """An empty commit on either side is not a difference -- the rule `commit_problems` already
    states. Nothing behind an attached URL declares a commit, and neither does a payload written
    before commits were recorded, so the build check must not start refusing them."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True, commit = "a" * 40),
        _side("treatment", "main", "http://127.0.0.1:5400", True, commit = ""),
    ]
    assert is_null_control(sides) is True


def test_the_table_names_both_builds_when_one_ref_resolved_to_two():
    """ "main -> main" over two different commits would read as a null control on a screenshot."""

    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True, commit = "a" * 40),
        _side("treatment", "main", "http://127.0.0.1:5400", True, commit = "b" * 40),
    ]
    label = _ab_label(sides, is_null_control(sides))
    assert label == "main aaaaaaaaaaaa -> bbbbbbbbbbbb"


def test_a_null_control_states_the_commit_it_compared_with_itself():
    sides = [
        _side("base", "main", "http://127.0.0.1:5399", True, commit = "a" * 40),
        _side("treatment", "main", "http://127.0.0.1:5400", True, commit = "a" * 40),
    ]
    assert _ab_label(sides, True) == "null control: main @ aaaaaaaaaaaa vs itself"


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


def test_ab_sides_never_share_an_explicit_home():
    base = side_home("/tmp/home", "/out", "base", ab = True)
    treatment = side_home("/tmp/home", "/out", "treatment", ab = True)
    assert base != treatment
    assert base == Path("/tmp/home/base")


def test_a_single_side_still_installs_where_it_was_told():
    assert side_home("/tmp/home", "/out", "base", ab = False) == Path("/tmp/home")


def test_without_home_each_side_lands_under_the_output_directory():
    assert side_home(None, "/out", "treatment", ab = True) == Path("/out/studio_home_treatment")


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


def test_a_fully_resumed_run_is_a_success():
    """Every cell was already complete, so the requested output exists. Exit 0."""

    assert completion_exit_code([], resumed = 4) == 0


def test_a_run_that_did_nothing_at_all_is_still_a_failure():
    assert completion_exit_code([], resumed = 0) == 1


def test_one_failed_cell_fails_the_run():
    assert completion_exit_code([{"completed": True}, {"completed": False}], resumed = 2) == 1
    assert completion_exit_code([{"completed": True}], resumed = 0) == 0


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
    """The ladder is every rung any session promised, so a resume's added rung is still owed."""

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


def _rungs(value):
    return planned_rungs(parse_args(["--tier", "standard", "--rungs", value]))


def test_a_space_after_the_comma_is_the_same_ladder():
    """Spaces around the commas in --rungs must not leave a rung name with a leading space."""

    assert _rungs("1K, 10K") == ["1K", "10K"]
    assert _rungs(" 1K , 10K ") == ["1K", "10K"]


def test_a_lowercase_suffix_is_the_same_ladder():
    assert _rungs("1k,10k") == ["1K", "10K"]
    assert _rungs("10k, 1m") == ["10K", "1M"]


def test_an_empty_field_is_not_a_rung():
    """A trailing or doubled comma is a typo, not a request for a nameless rung."""

    assert _rungs("1K,,10K") == ["1K", "10K"]
    assert _rungs("1K,10K,") == ["1K", "10K"]


def test_a_label_that_is_not_a_rung_is_refused_by_name():
    for value in ("1X", "2K", "1K,20K"):
        with pytest.raises(SystemExit) as excinfo:
            _rungs(value)
        assert "is not a rung" in str(excinfo.value)


def test_a_value_naming_no_rung_at_all_is_refused():
    for value in (",", " "):
        with pytest.raises(SystemExit) as excinfo:
            _rungs(value)
        assert "names nothing" in str(excinfo.value)


def test_the_tier_still_supplies_the_ladder_when_rungs_is_not_given():
    """The control: normalising the override may not disturb the default path."""

    assert planned_rungs(parse_args(["--tier", "standard"])) == ["1K", "10K", "100K"]
    assert planned_rungs(parse_args(["--tier", "fast"])) == ["100K"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
