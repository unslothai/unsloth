# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests that reproduce the shape of four wrong published numbers, not just their class."""

from __future__ import annotations

from pathlib import Path

from tests.studio.studiobench.scoring import payload_rules


def _ladder_payload() -> list[dict]:
    """Fixture of the withdrawn headline run: 100K completes on both arms; the 1M treatment cell
    does not."""
    rows: list[dict] = []
    for cell_id, n_windows, completed in (
        ("r100K.base.rep0", 17, True),
        ("r100K.treatment.rep0", 17, True),
        ("r1M.base.rep0", 6, True),
        ("r1M.treatment.rep0", 7, False),
    ):
        for i in range(n_windows):
            rows.append(
                {
                    "row_type": "window",
                    "cell_id": cell_id,
                    "name": f"stream:gap{i}",
                    "kind": "gap",
                    "instruments": {"frames": {"frames_attempted": True, "fps": 28.7}},
                }
            )
        rows.append({"row_type": "cell", "cell_id": cell_id, "completed": completed})
        if not completed:
            rows.append({"row_type": "cell_aborted", "cell_id": cell_id, "reason": "budget"})
    return rows


def test_windows_of_an_unfinished_cell_are_not_offered_to_an_analysis():
    rows = _ladder_payload()
    kept = payload_rules.windows_of_completed_cells(rows)
    cells = {w["cell_id"] for w in kept}
    assert "r1M.treatment.rep0" not in cells, (
        "windows from a cell that never finished were offered for pooling. Reading them is what "
        "reported the 1M rung at 28.7 fps against a 46.7 fps baseline, a regression drawn "
        "entirely from an unfinished film."
    )
    assert cells == {"r100K.base.rep0", "r100K.treatment.rep0", "r1M.base.rep0"}
    assert len(kept) == 17 + 17 + 6


def test_an_unfinished_cell_is_announced_by_a_terminal_row():
    rows = _ladder_payload()
    assert payload_rules.aborted_cell_ids(rows) == {"r1M.treatment.rep0"}, (
        "a reader scanning forward must be able to discard an aborted cell's windows without "
        "joining backwards to the cell row it may not have reached yet"
    )


def test_completed_cell_ids_ignores_rows_that_are_not_cells():
    rows = _ladder_payload() + [{"row_type": "action", "cell_id": "r1M.treatment.rep0"}]
    assert "r1M.treatment.rep0" not in payload_rules.completed_cell_ids(rows)


def test_census_peak_is_refused_for_cross_arm_comparison():
    why = payload_rules.refuse_uncomparable("census_peak")
    assert why is not None, (
        "census_peak was offered for differencing across arms. It is chosen by a max() over "
        "per-action censuses that race their own teardown, and on a NULL CONTROL -- one bundle "
        "on both sides -- the winner flipped between `settings` (panes collapsed) and "
        "`reasoning_toggle` (panes open), a 70.1% swing within one arm. Differencing it reported "
        "main as mounting 48% more Shiki spans than a baseline that in fact mounts the same "
        "document to within 0.3%."
    )
    assert "70.1%" in why
    assert payload_rules.refuse_uncomparable("census_peak.highlight_spans") is not None


def test_a_normal_metric_is_still_comparable():
    assert payload_rules.refuse_uncomparable("select_all_copy.copy_ms") is None


def _censored_payload() -> list[dict]:
    """`open_ms` measured at 100K and censored at 500K, which is what the ladder actually did."""
    rows: list[dict] = []
    for cell_id, censored in (
        ("r100K.base.rep0", False),
        ("r100K.treatment.rep0", False),
        ("r500K.base.rep0", True),
        ("r500K.treatment.rep0", True),
    ):
        rows.append({"row_type": "cell", "cell_id": cell_id, "completed": True})
        rows.append(
            {
                "row_type": "action",
                "cell_id": cell_id,
                "action": "reasoning_toggle",
                "ran": True,
                "expect": {
                    "open_censored": censored,
                    "close_censored": False,
                    "open_censored_reason": "the span census was still changing"
                    if censored
                    else None,
                },
                "timings": {} if censored else {"open_ms": 1777.8},
            }
        )
    return rows


def test_a_metric_censored_at_one_rung_is_refused_for_pooling():
    rows = _censored_payload()
    why = payload_rules.refuse_partial_censoring(rows, "reasoning_toggle.open_ms")
    assert why is not None, (
        "open_ms is censored on every cell above 100K, and a censored timing is simply ABSENT "
        "from `timings`. Pooled, the ladder row silently becomes a 100K-only number wearing a "
        "ladder label, and the cells that survive are the fast ones."
    )
    assert "r500K" in why and "r100K" in why


def test_censored_cells_are_enumerable():
    rows = _censored_payload()
    censored = payload_rules.censored_metrics(rows)
    assert censored["reasoning_toggle.open_ms"] == {"r500K.base.rep0", "r500K.treatment.rep0"}


def test_a_metric_measured_everywhere_is_poolable():
    rows = _censored_payload()
    assert payload_rules.refuse_partial_censoring(rows, "reasoning_toggle.close_ms") is None


def test_an_unsettled_census_is_not_marked_settled():
    unsettled = {
        "row_type": "action",
        "action": "reasoning_toggle",
        "expect": {"settled": False, "highlight_spans_while_open": None},
    }
    settled = {
        "row_type": "action",
        "action": "reasoning_toggle",
        "expect": {"settled": True, "highlight_spans_while_open": 74250},
    }
    assert not payload_rules.settled(unsettled)
    assert payload_rules.settled(settled)


def _meta(
    corpus_hash: str,
    tier: str = "full",
    engine: str = "chromium",
) -> dict:
    return {
        "row_type": "run_meta",
        "corpus_hash": corpus_hash,
        "tier": tier,
        "rungs": ["100K", "500K", "1M"],
        "tool_version": "0.1.0",
        "instrument_level": 0,
        "cadence": "field",
        "stream_tail_chars": None,
        "corpus_dollars": False,
        "platform": {"engine": engine},
    }


def test_a_probed_or_calibration_run_is_not_comparable_with_a_clean_one():
    """A probed or injected run must not be comparable with a clean one; they change what is measured."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    clean = _meta(corpus)
    for field, probed_value in (
        ("inject_stream_cost_ms", 40.0),
        ("click_probe", True),
        ("probe_init_script", "/tmp/probe.js"),
    ):
        probed = _meta(corpus)
        probed[field] = probed_value
        assert payload_rules.comparability_key(clean) != payload_rules.comparability_key(probed), (
            f"a run with {field}={probed_value!r} carries the same comparability key as a clean "
            f"run, so `--compare` calls them comparable"
        )
        assert any(
            line.startswith(f"{field}:")
            for line in payload_rules.explain_incomparable(clean, probed)
        )


def test_a_payload_written_before_the_probe_fields_existed_still_matches_a_clean_run():
    """click_probe is read through bool, so legacy payloads that omit it still match clean runs."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    legacy = _meta(corpus)
    legacy.pop("click_probe", None)
    current = _meta(corpus)
    current["click_probe"] = False
    assert payload_rules.comparability_key(legacy) == payload_rules.comparability_key(current)


def test_a_resumed_payload_is_keyed_on_the_ladder_it_grew_into(tmp_path):
    """A resumed payload is keyed on the ladder it grew into, not the one-rung header it started as."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    first = _meta(corpus)
    first["rungs"] = ["100K"]
    second = _meta(corpus)
    second["rungs"] = ["100K", "500K"]

    merged, conflicts = payload_rules.merged_run_meta([first, second])
    assert conflicts == []
    assert merged["rungs"] == [
        "100K",
        "500K",
    ], "the merge kept the ladder the run STARTED with rather than the one the file describes"
    one_rung = payload_rules.comparability_key(first)
    assert payload_rules.comparability_key(merged) != one_rung


def test_a_payload_whose_headers_disagree_has_no_key_at_all():
    """Headers that disagree on tool_version yield no key at all, since neither value describes the file."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    before = _meta(corpus)
    before["tool_version"] = "0.1.0"
    after = _meta(corpus)
    after["tool_version"] = "0.2.0"
    _merged, conflicts = payload_rules.merged_run_meta([before, after])
    assert conflicts == ["tool_version: '0.1.0' != '0.2.0'"]


def test_the_key_is_computed_over_the_fields_it_explains():
    """The key is hashed over the same fields its explanation lists, so the two cannot drift apart."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    a = _meta(corpus)
    nested = {"engine", "system", "machine", "node"}
    for field in payload_rules.comparability_fields(a):
        b = _meta(corpus)
        if field in nested:
            b["platform"] = dict(b.get("platform") or {})
            b["platform"][field] = "CHANGED"
        else:
            b[field] = "CHANGED"
        assert payload_rules.comparability_key(a) != payload_rules.comparability_key(
            b
        ), f"{field} is listed in comparability_fields but does not move the key"


def test_a_payload_from_another_host_is_not_comparable():
    """Engine cannot identify the host: Darwin and Linux both default to webkit, so platform is keyed."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    linux = _meta(corpus)
    linux["platform"] = {"engine": "webkit", "system": "Linux", "machine": "x86_64"}
    macos = _meta(corpus)
    macos["platform"] = {"engine": "webkit", "system": "Darwin", "machine": "arm64"}
    assert payload_rules.comparability_key(linux) != payload_rules.comparability_key(macos)
    differ = payload_rules.explain_incomparable(linux, macos)
    assert any(line.startswith("system:") for line in differ)
    assert any(line.startswith("machine:") for line in differ)


def test_two_linux_boxes_are_not_one_host():
    """Two Linux x86_64 hosts share system and machine, so the node name must be keyed or they match."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    here, there = _meta(corpus), _meta(corpus)
    here["platform"] = {
        "engine": "webkit",
        "system": "Linux",
        "machine": "x86_64",
        "node": "devbox-a",
    }
    there["platform"] = {
        "engine": "webkit",
        "system": "Linux",
        "machine": "x86_64",
        "node": "ci-runner-7",
    }
    assert payload_rules.comparability_key(here) != payload_rules.comparability_key(there), (
        "two different Linux x86_64 machines carry the same comparability key, so `--compare` "
        "certifies one machine's numbers against the other's"
    )
    assert any(line.startswith("node:") for line in payload_rules.explain_incomparable(here, there))

    same = _meta(corpus)
    same["platform"] = dict(here["platform"])
    assert payload_rules.comparability_key(here) == payload_rules.comparability_key(
        same
    ), "the same host on the same settings must still be comparable with itself"


def test_two_payloads_from_before_the_host_was_recorded_still_match_each_other():
    """Payloads recorded before node was written match each other, but not one that names a host."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    a, b = _meta(corpus), _meta(corpus)
    for row in (a, b):
        row["platform"] = {"engine": "webkit", "system": "Linux", "machine": "x86_64"}
    assert payload_rules.comparability_key(a) == payload_rules.comparability_key(b)
    named = _meta(corpus)
    named["platform"] = {
        "engine": "webkit",
        "system": "Linux",
        "machine": "x86_64",
        "node": "devbox-a",
    }
    assert payload_rules.comparability_key(a) != payload_rules.comparability_key(named)


def test_the_harness_records_the_host_it_ran_on():
    """A key field is inert unless run_meta carries it, so the harness must record platform.node."""
    import platform as _platform

    from tests.studio.studiobench import __main__ as sb_main

    source = Path(sb_main.__file__).read_text(encoding = "utf-8")
    assert "platform.node()" in source, (
        "run_meta does not record the host, so the `node` field of the comparability key is None "
        "on every payload and cannot separate two machines"
    )
    assert isinstance(_platform.node(), str)


def test_a_run_from_before_the_settling_fix_is_not_comparable_with_one_from_after():
    """Across the settling fix the corpus is unchanged, so only tool_version tells the two runs apart."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    before = _meta(corpus)
    before["tool_version"] = "0.1.0"
    after = _meta(corpus)
    after["tool_version"] = "0.2.0"
    assert payload_rules.comparability_key(before) != payload_rules.comparability_key(after)
    assert payload_rules.explain_incomparable(before, after) == ["tool_version: '0.1.0' != '0.2.0'"]


def test_the_shipping_tool_version_is_the_settled_one():
    """The bump has to be on the constant the harness actually stamps, not only in a fixture."""
    from tests.studio.studiobench.__main__ import TOOL_VERSION
    assert TOOL_VERSION != "0.1.0", (
        "TOOL_VERSION still reads 0.1.0, so every payload written after the settling fix carries "
        "the same comparability key as one written before it, and `--compare` reports the two as "
        "comparable."
    )


def test_the_key_is_keyed_on_the_computed_corpus_hash_not_the_harness_commit():
    old = _meta("f113503e6cea3574e9c1a99653ee140904ab24f7606505f9e0d8d66efda96070")
    new = _meta("ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c")
    assert payload_rules.comparability_key(old) != payload_rules.comparability_key(new), (
        "two runs on different corpora produced the same comparability key. A tree in this "
        "campaign ran a third corpus, silently and self-consistently, and the only reason anyone "
        "noticed is that the hash was printed. A commit is a claim; the hash is the thing."
    )
    assert payload_rules.explain_incomparable(old, new) == [
        "corpus_hash: 'f113503e6cea3574e9c1a99653ee140904ab24f7606505f9e0d8d66efda96070' != "
        "'ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c'"
    ]


def test_the_key_is_stable_for_identical_runs():
    a = _meta("ac9d5d8e")
    b = _meta("ac9d5d8e")
    assert payload_rules.comparability_key(a) == payload_rules.comparability_key(b)
    assert payload_rules.explain_incomparable(a, b) == []


def test_the_key_moves_with_the_engine_and_the_tier():
    base = _meta("ac9d5d8e")
    assert payload_rules.comparability_key(base) != payload_rules.comparability_key(
        _meta("ac9d5d8e", engine = "webkit")
    )
    assert payload_rules.comparability_key(base) != payload_rules.comparability_key(
        _meta("ac9d5d8e", tier = "standard")
    )


def test_the_key_looks_like_a_token_that_can_be_quoted():
    key = payload_rules.comparability_key(_meta("ac9d5d8e"))
    assert key.startswith("cmp:") and len(key) == len("cmp:") + 10


def _resumed_window_payload() -> list[dict]:
    """Fixture: a cell that died at 28.7 fps and its completed retry at 46.7 fps share one cell id."""
    cid = "r1M.treatment.rep0"

    def win(session: str, i: int, fps: float, frame_ms: float) -> dict:
        return {
            "row_type": "window",
            "cell_id": cid,
            "session_id": session,
            "name": f"stream:gap{i}",
            "kind": "gap",
            "t_open_ms": float(i * 100),
            "duration_ms": 33.0,
            "instruments": {"frames": {"fps": fps, "max_frame_ms": frame_ms, "long_frames": []}},
        }

    rows = [win("s1aborted", i, 28.7, 34.84) for i in range(7)]
    rows.append({"row_type": "cell", "cell_id": cid, "session_id": "s1aborted", "completed": False})
    rows.append(
        {"row_type": "cell_aborted", "cell_id": cid, "session_id": "s1aborted", "reason": "died"}
    )
    rows += [win("s2resumed", i, 46.7, 21.41) for i in range(6)]
    rows.append({"row_type": "cell", "cell_id": cid, "session_id": "s2resumed", "completed": True})
    return rows


def test_the_windows_of_a_dead_attempt_do_not_come_back_with_the_retry():
    """Matching on cell id alone returns the dead attempt's windows, inventing a jank score."""
    got = payload_rules.windows_of_completed_cells(_resumed_window_payload())
    sessions = {r.get("session_id") for r in got}
    assert sessions == {
        "s2resumed"
    }, f"windows came back from {sorted(sessions)}. The dead attempt's frames became the retry's."
    assert len(got) == 6
    fps = [r["instruments"]["frames"]["fps"] for r in got]
    assert set(fps) == {46.7}


def test_subtracting_the_aborted_ids_is_not_the_fix():
    """Subtracting aborted_cell_ids is wrong: a resumed id is in both sets and the good reading is lost."""
    rows = _resumed_window_payload()
    done = payload_rules.completed_cell_ids(rows)
    aborted = payload_rules.aborted_cell_ids(rows)
    assert done and aborted
    assert done - aborted == set()


def test_a_payload_with_no_resumed_attempt_is_unchanged():
    """The reduction must not drop windows from an ordinary single-attempt payload."""
    rows = [
        {
            "row_type": "window",
            "cell_id": "r100K.base.rep0",
            "session_id": "s1",
            "name": "stream:gap0",
            "kind": "gap",
            "instruments": {"frames": {"fps": 60.0}},
        },
        {"row_type": "cell", "cell_id": "r100K.base.rep0", "session_id": "s1", "completed": True},
    ]
    assert len(payload_rules.windows_of_completed_cells(rows)) == 1


def test_a_headed_run_is_not_comparable_with_a_headless_one():
    """Engine does not identify the renderer: headed and headless use different binaries and compositors."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    headless = _meta(corpus)
    headed = _meta(corpus)
    headed["headed"] = True
    assert payload_rules.comparability_key(headless) != payload_rules.comparability_key(headed)
    assert payload_rules.explain_incomparable(headless, headed) == ["headed: False != True"]


def test_a_payload_written_before_the_headed_field_reads_as_headless():
    """Absence is the headless default, not a third state that orphans every older payload."""
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    legacy = _meta(corpus)
    legacy.pop("headed", None)
    current = _meta(corpus)
    current["headed"] = False
    assert payload_rules.comparability_key(legacy) == payload_rules.comparability_key(current)


def test_the_launch_mode_is_an_identity_axis_a_resume_cannot_toggle():
    """headed is an identity axis: resume must refuse to toggle it, or one ladder mixes two browsers."""
    from tests.studio.studiobench.__main__ import HISTORICAL_DEFAULTS, IDENTITY_AXES

    assert "headed" in IDENTITY_AXES
    assert HISTORICAL_DEFAULTS["headed"] is False


def test_the_instrument_version_is_an_identity_axis_a_resume_cannot_cross():
    """The refusal must come before measuring, since detecting a tool_version mix afterwards is too late."""
    from tests.studio.studiobench.__main__ import IDENTITY_AXES, TOOL_VERSION, identity_problems

    assert "tool_version" in IDENTITY_AXES
    recorded = _meta("ac9d5d8e")
    recorded["tool_version"] = "0.1.0"
    requested = {"tool_version": TOOL_VERSION}
    problems = identity_problems(recorded, requested)
    assert any(p.startswith("tool_version:") for p in problems), problems


def test_compare_reads_an_interrupted_payload_instead_of_raising(tmp_path, capsys):
    """A payload torn by a kill ends in a malformed line by design, so --compare must skip it, not raise."""
    import json

    from tests.studio.studiobench.__main__ import main

    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    rows = [json.dumps(_meta(corpus)), json.dumps({"row_type": "cell", "cell_id": "r100K.A0.rep0"})]
    whole = tmp_path / "whole.jsonl"
    whole.write_text("\n".join(rows) + "\n", encoding = "utf-8")
    torn = tmp_path / "torn.jsonl"
    torn.write_text(
        "\n".join(rows) + "\n" + json.dumps({"row_type": "cell", "cell_id": "r100K"})[:18],
        encoding = "utf-8",
    )

    assert main(["--compare", str(torn), str(whole)]) == 0
    out = capsys.readouterr().out
    assert "malformed line" in out, out
    assert "comparable: every field the key covers matches." in out, out
