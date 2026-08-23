# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for four numbers this harness published and somebody acted on.

Each test reproduces the SHAPE of the specific wrong number, not merely the class of bug, so that
a future change which reintroduces the defect fails here rather than in a report two days later.
Every one of them fails on the tree before `payload_rules` existed.

The four share one root cause: measuring at a moment whose meaning is not stable across the things
being compared.
"""

from __future__ import annotations

from tests.studio.studiobench.scoring import payload_rules


# ── defect 4: window rows outlive their cell ────────────────────────────────


def _ladder_payload() -> list[dict]:
    """A 100K rung that finished on both arms, and a 1M rung whose treatment cell did not.

    The numbers are the ones from the run that produced the withdrawn headline: the completed 1M
    cell emitted 6 `stream:gap` windows where 100K emitted 17, and the unfinished treatment cell
    emitted 7 -- enough to look complete to a reader that does not check.
    """
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


# ── defect 1: census_peak is not comparable across arms ─────────────────────


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


# ── defect 2: a metric censored at some rungs but not others ───────────────


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


# ── defect 7: a census is only a census if the DOM had settled ─────────────


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


# ── the comparability key ──────────────────────────────────────────────────


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
    """The three fields that change what is MEASURED, not merely what it is measured on.

    All three are `IDENTITY_AXES`, which is to say `--resume` refuses to toggle them mid-payload.
    A comparability key that ignored them would call two payloads comparable that the harness
    itself refuses to continue as one run.

    `inject_stream_cost_ms` is the one that could publish a number: an arm running it is not a
    measurement of the build, because the harness added the slowdown deliberately, and nothing in
    the scoring path refuses an injected payload. Blessing it against a clean run is a direct route
    to reporting a difference the harness created.
    """
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
    """`click_probe` normalises through bool, so a legacy payload is not gratuitously orphaned.

    A refusal that invalidated every older payload would be a cost with no safety in it: absent
    and False mean the same thing here, which is why the harness reads absence as a value.
    """
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    legacy = _meta(corpus)
    legacy.pop("click_probe", None)
    current = _meta(corpus)
    current["click_probe"] = False
    assert payload_rules.comparability_key(legacy) == payload_rules.comparability_key(current)


def test_a_resumed_payload_is_keyed_on_the_ladder_it_grew_into(tmp_path):
    """Reading only the FIRST run_meta hashes a ladder the file has since outgrown.

    `--resume` appends a second header, and it may legitimately have extended the ladder:
    `IDENTITY_AXES` leaves `rungs` out on purpose because adding a rung ADDS cells rather than
    reinterpreting recorded ones. Keyed on the first header, a payload resumed from one rung to two
    carries the same key as the one-rung run it started as, and `--compare` pronounces them
    comparable while the payload's own later comparability row says otherwise.
    """
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
    """Two runs in one file that were not measuring the same thing cannot share a token.

    `tool_version` needs no exotic invocation to differ: resume a half-finished payload after
    pulling a harness upgrade and header one reads 0.1.0 while header two reads 0.2.0. Neither
    value describes the file.
    """
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    before = _meta(corpus)
    before["tool_version"] = "0.1.0"
    after = _meta(corpus)
    after["tool_version"] = "0.2.0"
    _merged, conflicts = payload_rules.merged_run_meta([before, after])
    assert conflicts == ["tool_version: '0.1.0' != '0.2.0'"]


def test_the_key_is_computed_over_the_fields_it_explains():
    """The token and its explanation must not be able to drift apart.

    They were written as two separate dicts kept in step by hand. A field added to one and missed
    in the other would make `--compare` hash something its own "these differ" list never mentions,
    which is the provenance failure this module argues against, committed by the module itself.
    """
    corpus = "ac9d5d8e37be2a3844deed559fde6070247ad2322377295fb383b60b5eec5a0c"
    a = _meta(corpus)
    for field in payload_rules.comparability_fields(a):
        b = _meta(corpus)
        if field == "engine":
            b["platform"] = {"engine": "webkit"}
        else:
            b[field] = "CHANGED"
        assert payload_rules.comparability_key(a) != payload_rules.comparability_key(
            b
        ), f"{field} is listed in comparability_fields but does not move the key"


def test_a_run_from_before_the_settling_fix_is_not_comparable_with_one_from_after():
    """The corpus did not change here, so `tool_version` is the only field that can separate them.

    This change redefines two published measures on an UNCHANGED corpus:
    `highlight_spans_while_open` reads 74,250 where it read 44,075 on the same bundle, and
    `open_ms` now terminates on a settled mount rather than on the `data-state` flip. Every other
    field the key covers -- corpus hash, tier, rungs, engine, cadence -- is identical across the
    change. If the key could not tell those two runs apart it would certify as comparable the pair
    that differs by the largest instrument change in the campaign, which is the failure it exists
    to prevent.
    """
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
