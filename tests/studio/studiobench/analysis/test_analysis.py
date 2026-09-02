# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the studiobench analysis layer, against a REAL captured trace.

The fixture in `testdata/` is a genuine Chrome trace captured through
`instruments/tracing.py`, trimmed to the renderer main thread plus the V8
profiler events. The page it was captured from does three things with known
counts, which is what makes the assertions below meaningful rather than
self-referential:

  * a `setInterval` loop, so there are timer tasks;
  * a `MessageChannel` ping-pong of exactly 120 round trips, which is the shape
    the React scheduler uses and the class this tool most needs to get right;
  * a `requestAnimationFrame` loop, so there are frame tasks;
  * real `page.keyboard` typing, so there are input tasks with `latencyInfo`.

Run with `python -m pytest tests/studio/studiobench/analysis/test_analysis.py`,
or standalone with `python tests/studio/studiobench/analysis/test_analysis.py`.
No browser and no network required.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDIO_TESTS = os.path.dirname(os.path.dirname(_HERE))
if _STUDIO_TESTS not in sys.path:
    sys.path.insert(0, _STUDIO_TESTS)

from studiobench.analysis import CellFailure  # noqa: E402
from studiobench.analysis import assert_no_bare_zero, measured, merge, unmeasured  # noqa: E402
from studiobench.analysis import classify as K  # noqa: E402
from studiobench.analysis import cpuprofile as C  # noqa: E402
from studiobench.analysis import fit as F  # noqa: E402
from studiobench.analysis import oracles as O  # noqa: E402
from studiobench.analysis import symbols as S  # noqa: E402
from studiobench.analysis.traceparse import Trace, build_tree  # noqa: E402

TRACE = os.path.join(_HERE, "testdata", "probe_msgchan_timer_raf.json.gz")

# Ground truth about the page the fixture was captured from.
EXPECTED_MESSAGE_CHANNEL_TASKS = 120
EXPECTED_RAF_ITERATIONS = 60


def _trace() -> Trace:
    return Trace.from_path(TRACE)


# --------------------------------------------------------------- traceparse


def test_loads_object_form_trace() -> None:
    tr = _trace()
    assert len(tr.events) > 1000
    assert tr.metadata, "metadata block should survive the round trip"


def test_truncated_trace_fails_the_cell_rather_than_parsing_short() -> None:
    # A stream cut off mid-document must not degrade into 'a shorter trace'; that reads exactly like
    # the expensive work not happening.
    try:
        Trace.from_json_text('{"traceEvents":[{"ph":"X","ts":1,"dur":2,"name":"RunTask"')
    except CellFailure as exc:
        assert exc.gate == "trace_truncated"
    else:
        raise AssertionError("a truncated trace must raise CellFailure")


def test_profiled_thread_is_the_renderer_main_not_the_profiler_thread() -> None:
    tr = _trace()
    pid, tid = tr.profiled_thread()
    assert tr.thread_name(pid, tid) == "CrRendererMain"
    # The chunks themselves live on a different thread; that is the trap.
    chunk_tids = {e.get("tid") for e in tr.events if e.get("name") == "ProfileChunk"}
    assert chunk_tids and chunk_tids != {tid}, (
        "fixture should exercise the case where ProfileChunk is emitted on the "
        "V8 profiler thread rather than the profiled thread"
    )


def test_tree_nests_and_self_time_never_goes_negative() -> None:
    tr = _trace()
    th = tr.renderer_main()
    for task in tr.run_tasks(th):
        for node in task.walk():
            assert node.self_dur >= 0
            assert sum(c.dur for c in node.children) <= node.dur + 1


def test_overlapping_events_become_siblings_not_corrupt_parents() -> None:
    # An event that starts inside another and ends after it is not nested.
    events = [
        {"ph": "X", "ts": 0, "dur": 100, "name": "outer", "cat": "c", "pid": 1, "tid": 1},
        {"ph": "X", "ts": 50, "dur": 200, "name": "straddles", "cat": "c", "pid": 1, "tid": 1},
    ]
    roots = build_tree(events)
    assert len(roots) == 2, "a straddling event must not be adopted as a child"


def test_begin_end_pairs_fold_into_complete_events() -> None:
    events = [
        {"ph": "B", "ts": 10, "name": "x", "cat": "c", "pid": 1, "tid": 1, "args": {"a": 1}},
        {"ph": "E", "ts": 40, "name": "x", "cat": "c", "pid": 1, "tid": 1, "args": {"b": 2}},
    ]
    roots = build_tree(events)
    assert len(roots) == 1 and roots[0].dur == 30
    assert roots[0].args == {"a": 1, "b": 2}


def test_unmatched_begin_invents_no_duration() -> None:
    roots = build_tree([{"ph": "B", "ts": 10, "name": "x", "cat": "c", "pid": 1, "tid": 1}])
    assert roots == []


# --------------------------------------------------------------- cpuprofile


def test_profile_parses_and_deltas_match_wall_clock() -> None:
    prof = C.main_thread_profile(_trace())
    report = prof.assert_deltas_match_wall()
    assert report["drift"] < C.DELTA_WALL_TOLERANCE
    assert report["sample_count"] > 1000


def test_nodes_accumulate_across_chunks() -> None:
    # Only a handful of chunks carry a `nodes` array; a parser that reads nodes per chunk and forgets
    # them resolves almost nothing.
    tr = _trace()
    chunks_with_nodes = sum(
        1
        for e in tr.events
        if e.get("name") == "ProfileChunk"
        and ((e.get("args") or {}).get("data") or {}).get("cpuProfile", {}).get("nodes")
    )
    prof = C.main_thread_profile(tr)
    assert chunks_with_nodes < prof.chunk_count, "fixture must exercise incremental nodes"
    assert len(prof.nodes) > chunks_with_nodes
    resolved = sum(1 for s in prof.samples if s.node_id in prof.nodes)
    assert resolved == len(prof.samples), "every sample must resolve to a known node"


def test_negative_time_deltas_are_summed_not_clamped() -> None:
    prof = C.main_thread_profile(_trace())
    assert prof.negative_deltas > 0, "fixture must exercise negative deltas"
    # Clamping would inflate the total past wall clock and break the gate above.
    naive = sum(max(0, s.delta) for s in prof.samples)
    honest = sum(s.delta for s in prof.samples)
    assert naive > honest


def test_ragged_chunk_fails_the_cell() -> None:
    tr = _trace()
    for e in tr.events:
        if e.get("name") == "ProfileChunk":
            data = (e.get("args") or {}).get("data") or {}
            if data.get("timeDeltas"):
                data["timeDeltas"] = data["timeDeltas"][:-1]
                break
    try:
        C.main_thread_profile(tr)
    except CellFailure as exc:
        assert exc.gate == "cpuprofile_chunk_ragged"
    else:
        raise AssertionError("samples and timeDeltas must be required to be 1:1")


def test_stacks_have_ancestry() -> None:
    prof = C.main_thread_profile(_trace())
    stacks = C.stacks_under(prof, "hotLeafFrame")
    assert stacks, "the known hot function must appear in sampled stacks"
    names = [f.function_name for f in stacks[0][1]]
    assert names[0] == "hotLeafFrame"
    assert "middleFrame" in names, "the caller must be recoverable from the stack"


def test_clock_anchor_is_the_event_timestamp() -> None:
    # `args.data.startTime` is in a different clock domain and is kept only to report the skew, never
    # used to anchor samples.
    prof = C.main_thread_profile(_trace())
    assert prof.declared_start_time
    assert abs(prof.clock_skew_us) < 10_000


def test_underpowered_windows_are_declared_not_hidden() -> None:
    prof = C.main_thread_profile(_trace())
    rows, diag = C.self_time_in_windows(prof, [(prof.samples[0].ts, prof.samples[0].ts + 50)])
    assert diag["underpowered"] is True
    assert diag["js_sample_count"] < C.MIN_JS_SAMPLES_FOR_RANKING
    assert isinstance(rows, list)


# ------------------------------------------------------------------ classify


def test_every_task_gets_an_origin() -> None:
    cls = K.classify_thread(_trace())
    assert cls.total_us > 0
    cls.assert_named()  # raises if unclassified time exceeds the limit
    assert cls.unclassified_pct == 0.0


def test_message_channel_count_matches_the_page() -> None:
    cls = K.classify_thread(_trace())
    assert cls.by_origin_count[K.MESSAGE_CHANNEL] == EXPECTED_MESSAGE_CHANNEL_TASKS


def test_origin_is_read_from_blinks_own_task_type() -> None:
    cls = K.classify_thread(_trace())
    evidence = {c.evidence for c in cls.tasks_of(K.MESSAGE_CHANNEL)}
    assert any(e.startswith("task_type:") for e in evidence), (
        "the scheduler category must be the authority for message-channel tasks, "
        "not the mojo src_file heuristic"
    )
    for c in cls.tasks_of(K.MESSAGE_CHANNEL):
        assert c.task_type, "task type should be recorded on the row"


def test_timers_and_frames_and_input_are_all_present() -> None:
    cls = K.classify_thread(_trace())
    assert cls.by_origin_count.get(K.TIMER, 0) > 100
    assert cls.by_origin_count.get(K.RAF, 0) >= EXPECTED_RAF_ITERATIONS
    assert cls.by_origin_count.get(K.INPUT, 0) > 0


def test_harness_cost_is_named_not_hidden() -> None:
    cls = K.classify_thread(_trace())
    # The devtools pipe running Runtime.evaluate is our own cost. It must have its own column so it
    # can be watched, and must not be inside the app's.
    assert K.AGENT_IPC in K.ORIGINS
    assert K.AGENT_IPC in K.HARNESS_ORIGINS


def test_unclassified_threshold_actually_fails() -> None:
    cls = K.classify_thread(_trace())
    for c in cls.tasks:
        c.origin = K.UNCLASSIFIED
    cls.by_origin_us = {K.UNCLASSIFIED: cls.total_us}
    try:
        cls.assert_named()
    except CellFailure as exc:
        assert exc.gate == "unclassified_task_pct"
    else:
        raise AssertionError("an all-unclassified run must fail the cell")


def test_task_duration_cross_check_fails_on_disagreement() -> None:
    cls = K.classify_thread(_trace())
    good = cls.total_us / 1e6
    K.cross_check_task_duration(cls, good * 1.02)  # inside 5%
    try:
        K.cross_check_task_duration(cls, good * 1.5)
    except CellFailure as exc:
        assert exc.gate == "task_duration_mismatch"
    else:
        raise AssertionError("a 50% disagreement must fail the cell")


# ---------------------------------------------------------------------- fit


def _pts(session: str, pairs) -> list[F.Point]:
    return [F.Point(length = x, value = y, session = session, rung = str(x)) for x, y in pairs]


def test_loglog_recovers_a_known_exponent() -> None:
    pairs = [(1000, 2.0), (10_000, 20.0), (100_000, 200.0), (1_000_000, 2000.0)]
    f = F.fit_loglog(_pts("s1", pairs))
    assert abs(f.b - 1.0) < 1e-6
    assert f.r2 > 0.999


def test_quadratic_growth_reads_as_two() -> None:
    pairs = [(10, 1.0), (100, 100.0), (1000, 10_000.0)]
    f = F.fit_loglog(_pts("s1", pairs), bootstrap = 0)
    assert abs(f.b - 2.0) < 1e-6


def test_cross_session_fit_is_refused() -> None:
    pts = _pts("s1", [(1, 1.0), (10, 10.0)]) + _pts("s2", [(100, 100.0)])
    try:
        F.fit_loglog(pts)
    except CellFailure as exc:
        assert exc.gate == "cross_session_fit"
    else:
        raise AssertionError("mixing sessions in one fit must be refused")


def test_zero_values_are_dropped_not_floored() -> None:
    pts = _pts("s1", [(10, 0.0), (100, 100.0), (1000, 10_000.0), (10_000, 1_000_000.0)])
    f = F.fit_loglog(pts, bootstrap = 0)
    assert f.n == 3, "the zero point must be dropped, not replaced by an epsilon"


def test_severity_is_zero_when_a_frame_grows_slower_than_the_total() -> None:
    assert F.severity(1000.0, 0.5, 1.2) == 0.0
    assert F.severity(100.0, 2.0, 1.0) == 100.0


def test_ranking_reports_what_it_could_not_fit() -> None:
    labels = {("a", "1", 0, 0): "steep", ("b", "2", 0, 0): "flat", ("c", "3", 0, 0): "sparse"}
    series = {
        ("a", "1", 0, 0): _pts("s", [(10, 1.0), (100, 100.0), (1000, 10_000.0)]),
        ("b", "2", 0, 0): _pts("s", [(10, 5.0), (100, 5.0), (1000, 5.0)]),
        ("c", "3", 0, 0): _pts("s", [(10, 1.0)]),
    }
    total = _pts("s", [(10, 10.0), (100, 100.0), (1000, 1000.0)])
    rows, diag = F.rank_frames(series, labels, total, bootstrap = 0)
    assert rows[0].frame_label == "steep"
    assert "sparse" in diag["frames_skipped"]
    assert rows[-1].severity == 0.0


# ------------------------------------------------------------------ oracles


def test_exact_match_is_a_naming() -> None:
    q = O.blocks_times_renders(685, 6, source = "DOM census")
    v = O.check("cloneChildFibers", 4110, [q])
    assert v.is_naming
    assert "4110" in v.detail and "685" in v.detail


def test_off_by_a_hair_is_not_a_naming() -> None:
    q = O.blocks_times_renders(685, 6, source = "DOM census")
    v = O.check("cloneChildFibers", 4111, [q])
    assert not v.is_naming
    assert v.verdict == O.NEAR_MISS
    assert "+1" in (v.ratio or "")


def test_double_invoke_is_reported_as_a_diagnosis() -> None:
    q = O.blocks_times_renders(100, 2, source = "DOM census")
    v = O.check("f", 400, [q])
    assert v.verdict == O.NEAR_MISS
    assert "StrictMode" in (v.ratio or "")


def test_a_frame_matching_nothing_is_unexplained_not_silent() -> None:
    q = O.blocks_times_renders(685, 6, source = "DOM census")
    v = O.check("Zk", 91_237, [q])
    assert v.verdict == O.UNEXPLAINED
    assert v.exact_call_count == 91_237


def test_no_count_means_not_measured_rather_than_a_guess() -> None:
    v = O.check("Zk", None, [O.blocks_times_renders(1, 1, source = "x")])
    assert v.verdict == O.NOT_MEASURED


def test_check_all_reports_every_bucket() -> None:
    q = O.blocks_times_renders(685, 6, source = "DOM census")
    out = O.check_all([("named", 4110), ("odd", 999_983), ("nocount", None)], [q])
    assert out["named_at_least_one_frame"]
    assert len(out["unexplained_hot_frames"]) == 1
    assert len(out["not_measured"]) == 1


# ------------------------------------------------------------------ symbols


class _Fn:
    def __init__(self, url, name, start, end, count):
        self.url, self.function_name = url, name
        self.start_offset, self.end_offset, self.count = start, end, count
        self.script_id = "1"


class _Snap:
    def __init__(self, fns):
        self.functions = fns


def _arms(dev_counts, prod_counts, anchor_dev, anchor_prod):
    """Two arms shaped like a REAL dev/prod pair.

    The two sides deliberately differ in script URL and byte offset, because
    that is what two different builds look like: a Vite dev server serves
    `/node_modules/.vite/deps/react-dom_client.js` while a production build
    inlines react-dom into a hashed app chunk at entirely different offsets. An
    earlier version of this helper gave both sides identical URLs and offsets,
    which is indistinguishable from pointing both arms at the same server, and
    the same-build guard now correctly refuses it.
    """
    dev = [
        _Snap(
            [
                _Fn("/deps/react-dom_client.js", n, i * 37 + 3, i * 37 + 21, c[r])
                for i, (n, c) in enumerate(dev_counts.items())
            ]
            + [
                _Fn("/src/app.jsx", n, 900 + i * 11, 930 + i * 11, c[r])
                for i, (n, c) in enumerate(anchor_dev.items())
            ]
        )
        for r in range(2)
    ]
    prod = [
        _Snap(
            [
                _Fn("/assets/index-abc123.js", n, i * 10, i * 10 + 5, c[r])
                for i, (n, c) in enumerate(prod_counts.items())
            ]
            + [
                _Fn("/assets/index-abc123.js", n, 500 + i, 505 + i, c[r])
                for i, (n, c) in enumerate(anchor_prod.items())
            ]
        )
        for r in range(2)
    ]
    return dev, prod


def test_bridge_resolves_a_minified_name_by_count_vector() -> None:
    dev, prod = _arms(
        {"cloneChildFibers": [340, 3400], "beginWork": [17, 170]},
        {"Zk": [340, 3400], "qi": [17, 170]},
        {"ThreadMessage": [50, 500]},
        {"ThreadMessage": [50, 500]},
    )
    b = S.build_bridge(
        dev,
        prod,
        rungs = ("s", "m"),
        react_version = "19.2.4",
        bundle_source = "x",
        anchor_names = ["ThreadMessage"],
        react_url_filter = None,
        anchor_url_filter = None,
    )
    assert b.status == S.OK
    assert b.resolve("/assets/index-abc123.js", 0, 5) == "cloneChildFibers"
    assert b.resolve("/assets/index-abc123.js", 10, 15) == "beginWork"
    # The anchor legitimately maps to itself, so some identity is expected; what must not happen is
    # identity DOMINATING, which would mean no minification was undone.
    assert b.identity_mappings / len(b.mapping) <= S.MAX_IDENTITY_MAPPING_FRACTION


def test_bridge_refuses_to_guess_an_ambiguous_vector() -> None:
    dev, prod = _arms(
        {"alpha": [7, 7], "beta": [7, 7]},
        {"Aa": [7, 7], "Bb": [7, 7]},
        {"ThreadMessage": [50, 500]},
        {"ThreadMessage": [50, 500]},
    )
    b = S.build_bridge(
        dev,
        prod,
        rungs = ("s", "m"),
        react_version = "19.2.4",
        bundle_source = "x",
        anchor_names = ["ThreadMessage"],
        react_url_filter = None,
        anchor_url_filter = None,
    )
    assert b.mapping == {}
    assert b.ambiguous_prod and b.ambiguous_dev


def test_anchor_mismatch_discards_the_whole_bridge() -> None:
    dev, prod = _arms(
        {"cloneChildFibers": [340, 3400]},
        {"Zk": [340, 3400]},
        {"ThreadMessage": [50, 500]},
        {"ThreadMessage": [51, 500]},  # counts are NOT invariant here
    )
    b = S.build_bridge(
        dev,
        prod,
        rungs = ("s", "m"),
        react_version = "19.2.4",
        bundle_source = "x",
        anchor_names = ["ThreadMessage"],
        react_url_filter = None,
        anchor_url_filter = None,
    )
    assert b.status == S.FAILED
    assert b.mapping == {}, "one bad anchor must discard every mapping, not just its own"
    assert b.resolve("/react-dom.js", 0, 5) is None


def test_same_build_on_both_arms_is_refused() -> None:
    # The one failure anchors are structurally blind to: if both arms are the same build every anchor
    # maps to itself PERFECTLY, so anchor validation passes and the bridge reports ok while mapping
    # minified names to themselves. Verified against the real implementation before this guard
    # existed: it returned status ok with {"Zk": "Zk"}.
    _, prod = _arms(
        {"a": [1, 1]},
        {"Zk": [340, 3400], "qi": [17, 170]},
        {"x": [1, 1]},
        {"ThreadMessage": [50, 500]},
    )
    b = S.build_bridge(
        prod,
        prod,
        rungs = ("s", "m"),
        react_version = "19.2.4",
        bundle_source = "x",
        anchor_names = ["ThreadMessage"],
        react_url_filter = None,
        anchor_url_filter = None,
    )
    assert b.status == S.FAILED
    assert b.mapping == {}
    assert "same bundle" in b.failure_reason
    assert not b.anchor_failures, "anchors PASS here; that is exactly why this guard is needed"


def test_an_all_identity_mapping_is_refused() -> None:
    # Different scripts and offsets, but no minification was undone: every resolved name is its own
    # name, so the bridge is doing nothing.
    dev, prod = _arms(
        {"Zk": [340, 3400], "qi": [17, 170]},
        {"Zk": [340, 3400], "qi": [17, 170]},
        {"ThreadMessage": [50, 500]},
        {"ThreadMessage": [50, 500]},
    )
    b = S.build_bridge(
        dev,
        prod,
        rungs = ("s", "m"),
        react_version = "19.2.4",
        bundle_source = "x",
        anchor_names = ["ThreadMessage"],
        react_url_filter = None,
        anchor_url_filter = None,
    )
    assert b.status == S.FAILED
    assert "map to their own name" in b.failure_reason


def test_single_rung_bridge_is_refused() -> None:
    b = S.build_bridge(
        [_Snap([])],
        [_Snap([])],
        rungs = ("s",),
        react_version = "19.2.4",
        bundle_source = "x",
        anchor_names = ["A"],
    )
    assert b.status == S.FAILED
    assert "single-rung" in b.failure_reason or "rung" in b.failure_reason


def test_no_dev_millisecond_can_enter_the_artefact() -> None:
    b = S.Bridge(status = S.OK, react_version = "19.2.4", bundle_sha = "abc")
    b.to_json()  # integers only: fine
    try:
        S.assert_no_measurements({"mapping": {"a": "b"}, "dev_render_ms": 12.5})
    except CellFailure as exc:
        assert exc.gate == "dev_measurement_leak"
    else:
        raise AssertionError("a float in a bridge artefact must be refused")


def test_bridge_round_trips_through_disk(tmpdir: str = "") -> None:
    import tempfile

    b = S.Bridge(status = S.OK, react_version = "19.2.4", bundle_sha = "deadbeefcafe0000")
    b.mapping["react-dom.js:0:5"] = "cloneChildFibers"
    b.evidence["react-dom.js:0:5"] = [340, 3400]
    with tempfile.TemporaryDirectory() as d:
        path = b.save(d)
        assert os.path.basename(path) == "react-dom@19.2.4-deadbeefcafe.json"
        again = S.Bridge.load(path)
    assert again.resolve("/whatever/react-dom.js", 0, 5) == "cloneChildFibers"


# ------------------------------------------------ the no-bare-zero convention


def test_zero_is_distinguishable_from_did_not_run() -> None:
    ran = measured("task_ms", 0.0)
    didnt = unmeasured("task_ms", "tracing never started")
    assert ran["task_ms"] == 0.0 and ran["task_ms_attempted"] is True
    assert didnt["task_ms"] is None and didnt["task_ms_attempted"] is False
    assert didnt["task_ms_reason"]
    assert_no_bare_zero(ran)
    assert_no_bare_zero(didnt)


def test_a_bare_zero_is_refused() -> None:
    try:
        assert_no_bare_zero({"frames_dropped": 0})
    except CellFailure as exc:
        assert exc.gate == "bare_zero"
    else:
        raise AssertionError("a bare zero must be refused")


def test_unmeasured_demands_a_reason() -> None:
    try:
        unmeasured("x", "")
    except ValueError:
        pass
    else:
        raise AssertionError("unmeasured without a reason must raise")


def test_merge_refuses_conflicting_keys() -> None:
    try:
        merge(measured("a", 1), measured("a", 2))
    except ValueError:
        pass
    else:
        raise AssertionError("a silent key collision must raise")


# ------------------------------------------------------------- M2/M3 oracles


def test_cumulative_reparse_reads_as_quadratic() -> None:
    out = O.reparse_regime(4_020_000, 40_000, 200)
    assert out["regime"] == O.REGIME_QUADRATIC
    assert out["evidence_class"] == "regime_test"


def test_incremental_parse_reads_as_linear() -> None:
    out = O.reparse_regime(40_000, 40_000, 200)
    assert out["regime"] == O.REGIME_LINEAR


def test_short_reply_refuses_to_call_the_regime() -> None:
    out = O.reparse_regime(300, 100, 2)
    assert out["regime"] == O.REGIME_UNDECIDED
    assert "refusal band" in out["reason"]


def test_m3_forced_layout_is_an_exact_oracle() -> None:
    v = O.forced_layout_per_callback(4110, 4110, source = "page counters")
    assert v.is_naming
    v2 = O.forced_layout_per_callback(4110, 2055, source = "page counters")
    assert not v2.is_naming


def test_missing_page_counters_are_skipped_loudly() -> None:
    out = O.evaluate_page_counters({})
    assert out["m2"]["skipped"] and out["m2"]["reason"]
    assert out["m3"]["skipped"] and out["m3"]["reason"]


def test_page_counter_contract_is_documented() -> None:
    # Layer 3 emits against these exact key names; a rename here is a contract change and must be
    # agreed, not discovered at analysis time.
    assert set(O.PAGE_COUNTER_CONTRACT) == {"m2_reparse", "m3_forced_layout"}
    assert "chars_rescanned" in O.PAGE_COUNTER_CONTRACT["m2_reparse"]
    assert "forced_layouts" in O.PAGE_COUNTER_CONTRACT["m3_forced_layout"]


def _run_all() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
