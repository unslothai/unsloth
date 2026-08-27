# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive Layer 2's instruments through Layer 1's ACTUAL protocol, in a real browser.

The unit tests next door prove the analysis is right about a trace. This proves
the instruments are wired correctly into the harness contract, which is a
different failure mode entirely and the one my own report flagged as untested:
four modules that imported cleanly and registered nothing.

It uses the real `Cell`, `Window`, `BenchContext` and `Paths` from
`runtime.types`, the real `instruments.build(level)` registry, and calls
`attach / start_cell / open / close / end_cell / detach` in the documented order
with the documented reverse-order close. Nothing is mocked except the page,
which is a local synthetic instead of an Unsloth install.

Requires Playwright with Chromium. Skips cleanly without it, because a machine
that cannot run a browser should report that rather than fail.

    python tests/studio/studiobench/analysis/test_instruments_live.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDIO_TESTS = os.path.dirname(os.path.dirname(_HERE))
if _STUDIO_TESTS not in sys.path:
    sys.path.insert(0, _STUDIO_TESTS)

from studiobench.analysis import assert_no_bare_zero  # noqa: E402
from studiobench.instruments import available, build, import_errors  # noqa: E402
from studiobench.runtime.types import (  # noqa: E402
    BenchContext,
    Cell,
    Paths,
    Recorder,
    Window,
    make_cell_id,
    new_session_id,
)

# A page with a known shape: a message-channel loop (the React scheduler's
# mechanism), a timer loop, and a function whose call count is exactly known.
PAGE = """<!doctype html><meta charset=utf-8><body><div id=o></div><script>
window.__N = 0;
window.__junk = [];
function hotLeafFrame(x){ var s=0; for(var i=0;i<400;i++){ s+=Math.sqrt(i*x)|0; } window.__N++; return s; }
function allocSiblings(n){ var out=[]; for(var i=0;i<n;i++) out.push({a:i,b:'sib'+i,c:[i,i,i]}); return out; }
function middleFrame(k){ var t=0; for(var j=0;j<20;j++) t+=hotLeafFrame(j+k); window.__junk=allocSiblings(200); return t; }
window.__runMsg = function(iters){ return new Promise(function(res){
  var mc = new MessageChannel(); var n = 0;
  mc.port1.onmessage = function(){ middleFrame(2); if(++n < iters) mc.port2.postMessage(0); else res(n); };
  mc.port2.postMessage(0); }); };
window.__runTimer = function(){ var id = setInterval(function(){ middleFrame(1); }, 4);
  setTimeout(function(){ clearInterval(id); }, 300); };
</script></body>"""

MESSAGE_ITERATIONS = 150
CALLS_PER_MESSAGE = 20  # middleFrame calls hotLeafFrame 20 times


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def _drive(instruments, ctx, cell, page, window_names):
    """Run the documented lifecycle over a list of windows.

    Open is in `name` order, close is in REVERSE `name` order, exactly as
    INTERFACES.md section 2 specifies. A raising instrument is caught and
    disabled for the rest of the cell rather than losing the window.
    """
    ordered = sorted(instruments, key = lambda i: i.name)
    for inst in ordered:
        inst.attach(ctx)
    for inst in ordered:
        inst.start_cell(cell)

    rows = {}
    for wname in window_names:
        w = Window(name = wname, kind = "action", cell = cell, t_open_ms = time.monotonic() * 1000)
        for inst in ordered:
            inst.open(w)
        page.evaluate(f"__runTimer(); __runMsg({MESSAGE_ITERATIONS});")
        page.wait_for_timeout(400)
        w.t_close_ms = time.monotonic() * 1000
        for inst in reversed(ordered):
            out = inst.close(w)
            if out is not None:
                w.instruments[inst.name] = out
        rows[wname] = w

    cell_rows = {}
    for inst in ordered:
        out = inst.end_cell(cell)
        if out is not None:
            cell_rows[inst.name] = out
    for inst in ordered:
        inst.detach()
    return rows, cell_rows


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return _skip("playwright is not installed")

    errs = import_errors()
    if errs:
        print(f"FAIL: instrument modules failed to import: {errs}")
        return 1

    names = dict(available())
    expected = {"tracing": 1, "cpu_profile": 1, "coverage": 3, "heap": 3}
    if names != expected:
        print(f"FAIL: registry is {names}, expected {expected}")
        return 1
    print(f"ok   registry: {sorted(names.items())}")

    if build(0):
        print("FAIL: build(0) must be empty so headline numbers can come from L0 only")
        return 1
    print("ok   build(0) is empty; L0 attaches nothing")

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        paths = Paths.under(Path(tmp))
        session_id = new_session_id()
        recorder = Recorder(paths.payload_jsonl, session_id)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = ["--no-sandbox"])
            try:
                for level in (1, 2, 3):
                    context = browser.new_context()
                    page = context.new_page()
                    page.set_content(PAGE)
                    cdp = context.new_cdp_session(page)

                    ctx = BenchContext(
                        browser = browser,
                        context = context,
                        page = page,
                        cdp = cdp,
                        base_url = "about:blank",
                        session_id = session_id,
                        tier = "quick",
                        instrument_level = level,
                        paths = paths,
                        recorder = recorder,
                        log = lambda m: None,
                    )
                    cell = Cell(
                        cell_id = make_cell_id("10K", "A0", 0),
                        rung = "10K",
                        rung_tokens = 10_000,
                        instrument_level = level,
                        session_id = session_id,
                    )
                    insts = build(level)
                    page.evaluate("window.__N = 0")
                    wrows, crows = _drive(insts, ctx, cell, page, ["action:stream"])
                    ground_truth = page.evaluate("window.__N")

                    failures += _check_level(level, wrows, crows, ground_truth)
                    context.close()
            finally:
                browser.close()
        recorder.close()

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


def _check_level(level, wrows, crows, ground_truth) -> int:
    bad = 0
    w = wrows["action:stream"]
    tag = f"L{level}"

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print(f"FAIL [{tag}] {msg}")

    # Every payload must obey the no-bare-zero rule before it can be emitted.
    for name, payload in list(w.instruments.items()) + list(crows.items()):
        try:
            assert_no_bare_zero(payload, f"{tag}.{name}")
        except Exception as exc:  # noqa: BLE001
            fail(f"{name} violates the no-bare-zero rule: {exc}")

    # The window row must be JSON-serialisable, since Recorder writes it.
    import json

    try:
        json.dumps(w.row())
    except Exception as exc:  # noqa: BLE001
        fail(f"window row is not JSON-safe: {exc}")

    # Overhead is mandatory from every instrument at level >= 1, and it is what
    # the report layer's overhead_growth_with_length gate consumes.
    for name, payload in crows.items():
        if "overhead_ms" not in payload or "overhead_ms_attempted" not in payload:
            fail(f"{name}.end_cell has no overhead_ms/overhead_ms_attempted")

    tracing = w.instruments.get("tracing", {})
    if tracing.get("task_ms") is None:
        fail(f"tracing reported no task_ms: {tracing.get('task_ms_reason')}")
    elif tracing.get("unclassified_task_pct") is None:
        fail("tracing reported no unclassified_task_pct")
    else:
        by_origin = tracing.get("task_count_by_origin") or {}
        print(
            f"ok   [{tag}] tracing task_ms={tracing['task_ms']} "
            f"unclassified={tracing['unclassified_task_pct']}% origins={by_origin}"
        )
        if by_origin.get("message_channel", 0) < MESSAGE_ITERATIONS * 0.8:
            fail(f"expected ~{MESSAGE_ITERATIONS} message-channel tasks, got {by_origin}")

    # THE LADDER. Naming requires the v8 profiler, which only L2+ turns on. At
    # L1 that must be an explicit null with a reason, never an empty list.
    if level == 1:
        if tracing.get("named_frames") is not None:
            fail("L1 must not report named frames; the profiler category is off")
        elif not tracing.get("named_frames_reason"):
            fail("L1 named_frames must carry a reason")
        else:
            print(f"ok   [{tag}] named_frames is null with a reason, not an empty list")
        cp = w.instruments.get("cpu_profile", {})
        if not cp.get("active"):
            fail(f"cpu_profile should be active at L1: {cp.get('self_ms_top_reason')}")
        elif cp.get("self_ms_top") is None:
            fail(f"cpu_profile ran but named nothing: {cp.get('self_ms_top_reason')}")
        else:
            top = [r["frame"].split(" @ ")[0] for r in cp["self_ms_top"][:3]]
            print(f"ok   [{tag}] standalone profiler named {top}")
    else:
        frames = tracing.get("named_frames")
        if frames is None:
            fail(f"L{level} must name frames: {tracing.get('named_frames_reason')}")
        else:
            top = [r["frame"].split(" @ ")[0] for r in frames[:3]]
            print(f"ok   [{tag}] tracing named {top}")
        cp = w.instruments.get("cpu_profile", {})
        if cp.get("active"):
            fail("cpu_profile must stand down at L2+; V8 has one CpuProfiler")
        elif not (crows.get("cpu_profile", {}).get("stand_down_reason")):
            fail("cpu_profile stood down without saying why")
        else:
            print(
                f"ok   [{tag}] cpu_profile stood down: "
                f"{crows['cpu_profile']['stand_down_reason'][:60]}..."
            )

    if level == 3:
        cov = w.instruments.get("coverage", {})
        if cov.get("total_calls") is None:
            fail(f"coverage reported nothing: {cov.get('total_calls_reason')}")
        else:
            hits = [f for f in cov["top_functions"] if f["function"] == "hotLeafFrame"]
            if not hits:
                fail("coverage did not count the known hot function")
            elif hits[0]["count"] != ground_truth:
                fail(f"coverage counted {hits[0]['count']}, page counted {ground_truth}")
            else:
                mids = [f for f in cov["top_functions"] if f["function"] == "middleFrame"]
                exact_oracle = mids and hits[0]["count"] == CALLS_PER_MESSAGE * mids[0]["count"]
                print(
                    f"ok   [{tag}] coverage counted hotLeafFrame exactly "
                    f"{hits[0]['count']} == page counter; structural oracle "
                    f"(hot == {CALLS_PER_MESSAGE} x middle) {'holds' if exact_oracle else 'MISSED'}"
                )
                if not exact_oracle:
                    fail("the structural count oracle did not hold")
            if not cov.get("timings_void"):
                fail("coverage must mark the cell timings_void")
            else:
                print(f"ok   [{tag}] coverage marked timings_void")
        heap = w.instruments.get("heap", {})
        if heap.get("allocated_bytes") is None:
            fail(f"heap reported nothing: {heap.get('allocated_bytes_reason')}")
        elif not heap.get("included_objects_collected_by_major_gc"):
            fail("heap must include objects collected by major GC")
        else:
            sites = [s["site"].split(" @ ")[0] for s in heap["top_sites"][:3]]
            print(f"ok   [{tag}] heap saw {heap['allocated_bytes']} bytes, top sites {sites}")
    else:
        if "coverage" in w.instruments or "heap" in w.instruments:
            fail(f"coverage/heap must not run below L3, got {sorted(w.instruments)}")

    return bad


if __name__ == "__main__":
    raise SystemExit(main())
