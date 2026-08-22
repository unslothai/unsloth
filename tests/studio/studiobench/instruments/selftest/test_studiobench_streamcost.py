# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The streaming-cost accumulator, on both sides of the page boundary.

WHY THE REAL JAVASCRIPT AND NOT A PYTHON PORT, same reason `test_studiobench_parity_digest.py`
gives: the file that ships is `instruments/streamcost.js`, and a re-implementation tested here
would pass forever while the shipped file drifted. So node runs the actual file against a shim of
the four globals it touches, and if node is missing the test SKIPS rather than passing on a
substitute.

The one thing that cannot be shimmed is the thing being tested: a real blocked main thread. The
stall below is a synchronous busy wait, so the 1 ms timer inside the instrument really is unable
to run for its duration, exactly as it would be during a long task in the app.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.instruments.streamcost import StreamCostInstrument  # noqa: E402

STREAMCOST_JS = Path(__file__).resolve().parents[1] / "streamcost.js"

#: `IDLE_GAP_MS` in the instrument. A stall longer than this is the case that used to vanish.
IDLE_GAP_MS = 1500

HARNESS_JS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const stallMs = Number(process.argv[3]);

const window = {};
const document = { querySelectorAll: () => [] };
(new Function("window", "document", src))(window, document);

// frames.js is what owns the clamp; only its shape matters here.
window.__sb.frames = { clamp: () => ({ clampMs: 1.0 }) };

const sc = window.__sb.streamcost;
sc.__markStreaming();

// A REAL block: synchronous, so the instrument's 1 ms timer cannot run for its duration and
// observes the whole stall as one gap once the thread is free again.
const started = performance.now();
while (performance.now() - started < stallMs) { /* spin */ }

// Let the timer catch up, then drain.
setTimeout(() => {
  console.log(JSON.stringify(sc.read(stallMs + 100)));
  // The instrument's 1 ms timer re-arms itself forever, exactly as it does in the page. Nothing
  // stops it, so the harness ends the process rather than waiting for an empty event loop.
  process.exit(0);
}, 60);
"""


def _node() -> str:
    exe = shutil.which("node") or shutil.which("nodejs")
    if exe is None:
        pytest.skip(
            "node is not installed, so the shipped streamcost.js could not be evaluated; "
            "this is NOT MEASURED rather than passing"
        )
    return exe


def drain_after_stall(stall_ms: float) -> dict:
    exe = _node()
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.js"
        harness.write_text(HARNESS_JS, encoding = "utf-8")
        got = subprocess.run(
            [exe, str(harness), str(STREAMCOST_JS), str(stall_ms)],
            capture_output = True,
            text = True,
            timeout = 120,
        )
    if got.returncode != 0:
        raise AssertionError(f"the streamcost.js harness failed: {got.stderr.strip()[-800:]}")
    return json.loads(got.stdout)


def test_a_stall_longer_than_the_idle_gap_is_still_charged_to_the_stream():
    """The worst stall in a window is the one the timer sees last, and it must not be dropped.

    A stall that outlasts `IDLE_GAP_MS` is only observed after it has ended, and by then the last
    SSE chunk is older than the idle threshold. Deciding the interval's attribution from the state
    at its END therefore threw away the whole stall, so a streaming regression read CHEAPER once
    it crossed 1.5 s -- the metric moving the wrong way as the defect got worse.
    """
    stall_ms = IDLE_GAP_MS + 400
    out = drain_after_stall(stall_ms)
    assert out["streaming_observed"] is True
    # The stall began while the stream was in flight, so it belongs to the stream.
    assert out["streaming_ms"] >= stall_ms * 0.9, out
    assert out["stream_blocked_ms"] >= stall_ms * 0.9, out


def test_a_stall_shorter_than_the_idle_gap_is_charged_too():
    """The case that always worked, kept so the fix above cannot be undone by loosening it."""
    stall_ms = 300
    out = drain_after_stall(stall_ms)
    assert out["streaming_ms"] >= stall_ms * 0.9, out
    assert out["stream_blocked_ms"] >= stall_ms * 0.9, out


class _FakeCell:
    cell_id = "100K.base.rep0"


def test_overhead_is_reported_per_cell_and_not_accumulated_across_them():
    """One instrument instance serves the whole session, and the rungs run in ascending order.

    An overhead accumulator that is never cleared reports cell k as the sum of cells 1..k, which
    climbs with the rung ladder however flat the instrument actually is. That is the exact shape
    `overhead_growth_with_length` exists to catch, manufactured by the instrument declaring it.
    """
    inst = StreamCostInstrument()

    inst.start_cell(_FakeCell())
    inst._overhead_ms += 40.0
    first = inst.end_cell(_FakeCell())
    assert first["overhead_ms"] == 40.0

    inst.start_cell(_FakeCell())
    inst._overhead_ms += 5.0
    second = inst.end_cell(_FakeCell())
    assert second["overhead_ms"] == 5.0, "the second cell must not carry the first cell's overhead"


class _FakeWindow:
    duration_ms = 10_000.0


class _FakeStreamCostPage:
    """The page-side accumulator, on exactly the contract `streamcost.js` implements.

    `read()` snapshots `overheadMs` into its result and THEN resets it; `replyChars()` adds the
    cost of its own `querySelectorAll` to whatever the accumulator currently holds; `reset()`
    zeroes it. Those three facts are the whole of the defect below, and re-stating them here rather
    than driving node keeps the test about the DRIVER's ordering, which is where the defect lives.
    """

    #: What one boundary scan costs. `querySelectorAll` collects its matches up front over the
    #: whole document, so this is the one part of the instrument that grows with the rung.
    SCAN_MS = 3.9

    def __init__(self) -> None:
        self.overhead_ms = 0.0
        self.scans = 0

    def evaluate(
        self,
        expr,
        arg = None,
    ):
        if "reset()" in expr:
            self.overhead_ms = 0.0
            return None
        if "replyChars" in expr:
            self.scans += 1
            self.overhead_ms += self.SCAN_MS
            return 1_000 * self.scans
        if "read(" in expr:
            snapshot = round(self.overhead_ms, 2)
            self.overhead_ms = 0.0
            return {"streaming_observed": True, "overhead_ms": snapshot}
        return None


def test_the_close_side_reply_scan_is_counted_in_the_declared_overhead():
    """REGRESSION. Half of every window's boundary scans were missing from `overhead_ms`.

    `close()` calls `read(ms)` first, which snapshots the page's overhead total and then resets it,
    and only afterwards calls `replyChars(force)` -- the FORCED, whole-document scan that is the
    one part of this instrument whose cost tracks the rung. That scan accumulated into a fresh
    page-side total which the next `open()` began by resetting, so it was never read by anyone.

    The number this corrupts is the only evidence for the level 0 claim: `end_cell` declares it
    precisely so the claim is checkable from the payload rather than from a docstring, and it was
    reporting about half of the rung-dependent cost it exists to expose.
    """
    inst = StreamCostInstrument()
    page = _FakeStreamCostPage()
    # After `start_cell`, which re-reads the page from the context every cell.
    inst.start_cell(_FakeCell())
    inst.page = page
    for _ in range(3):
        inst.open(_FakeWindow())
        inst.close(_FakeWindow())

    assert page.scans == 6, "three windows means three open scans and three close scans"
    declared = inst.end_cell(_FakeCell())["overhead_ms"]
    assert declared == pytest.approx(6 * _FakeStreamCostPage.SCAN_MS, abs = 0.05), (
        "the close-side scans are missing from the declared overhead",
        declared,
    )


def test_the_close_side_drain_does_not_disturb_the_window_s_own_reading():
    """The scan is harvested AFTER the window's numbers are taken, so it cannot move them."""
    inst = StreamCostInstrument()
    inst.start_cell(_FakeCell())
    inst.page = _FakeStreamCostPage()
    inst.open(_FakeWindow())
    out = inst.close(_FakeWindow())

    assert out["streaming_observed"] is True
    assert out["reply_chars_delta"] == 1_000
    # The window's own overhead figure is what `read()` returned; the close scan is reported
    # beside it rather than folded into it.
    assert out["overhead_ms"] == pytest.approx(_FakeStreamCostPage.SCAN_MS, abs = 0.05)
    assert out["close_scan_overhead_ms"] == pytest.approx(_FakeStreamCostPage.SCAN_MS, abs = 0.05)


def test_a_half_open_window_does_not_leak_its_open_reading_into_the_next_cell():
    """`_chars_open` is per window; a cell that died between open and close must not seed the next
    cell's first window with a stale character count."""
    inst = StreamCostInstrument()
    inst.start_cell(_FakeCell())
    inst._chars_open = 12345
    inst.start_cell(_FakeCell())
    assert inst._chars_open is None
