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


def test_a_half_open_window_does_not_leak_its_open_reading_into_the_next_cell():
    """`_chars_open` is per window; a cell that died between open and close must not seed the next
    cell's first window with a stale character count."""
    inst = StreamCostInstrument()
    inst.start_cell(_FakeCell())
    inst._chars_open = 12345
    inst.start_cell(_FakeCell())
    assert inst._chars_open is None
