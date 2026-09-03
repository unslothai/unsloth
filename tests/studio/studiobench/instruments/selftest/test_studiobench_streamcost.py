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


#: The same file, driven through the ordering that loses a burst: a window that closes while the
#: chain the last chunk started has not reached its macrotask yet. `readWhilePending` decides which
#: of the two orderings the harness produces, so the pinned case and its control differ by nothing
#: else.
PENDING_HARNESS_JS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const burnMs = Number(process.argv[3]);
const readWhilePending = process.argv[4] === "pending";

const window = {};
const document = { querySelectorAll: () => [] };
(new Function("window", "document", src))(window, document);
window.__sb.frames = { clamp: () => ({ clampMs: 1.0 }) };
const sc = window.__sb.streamcost;

// ONE BURST AND ITS TASK CHAIN. `__markStreaming` is the decode; the spin after it is the parse,
// the delta accumulation and the render that the chain exists to measure.
const burst = () => {
  sc.__markStreaming();
  const started = performance.now();
  while (performance.now() - started < burnMs) { /* spin */ }
};

const finish = (first) => setTimeout(() => {
  // A SECOND WINDOW, opened and closed after the macrotask has certainly run. Whatever the burst
  // cost belongs to the first window; anything landing here is the leak.
  const second = sc.read(0);
  console.log(JSON.stringify({ first: first, second: second }));
  process.exit(0);
}, 60);

if (readWhilePending) {
  burst();
  // The window closes IN THE SAME TASK, so the MessageChannel message posted by the decode is
  // still queued.
  finish(sc.read(burnMs + 50));
} else {
  burst();
  // The ordinary ordering: the chain reaches its macrotask first and the window closes after it.
  setTimeout(() => finish(sc.read(burnMs + 50)), 30);
}
"""


#: The same file again, driven through ONE `decode()` of a whole batched read. `payload` decides
#: what that read carries: a burst of the pacer's own frames larger than the decoder's scan bound,
#: the same burst small enough to sit under it, or a blob of the size that bound exists to keep out.
#: The burn after the decode stands in for the SSE parse, the delta accumulation and the render --
#: the task chain `delta_task_ms` is supposed to be charging.
BATCH_HARNESS_JS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const mode = process.argv[3];
const burnMs = Number(process.argv[4]);

const window = {};
const document = { querySelectorAll: () => [] };
(new Function("window", "document", src))(window, document);
window.__sb.frames = { clamp: () => ({ clampMs: 1.0 }) };
const sc = window.__sb.streamcost;

// pacer.py's own framing: `data: ` + the chunk object + a blank line, carrying 64 characters at
// fast cadence. Built here rather than imported so the harness stays a single node process.
const frame = (i) =>
  "data: " +
  JSON.stringify({
    id: "chatcmpl-0123456789abcdef0123",
    object: "chat.completion.chunk",
    created: 1780000000,
    model: "studiobench-pacer",
    choices: [{ index: 0, delta: { content: String.fromCharCode(97 + (i % 26)).repeat(64) },
                finish_reason: null }],
  }) +
  "\n\n";

let payload = "";
if (mode === "blob") {
  // A bundle, a blob or a paste: over the bound and with no relay framing anywhere in it.
  payload = "z".repeat(100000);
} else {
  // A BATCH ABOVE THE BOUND. Everything the browser buffered while the main thread was stalled,
  // handed to the app as one read, exactly as chromium does after a stall past about two seconds.
  let i = 0;
  const want = mode === "batch-over" ? 65537 : 40000;
  while (payload.length < want) payload += frame(i++);
}

// THROUGH THE REAL HOOK. `TextDecoder.prototype.decode` is what the instrument wraps and what the
// app reaches with the bytes of one `reader.read()`, so the batch is decoded in a single call.
const decoded = new TextDecoder().decode(
  new Uint8Array(Buffer.from(payload, "utf8")), { stream: true }
);

// The chain that decode started, still on the same task.
const started = performance.now();
while (performance.now() - started < burnMs) { /* spin */ }

setTimeout(() => {
  console.log(JSON.stringify({
    payload_chars: payload.length,
    decoded_chars: decoded.length,
    read: sc.read(null),
  }));
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


def burst_across_a_window_close(burn_ms: float, *, read_while_pending: bool) -> dict:
    exe = _node()
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "pending.js"
        harness.write_text(PENDING_HARNESS_JS, encoding = "utf-8")
        got = subprocess.run(
            [
                exe,
                str(harness),
                str(STREAMCOST_JS),
                str(burn_ms),
                "pending" if read_while_pending else "settled",
            ],
            capture_output = True,
            text = True,
            timeout = 120,
        )
    if got.returncode != 0:
        raise AssertionError(f"the streamcost.js harness failed: {got.stderr.strip()[-800:]}")
    return json.loads(got.stdout)


def one_decoded_batch(mode: str, burn_ms: float) -> dict:
    exe = _node()
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "batch.js"
        harness.write_text(BATCH_HARNESS_JS, encoding = "utf-8")
        got = subprocess.run(
            [exe, str(harness), str(STREAMCOST_JS), mode, str(burn_ms)],
            capture_output = True,
            text = True,
            timeout = 120,
        )
    if got.returncode != 0:
        raise AssertionError(f"the streamcost.js harness failed: {got.stderr.strip()[-800:]}")
    return json.loads(got.stdout)


#: The task chain one burst starts, in the harness above. Large enough that losing it is
#: unmistakable in the assertions and small enough that node's timers stay honest.
BURST_CHAIN_MS = 40.0


def test_a_burst_still_in_flight_at_the_window_close_is_charged_to_that_window():
    """REGRESSION. `read()` snapshotted `deltaTaskMs` and then `reset()` zeroed it while the chain
    the last chunk started was still open, so the MessageChannel callback charged that burst to a
    fresh accumulator that nobody ever reads: `StreamCostInstrument.close` discards everything but
    `overhead_ms` from its tail `read(0)`, and `open()` resets before the next window in any case.
    The burst's characters stayed in the denominator and its targeted cost left the numerator.

    `delta_task_ms` is the TARGETED numerator -- the one quantity that separates stream cost from
    the action windows around it, and the one the `--inject-stream-cost-ms` recovery fraction is
    computed from -- so it may not lose a burst it counted.

    HOW OFTEN, MEASURED, because the answer is small and the reader should have it: driving the
    shipped file in real chromium against a real SSE response read through a fetch reader at field
    cadence, a driver-side `read()` found a chain still open 10 to 21 times in 2,600 to 5,500
    reads, or 0.4 to 0.65 per cent. A cell opens about eight windows over its streaming phase, so
    what this costs a real run is a fraction of one burst chain. It is pinned here rather than left
    because the file's own contract is that a burst is charged once, from the first chunk to the
    loop draining, and a burst that is charged to nothing breaks it in the direction that reads
    cheaper.

    The same-task close below is the deterministic way to put the accumulator in the state chromium
    reaches by racing; the invariant it pins is the production one.
    """
    out = burst_across_a_window_close(BURST_CHAIN_MS, read_while_pending = True)

    assert out["first"]["delta_task_ms"] >= BURST_CHAIN_MS * 0.9, (
        "the burst in flight when the window closed was dropped from its own window",
        out,
    )
    assert out["second"]["delta_task_ms"] < BURST_CHAIN_MS * 0.1, (
        "the burst was charged a second time to the window that followed",
        out,
    )


def test_a_burst_whose_chain_has_already_closed_is_charged_once():
    """THE CONTROL, and it passes with or without the flush: the ordinary ordering, where the
    chain reaches its macrotask before the window closes, must keep charging exactly once."""

    out = burst_across_a_window_close(BURST_CHAIN_MS, read_while_pending = False)

    assert out["first"]["delta_task_ms"] >= BURST_CHAIN_MS * 0.9, out
    assert out["second"]["delta_task_ms"] < BURST_CHAIN_MS * 0.1, out


#: `MAX_SSE_CHUNK_CHARS` in the instrument. A single decode above it is the case that used to be
#: discarded whole.
MAX_SSE_CHUNK_CHARS = 65536


def test_a_batched_sse_read_above_the_decoder_scan_bound_is_still_detected():
    """REGRESSION. The detector read `out.length <= MAX_SSE_CHUNK_CHARS` and skipped anything
    longer, on the premise that a decode that large is not relay traffic.

    A read does not carry one cadence gap of the stream, it carries everything the browser buffered
    since the last one, so its size is the arrival rate times the stall in front of it. Measured
    against real chromium reading the real pacer through the app's own `getReader()` loop, the
    largest read of a stream is 32.5 characters per millisecond of stall at fast cadence: a 3,000 ms
    stall lands one well-formed 97,500 character read of 470 `data:` frames, and the guard dropped
    it entirely -- `sseChunks`, `sseBursts`, `lastSseAt` and the `deltaTaskMs` numerator all missed
    the largest burst of the stream, at the moment a stall makes it largest.
    """
    out = one_decoded_batch("batch-over", BURST_CHAIN_MS)

    assert out["decoded_chars"] > MAX_SSE_CHUNK_CHARS, out
    # The decode was seen either way; what used to be lost is everything downstream of it.
    assert out["read"]["decode_calls"] == 1, out
    assert out["read"]["sse_chunks"] == 1, out
    assert out["read"]["sse_bursts"] == 1, out
    assert out["read"]["streaming_observed"] is True, out
    # ONE chain for the batch, charged once: a burst delivered in one task is one task chain.
    assert out["read"]["delta_task_ms"] >= BURST_CHAIN_MS * 0.9, out


def test_a_batched_sse_read_below_the_decoder_scan_bound_is_detected_too():
    """THE CONTROL, and it passes with or without the fix: the same batch, built to sit under the
    bound, is the path that always worked and may not be broken by widening the one above it."""

    out = one_decoded_batch("batch-under", BURST_CHAIN_MS)

    assert out["decoded_chars"] < MAX_SSE_CHUNK_CHARS, out
    assert out["read"]["sse_chunks"] == 1, out
    assert out["read"]["delta_task_ms"] >= BURST_CHAIN_MS * 0.9, out


def test_a_decoded_blob_above_the_scan_bound_is_still_kept_out_of_the_detector():
    """THE OTHER CONTROL, and it also passes with or without the fix: the bound still has a job.

    A bundle, a blob or a paste reaches the same wrapper, and counting one as a stream would set
    `lastSseAt`, open a chain and charge that chain's cost to a window with no stream in it -- an
    error in the other direction, which the bound exists to prevent. Widening the guard from a
    rejection to a bounded scan may not turn it into no guard at all.
    """
    out = one_decoded_batch("blob", BURST_CHAIN_MS)

    assert out["decoded_chars"] > MAX_SSE_CHUNK_CHARS, out
    assert out["read"]["decode_calls"] == 1, out
    assert out["read"]["sse_chunks"] == 0, out
    assert out["read"]["streaming_observed"] is False, out
    assert out["read"]["delta_task_ms"] == 0, out


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
