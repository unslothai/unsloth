# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs the shipped streamcost.js under node, since a Python port would drift; skips without node."""

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

# Mirrors IDLE_GAP_MS in streamcost.js.
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


# Same file, driven through the ordering where a window closes before the chain's macrotask.
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


# Same file, driven through one decode() of a whole batched read.
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


BURST_CHAIN_MS = 40.0


def test_a_burst_still_in_flight_at_the_window_close_is_charged_to_that_window():
    """A burst still in flight at window close must be charged to that window, not dropped by reset()."""
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


# Mirrors MAX_SSE_CHUNK_CHARS in streamcost.js.
MAX_SSE_CHUNK_CHARS = 65536


def test_a_batched_sse_read_above_the_decoder_scan_bound_is_still_detected():
    """Batched SSE reads above MAX_SSE_CHUNK_CHARS must still be detected, since stalls make them
    largest."""
    out = one_decoded_batch("batch-over", BURST_CHAIN_MS)

    assert out["decoded_chars"] > MAX_SSE_CHUNK_CHARS, out
    assert out["read"]["decode_calls"] == 1, out
    assert out["read"]["sse_chunks"] == 1, out
    assert out["read"]["sse_bursts"] == 1, out
    assert out["read"]["streaming_observed"] is True, out
    # One chain per batch: a burst delivered in one task is charged once.
    assert out["read"]["delta_task_ms"] >= BURST_CHAIN_MS * 0.9, out


def test_a_batched_sse_read_below_the_decoder_scan_bound_is_detected_too():
    """THE CONTROL, and it passes with or without the fix: the same batch, built to sit under the
    bound, is the path that always worked and may not be broken by widening the one above it."""

    out = one_decoded_batch("batch-under", BURST_CHAIN_MS)

    assert out["decoded_chars"] < MAX_SSE_CHUNK_CHARS, out
    assert out["read"]["sse_chunks"] == 1, out
    assert out["read"]["delta_task_ms"] >= BURST_CHAIN_MS * 0.9, out


def test_a_decoded_blob_above_the_scan_bound_is_still_kept_out_of_the_detector():
    """A decoded blob above the scan bound must stay out of the detector, or it is charged as a stream."""
    out = one_decoded_batch("blob", BURST_CHAIN_MS)

    assert out["decoded_chars"] > MAX_SSE_CHUNK_CHARS, out
    assert out["read"]["decode_calls"] == 1, out
    assert out["read"]["sse_chunks"] == 0, out
    assert out["read"]["streaming_observed"] is False, out
    assert out["read"]["delta_task_ms"] == 0, out


def test_a_stall_longer_than_the_idle_gap_is_still_charged_to_the_stream():
    """A stall longer than IDLE_GAP_MS is charged to the stream, not judged by the idle state at its end."""
    stall_ms = IDLE_GAP_MS + 400
    out = drain_after_stall(stall_ms)
    assert out["streaming_observed"] is True
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
    """Overhead must reset per cell; a running total would climb with the rung and look like growth."""
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
    """Mirrors streamcost.js: read() snapshots then resets overheadMs, so driver call order matters."""

    # querySelectorAll scans the whole document, so this is the cost that grows with rung.
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
    """The close-side forced reply scan must be counted in overhead_ms, not lost to the next reset."""
    inst = StreamCostInstrument()
    page = _FakeStreamCostPage()
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
