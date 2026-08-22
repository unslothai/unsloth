// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// The streaming-phase cost accumulator. Installed as an init script before any app code runs.
//
// WHAT IS MISSING FROM THE HARNESS THAT THIS FILLS. `time_in_jank_pct`, `jank_index` and
// `max_frame_ms` do cover the stream -- `_frame_measures` pools every window whose kind is not
// `idle`, and the streaming windows are in that pool. What they cannot do is SEPARATE it. One
// 57.3 s film collapses eighteen action windows and the streaming stretch into a single number,
// and the action windows dominate it: measured on a 100K null control, `reasoning_toggle` alone
// contributes 2,865 ms of blocked time at 99.3% busy with a 1,866 ms worst frame, while the
// streaming stretch next to it runs at 3.6% busy. A change to the streaming path moves the second
// and is scored against the first.
//
// The window KIND cannot be used to make that separation, which is the trap this file exists past.
// `SceneRunner._gap_window` opens EVERY inter-slot gap as `kind = "stream"`, so on the standard
// film eighteen windows are labelled `stream:` and only the first four contain any streaming at
// all; `stream:drain` at the end of the cell is 7 ms of nothing, the stream having finished
// forty seconds earlier. The streaming phase is therefore detected here, from the SSE traffic
// itself, and not read off a label.
//
// THREE THINGS ARE ACCUMULATED, and they are different quantities on purpose:
//
//   deltaTaskMs   The main-thread time of the task chains that SSE chunks start. Measured from
//                 the first decode of a burst to the moment the event loop next reaches a
//                 macrotask, so it spans the SSE parse, the delta accumulation, the cumulative
//                 re-parse and React's own render and commit. This is the TARGETED numerator: it
//                 excludes the background churn -- highlighting, GC, the app's own timers -- that
//                 a whole-window blocked-time figure charges to the stream.
//
//   blockedMs     Blocked time, on exactly the discipline frames.js uses (a 1 ms timer, gap minus
//                 the calibrated clamp), but ACCUMULATED ONLY WHILE THE STREAM IS RUNNING. This
//                 is the BROAD numerator, and it is the honest one: it catches stream-driven cost
//                 that lands outside the delta's own task chain, which is most of the async
//                 highlighting work.
//
//   streamingMs   How much of the window the stream was actually running, so a window that is
//                 half stream and half idle is not read as a whole window of streaming.
//
// WHY THE END OF A TASK CHAIN IS TIMED WITH MessageChannel AND NOT setTimeout. A nested
// `setTimeout(fn, 0)` is clamped to 4 ms once the nesting level passes five, and a chunk arriving
// inside a fetch reader continuation inherits that nesting. That clamp would add a systematic
// ~4 ms to every chunk -- about 720 ms over a 13 s stream -- which is larger than most effects
// this is meant to resolve. A MessageChannel message is not clamped and runs as a macrotask
// immediately after the current task and its microtasks, which is exactly the boundary wanted.
//
// This is NOT the MessageChannel pattern frames.js bans. That one is a ping-pong LOOP that
// re-posts itself and ticks about 150,000 times a second, halving the frame rate before any app
// code runs. This posts once per SSE burst, about fourteen times a second at field cadence, and
// only while a stream is in flight.
//
// AND IT DOES NOT TOUCH requestAnimationFrame. frames.js owns the one rAF loop that counts
// frames. Nothing here schedules, wraps or counts a rAF, so the 888-fps inversion cannot recur
// through this file.

(() => {
  if (window.__sb && window.__sb.streamcost) return;
  window.__sb = window.__sb || {};

  // How long after the last SSE chunk the stream is still considered in flight. Field cadence is
  // 24 characters every 73 ms and the pacer is deficit-scheduled, so it bursts after a jam rather
  // than sleeping a gap per chunk; a jammed renderer can therefore go quiet for a while and then
  // receive a burst. 1500 ms is about twenty cadence gaps, long enough not to chop a jam in half
  // and short enough that the forty seconds of post-stream film are never counted as streaming.
  const IDLE_GAP_MS = 1500;

  // A decoded chunk longer than this is not an SSE frame from the relay; it is a bundle, a blob
  // or a paste. The cap keeps the substring scan below O(1) in practice and keeps unrelated
  // TextDecoder traffic out of the stream detector.
  const MAX_SSE_CHUNK_CHARS = 65536;

  const S = {
    sseChunks: 0,
    sseBursts: 0,
    deltaTaskMs: 0,
    blockedMs: 0,
    streamingMs: 0,
    lastSseAt: 0,
    // Wall time this file spent inside its own hooks, so the overhead it declares is measured
    // rather than asserted. An instrument that guesses its own cost cannot be checked against
    // the overhead_growth_with_length gate.
    overheadMs: 0,
    decodeCalls: 0,
    everStreamed: false,
  };

  const now = () => performance.now();
  // Takes the instant to judge, rather than reading the clock itself, so a caller that has to
  // attribute a whole INTERVAL can ask about the instant the interval began. See the timer below.
  const streamingAt = (t) => S.lastSseAt > 0 && t - S.lastSseAt < IDLE_GAP_MS;

  // ── the task-chain timer ──────────────────────────────────────────────────────────────────
  //
  // One pending measurement at a time. A burst of chunks delivered in one task must be charged
  // once, from the first chunk to the loop draining, not once per chunk: the pacer sends the
  // whole shortfall in a single burst when it has fallen behind, and charging per chunk would
  // multiply one task chain by the number of chunks that started it.
  let chainStart = null;
  const chan = new MessageChannel();
  chan.port1.onmessage = () => {
    if (chainStart === null) return;
    S.deltaTaskMs += now() - chainStart;
    chainStart = null;
  };

  const noteSse = () => {
    S.sseChunks += 1;
    S.everStreamed = true;
    S.lastSseAt = now();
    if (chainStart === null) {
      chainStart = S.lastSseAt;
      S.sseBursts += 1;
      chan.port2.postMessage(0);
    }
  };

  // ── the SSE detector ──────────────────────────────────────────────────────────────────────
  //
  // The app reads the relay's response through its own TextDecoder, so decode() is the first
  // main-thread code that sees a chunk. Wrapping it costs one call per chunk and is O(1) in the
  // thread's size. This hook is the flat part of the instrument; the reply-length read below is
  // the part that is not, and its measured cost is recorded there rather than claimed away here.
  const nativeDecode = TextDecoder.prototype.decode;
  TextDecoder.prototype.decode = function (input, options) {
    const out = nativeDecode.call(this, input, options);
    const t = now();
    S.decodeCalls += 1;
    if (typeof out === "string" && out.length > 0 && out.length <= MAX_SSE_CHUNK_CHARS) {
      if (out.indexOf("data:") >= 0) noteSse();
    }
    S.overheadMs += now() - t;
    return out;
  };

  // ── blocked time while streaming ──────────────────────────────────────────────────────────
  //
  // The same 1 ms timer discipline as frames.js, and deliberately the same calibrated clamp:
  // blocked time is a SUBTRACTION against an idle floor, and two instruments subtracting two
  // different floors would report two different amounts of block for one page. The clamp is read
  // from frames.js rather than calibrated again here, so if frames.js could not establish one
  // this reports null with frames.js's own reason.
  let lastTick = now();
  const tick = () => {
    const t = now();
    const gap = t - lastTick;
    // Attributed by the state at the START of the interval, never at its end. A timer callback
    // cannot run while the main thread is blocked -- it is queued when the timer expires and
    // waits for the stack to empty -- so a stall is only ever observed once it is already over.
    // Reading `streaming()` here would therefore ask whether the stream is in flight NOW, after
    // the stall, and a stall longer than IDLE_GAP_MS answers no: the whole interval, which IS
    // the stall, would be dropped from both accumulators. That discards precisely the worst
    // stream-induced stalls, and it discards them at exactly the point where a regression grows
    // past 1.5 s, so a build getting worse would read cheaper. A negative difference (a chunk
    // that arrived DURING the interval) is below the threshold too, which is the overlap case.
    const wasStreaming = streamingAt(lastTick);
    lastTick = t;
    if (wasStreaming) {
      const f = window.__sb.frames;
      const clamp = f && f.clamp ? f.clamp().clampMs : null;
      S.streamingMs += gap;
      if (clamp !== null && clamp !== undefined) S.blockedMs += Math.max(0, gap - clamp);
    }
    setTimeout(tick, 1);
  };
  setTimeout(tick, 1);

  // Characters of the reply currently being streamed. The LAST assistant message only, not the
  // whole thread: at the 100K rung the thread holds about 190,000 assistant characters and
  // reading all of them costs real time on every window boundary, while the quantity wanted is
  // the growth of the one reply the pacer is feeding. Reading the last message alone is O(reply)
  // rather than O(thread), so this stays flat as the rung climbs.
  //
  // It is also the only reading that survives the end of the film. `thread_reopen` and
  // `delete_message` rebuild and then cut the thread, so a whole-thread character count jumps and
  // then falls for reasons that have nothing to do with streaming.
  // MEASURED, and it is not free. `querySelectorAll` is O(the whole DOM), not O(the reply), so
  // this is the one part of the instrument whose cost grows with the rung. Called at both ends of
  // every window it totalled 38.8 ms per cell at 10K and 289.6 ms at 100K -- about 3.9 ms per
  // call against 42,000 elements.
  //
  // Most of that was spent on windows with no stream in them. The standard film opens about
  // thirty-seven windows and only eight carry streaming, so the read is skipped once the stream
  // has been finished for longer than the idle gap. `null` is returned rather than a stale count,
  // and a window that then turns out to have carried traffic reports its growth as unmeasurable
  // with a reason instead of inventing a delta from a reading that was never taken.
  const replyChars = (force) => {
    if (!force && S.lastSseAt > 0 && now() - S.lastSseAt >= IDLE_GAP_MS) return null;
    const all = document.querySelectorAll('[data-role="assistant"]');
    if (all.length === 0) return null;
    const el = all[all.length - 1];
    return (el.textContent || "").length;
  };

  window.__sb.streamcost = {
    // Drain the window. `elapsedMs` is the DRIVER's measure, passed in for the same reason
    // frames.js takes it: the page cannot be trusted to read its own clock promptly in the very
    // windows this is measuring.
    read(elapsedMs) {
      const t = now();
      const f = window.__sb.frames;
      const clampInfo = f && f.clamp ? f.clamp() : { clampMs: null, reason: "frames.js absent" };
      const out = {
        sse_chunks: S.sseChunks,
        sse_chunks_attempted: true,
        sse_bursts: S.sseBursts,
        decode_calls: S.decodeCalls,
        // Never a bare zero: a window with no streaming in it says so with a flag, and the
        // scoring layer skips it rather than folding a zero cost into the numerator.
        streaming_observed: S.sseChunks > 0,
        streaming_ms: Math.round(S.streamingMs * 10) / 10,
        streaming_ms_attempted: true,
        delta_task_ms: Math.round(S.deltaTaskMs * 10) / 10,
        delta_task_ms_attempted: true,
        driver_elapsed_ms: elapsedMs === null || elapsedMs === undefined ? null : elapsedMs,
        clamp_ms: clampInfo.clampMs === null ? null : clampInfo.clampMs,
      };
      if (clampInfo.clampMs === null || clampInfo.clampMs === undefined) {
        // Blocked time is a subtraction against the idle floor. Without a floor there is no
        // subtraction to make, and reporting the raw lag instead would be a different quantity
        // wearing this one's name.
        out.stream_blocked_ms = null;
        out.stream_blocked_ms_reason =
          "no timer clamp was established, so there is no idle floor to subtract: " +
          (clampInfo.reason || "unknown");
      } else {
        out.stream_blocked_ms = Math.round(S.blockedMs * 10) / 10;
        out.stream_blocked_ms_attempted = true;
      }
      S.overheadMs += now() - t;
      out.overhead_ms = Math.round(S.overheadMs * 100) / 100;
      out.overhead_attempted = true;
      this.reset();
      return out;
    },

    // Read the streamed reply's length WITHOUT draining anything. Called at window open and
    // window close, so the denominator is the growth across the window.
    // `force` is passed at window CLOSE when the window turned out to carry traffic, so a
    // window whose open-read was skipped is not silently given a null close-read as well.
    replyChars(force) {
      const t = now();
      const n = replyChars(Boolean(force));
      S.overheadMs += now() - t;
      return n;
    },

    reset() {
      S.sseChunks = 0;
      S.sseBursts = 0;
      S.deltaTaskMs = 0;
      S.blockedMs = 0;
      S.streamingMs = 0;
      S.decodeCalls = 0;
      S.overheadMs = 0;
    },

    // For the selftest: force the detector into the streaming state without a real stream, so a
    // synthetic injection can be measured on exactly the accumulators a real stream uses.
    __markStreaming() {
      noteSse();
    },
  };
})();
