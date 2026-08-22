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
    // CUMULATIVE characters of assistant text this page has been SENT, counted off the wire.
    // Never reset by `reset()`: the denominator a window wants is the growth across it, which is
    // the difference of two readings of this, and resetting it per window would make every
    // reading zero.
    wireChars: 0,
    wireFrames: 0,
    wireParseFailures: 0,
  };
  // The incremental SSE buffer. A decode() call is a slice of the socket, not an SSE frame: one
  // call can carry three frames and half of a fourth. Whatever is left after the last blank line
  // stays here until the rest of it arrives.
  let pending = "";
  // A frame is a few hundred bytes. If this ever grows past a sane bound the stream is not what
  // we think it is, and dropping the buffer is better than growing it without limit inside a hook
  // that runs fourteen times a second.
  const MAX_PENDING_CHARS = 262144;

  // ── the frame marker, and the halves of it a split can leave ──────────────────────────────────
  //
  // The socket can cut a frame ANYWHERE, including inside these five characters: one decode()
  // returns "da" and the next "ta: {...}\n\n". Neither chunk contains the marker, so the detector
  // below saw two unrelated chunks, discarded both, and lost the frame WITHOUT counting a parse
  // failure -- which left the wire character count short with nothing anywhere to say so, and
  // `reply_chars_scoreable` reporting the window as sound. The denominator going quietly short
  // inflates every cost-per-character above it, and it goes short exactly when the renderer is
  // jammed and chunks arrive ragged, which is the moment the instrument exists to measure.
  const SSE_MARKER = "data:";
  // The fragment of the marker the last chunk MIGHT have ended on. At most four characters ("d",
  // "da", "dat", "data"), so this cannot become the memory hazard MAX_PENDING_CHARS guards
  // `pending` against: buffering whole unrelated TextDecoder chunks on the chance that one of them
  // is an SSE frame would be a worse bug than the one being fixed.
  let markerTail = "";

  //: The longest tail of `s` that is a PROPER PREFIX of the marker, or "".
  const partialMarkerTail = (s) => {
    for (let n = Math.min(SSE_MARKER.length - 1, s.length); n > 0; n -= 1) {
      if (s.endsWith(SSE_MARKER.slice(0, n))) return SSE_MARKER.slice(0, n);
    }
    return "";
  };

  //: Does `s` CONTINUE the marker that `frag` started? This is what keeps a speculative fragment
  //: harmless. Unrelated traffic ending in "d" would otherwise be glued onto the front of the next
  //: chunk, and a real frame arriving there would become "ddata: {...}", which does not start with
  //: the marker and would be skipped in silence -- the same defect one step to the left.
  const continuesMarker = (frag, s) => {
    const rest = SSE_MARKER.slice(frag.length);
    const n = Math.min(rest.length, s.length);
    return n > 0 && s.slice(0, n) === rest.slice(0, n);
  };

  const now = () => performance.now();
  const streaming = () => S.lastSseAt > 0 && now() - S.lastSseAt < IDLE_GAP_MS;

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

  // ── the wire-side character counter ───────────────────────────────────────────────────────
  //
  // THIS REPLACES AN O(DOCUMENT) READ THAT BIASED THE COMPARISON, and the bias only became
  // visible once an arm existed that changes the size of the document.
  //
  // The denominator used to be read from the DOM: `querySelectorAll('[data-role="assistant"]')`,
  // last element, `textContent.length`, at both ends of every window. That is O(the whole
  // document) regardless of how few elements match, and it measured 3.9 ms per call against
  // 42,000 elements -- 38.8 ms per cell at 10K and 289.6 ms at 100K. The file's own note said it
  // "is identical on both arms of an A/B and cancels in a paired ratio". That was true of every
  // arm this project had ever run, and it is FALSE for a virtualised one: an arm whose entire
  // purpose is to put a tenth of the elements in the document pays a tenth of this cost, so the
  // instrument hands the treatment a saving it did not earn, in the direction that flatters
  // exactly the hypothesis under test. Nothing measured on such an arm could be quoted while that
  // read was in the paired path.
  //
  // Counting off the wire removes it rather than balancing it. Both arms are fed by the SAME
  // pacer -- that is a design invariant of runtime/ab.py, not a coincidence -- so the bytes are
  // identical by construction and this counter is identical by construction. It is O(the chunk),
  // about fourteen chunks a second, and independent of the thread's size, the rung and the arm.
  //
  // It is also a BETTER denominator than the one it replaces. The DOM read measured the last
  // assistant message, so a `send_turn` mid-film made the reading shrink and the window's growth
  // unmeasurable ("it is a different message"). Characters delivered in the window is the
  // quantity the cost per character actually wants, and it does not care how many messages they
  // were spread over.
  const countDeltaChars = (text) => {
    pending += text;
    if (pending.length > MAX_PENDING_CHARS) {
      S.wireParseFailures += 1;
      pending = "";
      return;
    }
    // Frames are separated by a blank line. Anything after the last one is incomplete.
    const parts = pending.split("\n\n");
    pending = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const body = line.slice(5).trim();
      if (body === "" || body === "[DONE]") continue;
      try {
        const frame = JSON.parse(body);
        const choices = frame && frame.choices;
        if (!choices || !choices.length) continue;
        const delta = choices[0].delta || {};
        // Both fields, and both are counted. `_gguf_chat_delta_line` emits reasoning as
        // `reasoning_content` WITH `content: ""` beside it, so summing them is not double
        // counting; it is the two halves of one turn.
        const content = typeof delta.content === "string" ? delta.content.length : 0;
        const reasoning =
          typeof delta.reasoning_content === "string" ? delta.reasoning_content.length : 0;
        S.wireChars += content + reasoning;
        S.wireFrames += 1;
      } catch (err) {
        // COUNTED, NOT SWALLOWED. A parse failure means the denominator is short by an unknown
        // amount, and a silently short denominator inflates every cost-per-character above it.
        S.wireParseFailures += 1;
      }
    }
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
      // The fragment the previous chunk ended on, but only if THIS chunk continues it. A split
      // inside the marker is repaired here rather than in the buffer, so a fragment that turns out
      // to be ordinary text ending in "d" is dropped instead of corrupting the frame behind it.
      // `markerTail` is only ever set when `pending` is empty, so the two can never both hold a
      // half of the same frame.
      const chunk = markerTail && continuesMarker(markerTail, out) ? markerTail + out : out;
      markerTail = "";
      const looksSse = chunk.indexOf(SSE_MARKER) >= 0;
      if (looksSse) noteSse();
      // `looksSse || pending` and not just `looksSse`. THE SECOND HALF OF A SPLIT FRAME CONTAINS
      // NO "data:" -- it is the tail of a JSON body and a blank line -- so gating the counter on
      // that marker dropped the whole frame whenever the socket cut one in two. Found by
      // test_the_counter_survives_a_frame_split_across_two_decode_calls, which is precisely the
      // condition the instrument exists to measure: chunks arrive ragged when the renderer is
      // jammed, so the denominator would have gone quietly short exactly where the numerator went
      // up, and the cost per character would have been overstated at the worst moment.
      //
      // Once a partial frame is held, every subsequent chunk is fed until it completes. Unrelated
      // TextDecoder traffic can therefore land in the buffer; it cannot be counted, because it
      // will not parse as a frame, and it is bounded by MAX_PENDING_CHARS and reported through
      // wire_parse_failures rather than absorbed.
      //
      // AND THE CHUNK THAT IS NEITHER is kept only as far as it could be the START of a marker.
      // That is the third case, the one a complete-marker test cannot see: "da" carries no marker
      // and completes no buffered frame, and discarding it loses the frame that arrives next.
      if (looksSse || pending.length > 0) countDeltaChars(chunk);
      else markerTail = partialMarkerTail(chunk);
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
    lastTick = t;
    if (streaming()) {
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

    // THE DENOMINATOR, read off the wire. O(1): it returns a counter the decode hook maintains.
    // Called at window open and window close, so the window's growth is the difference.
    //
    // Cumulative since page load and monotonic, so unlike the DOM reading it replaces there is no
    // "the reply shrank, so it is a different message" case to handle: characters delivered only
    // ever go up.
    // The two things that can make `wireChars` short by an unknown amount, read at the same O(1)
    // cost as the counter itself so a window boundary can capture both ends.
    wireIntegrity() {
      // The marker fragment counts as buffered, because it is: the frame it begins has not been
      // counted yet, so a window closing on it has a denominator that is short by that frame. The
      // cost of being honest here is that a stray one to four characters of unrelated traffic can
      // mark a window unscoreable, which is the direction to err in -- an unscoreable window is
      // "we could not tell", and a silently short denominator is "it was fine".
      return { failures: S.wireParseFailures, pending_chars: pending.length + markerTail.length };
    },
    replyChars() {
      return S.wireChars;
    },

    // The OLD reading, kept as a cross-check and NEVER called inside a measured window. See
    // `end_cell` in streamcost.py: it runs once per cell, after the film, where its cost is
    // charged to nothing.
    //
    // Worth keeping rather than deleting, because the two numbers answer different questions and
    // a disagreement between them is a finding: the wire count is what the app was SENT and the
    // DOM count is what it RENDERED. On a windowed arm they are expected to disagree, by exactly
    // the messages that are not mounted.
    replyCharsDom(force) {
      const t = now();
      const n = replyChars(Boolean(force));
      S.overheadMs += now() - t;
      return n;
    },

    wireStats() {
      return {
        wire_chars: S.wireChars,
        wire_frames: S.wireFrames,
        wire_parse_failures: S.wireParseFailures,
        wire_pending_chars: pending.length + markerTail.length,
      };
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
