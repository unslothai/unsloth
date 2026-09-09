// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// The streaming-phase cost accumulator, installed as an init script before any app code runs.
// WHAT THE HARNESS CANNOT DO: `time_in_jank_pct`, `jank_index` and `max_frame_ms` cover the
// stream but cannot SEPARATE it. One 57.3 s film collapses eighteen action windows and the
// streaming stretch into one number; on a 100K null control `reasoning_toggle` alone
// contributes 2,865 ms of blocked time at 99.3% busy with a 1,866 ms worst frame, while the
// streaming stretch next to it runs at 3.6% busy.
// The window KIND cannot separate them either: `SceneRunner._gap_window` opens every
// inter-slot gap as `kind = "stream"`, so eighteen windows are labelled `stream:` and only
// four carry streaming. The phase is detected here from the SSE traffic itself.
// THREE ACCUMULATORS, different quantities on purpose. `deltaTaskMs`: main-thread time of the
// task chains SSE chunks start, first decode of a burst to the next macrotask (the TARGETED
// numerator). `blockedMs`: frames.js blocked time, but only while the stream runs (the BROAD
// numerator, catching async highlighting off the chain). `streamingMs`: how much of the window
// the stream was actually running.
// Task-chain end is timed with MessageChannel, not setTimeout: a nested `setTimeout(fn, 0)` is
// clamped to 4 ms past nesting level five, ~720 ms over a 13 s stream. Not the ping-pong loop
// frames.js bans (one post per SSE burst, ~14/s, only in flight) and it never touches rAF, so
// the 888-fps inversion cannot recur here.

(() => {
  if (window.__sb && window.__sb.streamcost) return;
  window.__sb = window.__sb || {};

  // How long after the last SSE chunk the stream still counts as in flight. The
  // deficit-scheduled pacer bursts after a jam rather than sleeping per chunk, so 1500 ms is
  // about twenty cadence gaps: long enough not to chop a jam, short enough that the post-stream
  // film is never counted as streaming.
  // Field cadence is 24 characters every 73 ms.
  const IDLE_GAP_MS = 1500;

  // How much of a decoded chunk is scanned for the relay's framing: the substring search below
  // is the per-event work this file declares O(1), and an unbounded scan would make it O(the
  // page's whole TextDecoder traffic).
  // IT BOUNDS THE SCAN, NOT THE PAYLOAD. The guard used to drop any decode longer than the cap,
  // on the false premise that a large decode is not relay traffic: a read carries everything
  // buffered since the last one, and at fast cadence the largest read is 32.5 characters per ms
  // of stall, so a 3,000 ms stall lands a well-formed 97,500 character read of 470 `data:`
  // frames, which the old guard discarded whole. Swept against the real pacer that cost 46.8% to
  // 58.2% of the numerator, and nothing at 0 ms per 1,000 characters: it hid on a harness that
  // does no per-character work and got worse as the app got slower. Scanning a bounded head
  // keeps both original goals: constant-bounded work, and a bundle must still put `data:` in its
  // first 65,536 characters.
  // Field cadence arrives at 2.97 characters per millisecond.
  // Measured in v8: 0.04 us on an SSE batch and 0.68 us on a 2 MB blob with no frame in it,
  // against 0.70 us for the same scan left unbounded.
  const MAX_SSE_CHUNK_CHARS = 65536;

  const S = {
    sseChunks: 0,
    sseBursts: 0,
    deltaTaskMs: 0,
    blockedMs: 0,
    streamingMs: 0,
    lastSseAt: 0,
    // Wall time spent inside this file's own hooks, so its declared overhead is measured rather
    // than asserted; an instrument that guesses its own cost cannot be checked against the
    // overhead_growth_with_length gate.
    overheadMs: 0,
    decodeCalls: 0,
    everStreamed: false,
    // CUMULATIVE characters of assistant text this page has been SENT, counted off the wire. Never
    // reset by `reset()`: a window wants the growth, which is the difference of two readings.
    wireChars: 0,
    wireFrames: 0,
    wireParseFailures: 0,
  };
  // The incremental SSE buffer: a decode() call is a slice of the socket, so one call can carry
  // three frames and half of a fourth. PER DECODER, NOT PER PAGE: a `TextDecoder` belongs to one
  // response, and a page-wide buffer modelled the socket instead of reassembly, so a
  // `stop_generation` cutting a socket mid-frame glued the next response's first chunk behind an
  // unclosed JSON tail, skipped without counting a parse failure while `pending_chars` stayed
  // above zero and refused every later window. Keyed weakly, so the buffer dies with its
  // decoder.
  // It failed `startsWith("data:")`.
  const DECODER_STATE = new WeakMap();
  //: How many times a frame already buffered when a decode call began was completed and counted.
  //: Cumulative and never reset, like `wireChars`: this turns "a buffer was pending at the open"
  //: into "characters that arrived before this window were counted inside it". An aborted
  //: response's buffer never completes, so it never increments this. PER DECODER, like `pending`:
  //: page-global, `close` paired one decoder's buffer with flushes counted across all of them, so
  //: a split after a `stop_generation` refused the window for a stale buffer that never flushed.
  const CARRIED_BY_ID = new Map();
  //: A bounded history, on the same discipline as `MAX_PENDING_CHARS`: `close` asks about the
  //: decoder pending when the window OPENED, which may be neither active nor alive, so the count
  //: cannot live only in the `WeakMap`. Keyed by an integer holding no reference to any
  //: `TextDecoder`, and trimmed so a long film cannot grow it without limit.
  const MAX_DECODER_HISTORY = 64;
  let DECODER_SEQ = 0;
  const noteCarried = (st) => {
    st.carriedFlushes += 1;
    CARRIED_BY_ID.set(st.id, st.carriedFlushes);
    while (CARRIED_BY_ID.size > MAX_DECODER_HISTORY) {
      CARRIED_BY_ID.delete(CARRIED_BY_ID.keys().next().value);
    }
  };
  //: The carried count of a NAMED decoder, or of whichever is active when no name is given. A
  //: decoder that never carried a frame is absent and answers 0, as it would have while alive.
  const carriedFor = (id) => {
    if (typeof id !== "number") return active.carriedFlushes;
    return CARRIED_BY_ID.get(id) || 0;
  };
  const newState = () => ({
    pending: "",
    markerTail: "",
    carriedFlushes: 0,
    id: (DECODER_SEQ += 1),
  });
  const stateFor = (decoder) => {
    let st = DECODER_STATE.get(decoder);
    if (!st) {
      st = newState();
      DECODER_STATE.set(decoder, st);
    }
    return st;
  };
  //: The decoder that most recently delivered a chunk, i.e. the stream a window is measuring.
  //: `wireIntegrity` reports THIS buffer: half a frame of the measured response is a short
  //: denominator, while half a frame of a response aborted three slots ago says nothing. Built by
  //: `newState()` so a window opening before the first stream reads zero, not undefined.
  let active = newState();
  //: The decoder holding a speculative marker fragment, if any. A fragment is at most four
  //: characters and lives on the decoder that produced it, but must be reported whoever holds it:
  //: a decoder whose first chunk is "dat" is not identified as the stream yet, and omitting its
  //: fragment would claim nothing was outstanding while a frame was. Erring the other way costs
  //: at most four characters marking a window unscoreable. Dropped once another decoder is the
  //: stream.
  let markerHold = null;
  const setMarkerTail = (st, frag) => {
    st.markerTail = frag;
    if (frag) markerHold = st;
    else if (markerHold === st) markerHold = null;
  };
  const heldMarkerChars = () =>
    active.markerTail.length +
    (markerHold && markerHold !== active ? markerHold.markerTail.length : 0);
  // A frame is a few hundred bytes; past a sane bound the stream is not what we think it is, and
  // dropping the buffer beats growing it without limit in a hook that runs 14 times a second.
  const MAX_PENDING_CHARS = 262144;

  // The socket can cut a frame anywhere, including inside these five characters: one decode()
  // returns "da" and the next "ta: {...}\n\n". Neither contains the marker, so the detector
  // discarded both and lost the frame without counting a parse failure, leaving the denominator
  // short exactly when the renderer is jammed and chunks arrive ragged.
  const SSE_MARKER = "data:";
  // The fragment of the marker the last chunk might have ended on: at most four characters, so
  // it cannot become the memory hazard MAX_PENDING_CHARS guards `pending` against. Lives in the
  // per-decoder state, for the same reason `pending` does.

  //: The longest tail of `s` that is a PROPER PREFIX of the marker, or "".
  const partialMarkerTail = (s) => {
    for (let n = Math.min(SSE_MARKER.length - 1, s.length); n > 0; n -= 1) {
      if (s.endsWith(SSE_MARKER.slice(0, n))) return SSE_MARKER.slice(0, n);
    }
    return "";
  };

  //: Does `s` CONTINUE the marker `frag` started? Without this, unrelated traffic ending in "d"
  //: would be glued onto the next chunk and a real frame would become "ddata: {...}", skipped in
  //: silence: the same defect one step to the left.
  const continuesMarker = (frag, s) => {
    const rest = SSE_MARKER.slice(frag.length);
    const n = Math.min(rest.length, s.length);
    return n > 0 && s.slice(0, n) === rest.slice(0, n);
  };

  const now = () => performance.now();
  // Takes the instant to judge rather than reading the clock, so a caller attributing a whole
  // INTERVAL can ask about the instant it began. See the timer below.
  const streamingAt = (t) => S.lastSseAt > 0 && t - S.lastSseAt < IDLE_GAP_MS;

  // One pending measurement at a time: a burst delivered in one task must be charged once, from
  // the first chunk to the loop draining. The pacer sends the whole shortfall in one burst when
  // behind, and charging per chunk would multiply one chain by the chunks that started it.
  let chainStart = null;
  // Close whatever chain is open and charge it to the accumulator it was opened against. Called
  // from the MessageChannel callback and from `read()`, the case that used to lose it: `read()`
  // arrives on its own task at a window boundary, on a different queue, so a window could close
  // between a burst's decode and its chain's macrotask, counting the characters and losing the
  // cost. Reproduced in chromium: 0 to 2 of 200 bursts lost per run, always downward. A stale
  // message is harmless, finding `chainStart === null`.
  const closeChain = () => {
    if (chainStart === null) return;
    S.deltaTaskMs += now() - chainStart;
    chainStart = null;
  };
  const chan = new MessageChannel();
  chan.port1.onmessage = closeChain;

  // THIS REPLACES AN O(DOCUMENT) READ THAT BIASED THE COMPARISON. The denominator used to be a
  // `querySelectorAll('[data-role="assistant"]')` last-element textContent read at both ends of
  // every window: 38.8 ms per cell at 10K and 289.6 ms at 100K. "It cancels in a paired ratio"
  // is FALSE for a virtualised arm, which pays a tenth of the cost and is handed a saving it did
  // not earn, flattering the hypothesis under test. Counting off the wire removes it: both arms
  // are fed by the SAME pacer (a runtime/ab.py invariant), so the counter is identical by
  // construction, O(the chunk), and independent of thread size, rung and arm. It is also the
  // better denominator: the DOM read shrank on a mid-film `send_turn`.
  // About 3.9 ms per call against 42,000 elements.
  // `textContent.length` over the whole document.
  const countDeltaChars = (st, text, carriedMarker) => {
    const carried = st.pending.length > 0 || Boolean(carriedMarker);
    st.pending += text;
    if (st.pending.length > MAX_PENDING_CHARS) {
      S.wireParseFailures += 1;
      st.pending = "";
      return;
    }
    // Frames are separated by a blank line; anything after the last one is incomplete.
    const parts = st.pending.split("\n\n");
    st.pending = parts.pop();
    // Something buffered before this call just became a counted frame. A window whose OPEN saw a
    // non-empty buffer is only wrong if this happens inside it.
    if (carried && parts.length > 0) noteCarried(st);
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
        // Both fields, and both counted: `_gguf_chat_delta_line` emits reasoning as
        // `reasoning_content` with `content: ""` beside it, so summing them is the two halves of one
        // turn, not double counting.
        const content = typeof delta.content === "string" ? delta.content.length : 0;
        const reasoning =
          typeof delta.reasoning_content === "string" ? delta.reasoning_content.length : 0;
        S.wireChars += content + reasoning;
        S.wireFrames += 1;
      } catch (err) {
        // COUNTED, NOT SWALLOWED: a parse failure means the denominator is short by an unknown
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

  // The app reads the relay's response through its own TextDecoder, so decode() is the first
  // main-thread code to see a chunk: one call per chunk, O(1) in thread size. This hook is the
  // flat part of the instrument; the reply-length read below is not, and records its own cost.
  const nativeDecode = TextDecoder.prototype.decode;
  TextDecoder.prototype.decode = function (input, options) {
    const out = nativeDecode.call(this, input, options);
    const t = now();
    S.decodeCalls += 1;
    if (typeof out === "string" && out.length > 0) {
      // Reassembly state belongs to THIS decoder. Whether it is also the stream being MEASURED is
      // decided below, once the chunk has been looked at: promoting here handed `active` to any
      // decoder in the page, so `wireIntegrity` reported nothing outstanding while an SSE decoder
      // held half a frame.
      const st = stateFor(this);
      // The fragment the previous chunk ended on, but only if THIS chunk continues it: a split
      // inside the marker is repaired here rather than in the buffer, so a fragment that turns out
      // to be ordinary text ending in "d" is dropped. `markerTail` is only set when `pending` is
      // empty, so the two can never hold halves of the same frame.
      const carriedMarker = Boolean(st.markerTail && continuesMarker(st.markerTail, out));
      const chunk = carriedMarker ? st.markerTail + out : out;
      setMarkerTail(st, "");
      // THE BOUND IS ON THE SCAN, NOT THE PAYLOAD. A chunk at or under the cap is its own head, so
      // the ordinary path allocates nothing, and over the cap v8 slices by reference. Both original
      // goals survive: a constant-bounded search, and a bundle must still put the marker in its
      // first MAX_SSE_CHUNK_CHARS characters. What does not survive is the old rejection, which
      // dropped a well-formed 97,500 character batch at the moment a stall made it largest.
      const head = chunk.length <= MAX_SSE_CHUNK_CHARS ? chunk : chunk.slice(0, MAX_SSE_CHUNK_CHARS);
      const looksSse = head.indexOf(SSE_MARKER) >= 0;
      // A CONTINUATION IS STREAM TRAFFIC AND STARTS A TASK CHAIN LIKE ANY OTHER CHUNK. `noteSse`
      // was gated on the marker alone while the counter was gated on `looksSse || pending`, so the
      // tail of a frame cut inside its JSON body was counted in the denominator and charged to
      // nothing, biasing `stream_delta_cost_ms_per_kchar` DOWNWARD as fragmentation rises. A stale
      // `lastSseAt` also lets `replyChars` call the stream idle for a window still carrying it.
      const continuesFrame = st.pending.length > 0;
      if (looksSse || continuesFrame) noteSse();
      // `looksSse || pending`, not just `looksSse`: THE SECOND HALF OF A SPLIT FRAME CONTAINS NO
      // "data:", so gating the counter on the marker dropped the whole frame whenever the socket cut
      // one in two. Chunks arrive ragged when the renderer is jammed, so the denominator went short
      // exactly where the numerator went up. Once a partial frame is held every chunk is fed until
      // it completes; unrelated traffic is bounded by MAX_PENDING_CHARS and reported through
      // wire_parse_failures. A chunk that is neither is kept only as far as it could START a marker:
      // "da" carries no marker and completes no frame, and discarding it loses the next frame.
      // Found by test_the_counter_survives_a_frame_split_across_two_decode_calls.
      // The whole chunk is fed to the counter and only the SCAN was bounded above: the denominator
      // is characters delivered, so counting a batched read's head would understate it by exactly
      // the amount a stall made it large.
      // ONLY NOW IS THIS DECODER THE ONE A WINDOW IS MEASURING: it either carries the relay's
      // framing or is completing a frame of its own, and a decoder that is neither cannot take
      // `active` from one that is.
      if (looksSse || continuesFrame) {
        // A fragment held by a DIFFERENT decoder cannot be part of this stream. Dropped rather than
        // carried, so unrelated traffic ending in "data" cannot report an outstanding frame.
        if (markerHold && markerHold !== st) setMarkerTail(markerHold, "");
        active = st;
        countDeltaChars(st, chunk, carriedMarker);
      } else {
        // A speculative fragment stays on the decoder that produced it and is reported through
        // `wireIntegrity` only if that decoder is the active stream, so unrelated traffic ending in
        // "d" never claims an outstanding frame.
        setMarkerTail(st, partialMarkerTail(chunk));
      }
    }
    S.overheadMs += now() - t;
    return out;
  };

  // The same 1 ms timer discipline as frames.js, and deliberately the same calibrated clamp:
  // blocked time is a subtraction against an idle floor, and two instruments subtracting
  // different floors would report two amounts of block for one page. The clamp is read from
  // frames.js, so if none could be established this reports null with frames.js's reason.
  let lastTick = now();
  const tick = () => {
    const t = now();
    const gap = t - lastTick;
    // Attributed by the state at the START of the interval, never at its end: a timer callback
    // cannot run while the main thread is blocked, so a stall is only observed once over, and
    // reading `streaming()` here would drop the whole interval for any stall longer than
    // IDLE_GAP_MS, discarding the worst stream-induced stalls so a worsening build reads cheaper.
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

  // Characters of the reply currently streaming: the LAST assistant message only, since at 100K
  // the thread holds about 190,000 assistant characters while the quantity wanted is the growth
  // of the one reply the pacer is feeding. O(reply) rather than O(thread), and the only reading
  // that survives `thread_reopen` and `delete_message`. Not free: `querySelectorAll` is O(the
  // whole DOM), 289.6 ms per cell at 100K, and most of it was spent on windows with no stream,
  // so the read is skipped once the stream has been idle past the gap; `null` is returned rather
  // than a stale count, and a window that did carry traffic reports unmeasurable with a reason.
  const replyChars = (force) => {
    if (!force && S.lastSseAt > 0 && now() - S.lastSseAt >= IDLE_GAP_MS) return null;
    const all = document.querySelectorAll('[data-role="assistant"]');
    if (all.length === 0) return null;
    const el = all[all.length - 1];
    return (el.textContent || "").length;
  };

  window.__sb.streamcost = {
    // Drain the window. `elapsedMs` is the DRIVER's measure, passed in for the same reason
    // frames.js takes it: the page cannot be trusted to read its own clock promptly in exactly
    // the windows this is measuring.
    read(elapsedMs) {
      const t = now();
      // BEFORE the snapshot, because a chain still in flight belongs to the window that started it
      // and `reset()` is about to throw it away. See `closeChain`.
      closeChain();
      const f = window.__sb.frames;
      const clampInfo = f && f.clamp ? f.clamp() : { clampMs: null, reason: "frames.js absent" };
      const out = {
        sse_chunks: S.sseChunks,
        sse_chunks_attempted: true,
        sse_bursts: S.sseBursts,
        decode_calls: S.decodeCalls,
        // Never a bare zero: a window with no streaming says so with a flag, and the scoring layer
        // skips it rather than folding a zero cost into the numerator.
        streaming_observed: S.sseChunks > 0,
        streaming_ms: Math.round(S.streamingMs * 10) / 10,
        streaming_ms_attempted: true,
        delta_task_ms: Math.round(S.deltaTaskMs * 10) / 10,
        delta_task_ms_attempted: true,
        driver_elapsed_ms: elapsedMs === null || elapsedMs === undefined ? null : elapsedMs,
        clamp_ms: clampInfo.clampMs === null ? null : clampInfo.clampMs,
      };
      if (clampInfo.clampMs === null || clampInfo.clampMs === undefined) {
        // Blocked time is a subtraction against the idle floor; without a floor there is no
        // subtraction to make, and the raw lag would be a different quantity wearing this one's name.
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

    // THE DENOMINATOR, read off the wire. O(1), called at window open and close so the growth is
    // the difference. Cumulative since page load and monotonic, so unlike the DOM reading it
    // replaces there is no "the reply shrank, so it is a different message" case.
    // The two things that can make `wireChars` short by an unknown amount, at the counter's O(1)
    // cost. `forId` names the decoder to answer about, because the buffer and the flush must
    // belong to the SAME decoder: the one pending at the open can complete its carried frame
    // inside the window and be replaced as active before the close, so comparing the two ends'
    // ids would discard the very carry it looks for.
    wireIntegrity(forId) {
      // The marker fragment counts as buffered, because it is: the frame it begins has not been
      // counted, so a window closing on it is short by that frame. The cost is that one to four
      // characters of unrelated traffic can mark a window unscoreable, the right way to err.
      return {
        failures: S.wireParseFailures,
        pending_chars: active.pending.length + heldMarkerChars(),
        // WHICH decoder the two numbers above are about, so the close can ask about the same one.
        decoder_id: active.id,
        // Read as a DELTA across the window by `StreamCostInstrument.close`, so a buffer pending at
        // the open refuses the window only when its own frame was completed inside it.
        carried_flushes: carriedFor(forId),
      };
    },
    replyChars() {
      return S.wireChars;
    },

    // The OLD reading, kept as a cross-check and never called inside a measured window (see
    // `end_cell` in streamcost.py, once per cell after the film). Worth keeping because the two
    // answer different questions, what the app was SENT versus what it RENDERED, and on a windowed
    // arm they should disagree by exactly the unmounted messages.
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
        wire_pending_chars: active.pending.length + heldMarkerChars(),
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
    // synthetic injection is measured on the accumulators a real stream uses.
    __markStreaming() {
      noteSse();
    },
  };
})();
