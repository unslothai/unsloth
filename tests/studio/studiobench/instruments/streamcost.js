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

  // How much of a decoded chunk is scanned for the relay's framing. The substring search below is
  // the per-event work this file declares as O(1), and an unbounded scan over a bundle, a blob or
  // a paste would make it O(the page's whole TextDecoder traffic) instead.
  //
  // IT BOUNDS THE SCAN AND NOT THE PAYLOAD, and it used to bound the payload: the guard read
  // `out.length <= MAX_SSE_CHUNK_CHARS` and dropped anything longer, on the premise that a decode
  // this large is not relay traffic. That premise is false, and it is false in exactly the case
  // this instrument exists for. A read does not carry one cadence gap of the stream, it carries
  // everything the browser buffered since the last one, so the size of a read is the arrival rate
  // times the length of the stall in front of it. Measured against real chromium reading the real
  // pacer through the app's own `getReader()` loop, the largest read of a stream is 32.5
  // characters per millisecond of stall at fast cadence, dead linear over 500 to 3,000 ms: a
  // 2,000 ms stall lands a 65,000 character read and a 3,000 ms stall a 97,500 character one,
  // which the old guard discarded whole. That payload is well-formed SSE -- 470 `data:` frames of
  // the pacer's own framing -- and it is the single largest task chain of the stream, so the guard
  // took the worst burst out of `sseChunks`, `lastSseAt` and the `deltaTaskMs` numerator at the
  // one moment they matter most. Field cadence arrives at 2.97 characters per millisecond and
  // would need a 22 s stall, so this is reachable on `--cadence fast` and not on the default.
  //
  // HOW MUCH IT COST, and it is not a rounding error. The size of the loss is whatever the app
  // spends per streamed character, because what is dropped is one chain over a third of the reply.
  // Sweeping that rate against the real pacer, with the read loop otherwise identical:
  //
  //   0.0 ms per 1,000 characters   15.7 vs 15.4 ms   -1.3%, which is the noise floor
  //   0.5 ms per 1,000 characters   55.7 vs 104.7 ms   46.8% of the numerator gone
  //   2.0 ms per 1,000 characters   154.0 vs 350.5 ms  56.1%
  //   5.0 ms per 1,000 characters   349.1 vs 835.5 ms  58.2%
  //
  // A harness that does no work per character loses nothing measurable, which is why this hid: it
  // needs an app that actually renders what it streams. It also gets WORSE as the app gets slower,
  // because a slower reader falls further behind and the batch it is then handed is bigger -- the
  // same shape as the timer attribution above, a build getting worse reading cheaper.
  //
  // Scanning a bounded head rather than rejecting keeps both of the things the cap was for. The
  // work stays bounded by a constant that does not grow with the payload or with the rung, and a
  // bundle still has to put `data:` in its first 65,536 characters to be mistaken for a stream --
  // while a real burst, which starts a frame every 279 characters, is found immediately. Measured
  // in v8: 0.04 us on an SSE batch and 0.68 us on a 2 MB blob with no frame in it, against 0.70 us
  // for the same scan left unbounded.
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
    // How many times a frame that was ALREADY BUFFERED when this call began was completed and
    // counted. Cumulative and never reset, for the same reason `wireChars` is not: a window wants
    // the growth across it. This is what turns "a buffer was pending when the window opened" into
    // "characters that arrived before this window were counted inside it", which is the thing that
    // actually makes the denominator wrong. A buffer left behind by an ABORTED response never
    // completes, so it never increments this and never costs the next response its reading.
    carriedFlushes: 0,
  };
  // The incremental SSE buffer. A decode() call is a slice of the socket, not an SSE frame: one
  // call can carry three frames and half of a fourth. Whatever is left after the last blank line
  // stays here until the rest of it arrives.
  //
  // PER DECODER, NOT PER PAGE. A `TextDecoder` belongs to ONE response: the app builds a new one
  // beside its own `buffer` for every streaming request, so no legitimate frame ever spans two of
  // them. A single page-wide buffer therefore did not model reassembly, it modelled the socket,
  // and `stop_generation` exists to cut a socket mid-frame. The abandoned JSON tail then had no
  // `\n\n` to close it, so the next response's first chunk was glued behind it, the merged part
  // failed `startsWith("data:")` and was skipped by `continue` WITHOUT counting a parse failure --
  // the silently short denominator this file's own comments call the unacceptable outcome. Worse,
  // the residue never cleared, so `pending_chars` stayed above zero and `reply_chars_scoreable`
  // refused EVERY window from the abort onwards, and once non-empty it pulled unrelated decoder
  // traffic in through the `pending.length > 0` branch below.
  //
  // Keyed weakly, so a finished response's buffer is collected with its decoder rather than
  // retained. Within one response the same decoder is reused for every `decode(chunk, {stream:
  // true})` call, so a genuine split still reassembles exactly as before.
  const DECODER_STATE = new WeakMap();
  const stateFor = (decoder) => {
    let st = DECODER_STATE.get(decoder);
    if (!st) {
      st = { pending: "", markerTail: "" };
      DECODER_STATE.set(decoder, st);
    }
    return st;
  };
  //: The decoder that most recently delivered a chunk, which is the stream a window is measuring.
  //: `wireIntegrity` reports THIS buffer: half a frame of the response being measured is a short
  //: denominator and must refuse the window, while half a frame of a response that was aborted
  //: three slots ago says nothing about it.
  let active = { pending: "", markerTail: "" };
  //: The decoder holding a speculative marker fragment, if any, and there is only ever one worth
  //: holding. A fragment is at most four characters and lives on the decoder that produced it, but
  //: it has to be REPORTED whoever holds it: a decoder whose first chunk is "dat" has not been
  //: identified as the stream yet, and leaving its fragment out of `wireIntegrity` would be the
  //: instrument saying nothing was outstanding while a frame was. Erring the other way costs at
  //: most four characters of unrelated traffic marking a window unscoreable, which is the
  //: direction this file has always chosen. It is dropped as soon as some OTHER decoder is
  //: identified as the stream, so an unrelated chunk that happened to end in "data" cannot leave a
  //: permanent residue behind it.
  let markerHold = null;
  const setMarkerTail = (st, frag) => {
    st.markerTail = frag;
    if (frag) markerHold = st;
    else if (markerHold === st) markerHold = null;
  };
  const heldMarkerChars = () =>
    active.markerTail.length +
    (markerHold && markerHold !== active ? markerHold.markerTail.length : 0);
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
  // Lives in the per-decoder state above, for the same reason `pending` does.

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
  // Close whatever chain is open and charge it to the accumulator it was opened against. Called
  // from the MessageChannel callback -- the ordinary end of a chain -- and from `read()`, which
  // is the case that used to lose it.
  //
  // WHY read() HAS TO DO THIS. `read()` arrives on its own task, from the driver, at a window
  // boundary; the message closing the last burst's chain is a task too, and the two are on
  // different task queues, so a window can close in the gap between a burst's decode and its
  // chain's macrotask. The snapshot then takes `deltaTaskMs` without that burst in it and
  // `reset()` zeroes the accumulator, so the callback charges the burst to a fresh one -- which
  // the tail `read(0)` in `StreamCostInstrument.close` discards, and which the next `open()`
  // resets in any case. The burst's characters are counted in the denominator and its targeted
  // cost is silently gone from the numerator. Reproduced against real chromium under a synthetic
  // stream of known per-burst cost: 0 to 2 of 200 bursts lost per run, one per window boundary at
  // most, and always downward.
  //
  // The stale message is harmless: it finds `chainStart === null` and returns. A burst that
  // starts after this posts a message of its own.
  const closeChain = () => {
    if (chainStart === null) return;
    S.deltaTaskMs += now() - chainStart;
    chainStart = null;
  };
  const chan = new MessageChannel();
  chan.port1.onmessage = closeChain;

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
  const countDeltaChars = (st, text, carriedMarker) => {
    const carried = st.pending.length > 0 || Boolean(carriedMarker);
    st.pending += text;
    if (st.pending.length > MAX_PENDING_CHARS) {
      S.wireParseFailures += 1;
      st.pending = "";
      return;
    }
    // Frames are separated by a blank line. Anything after the last one is incomplete.
    const parts = st.pending.split("\n\n");
    st.pending = parts.pop();
    // Something that was in the buffer before this call just became a counted frame. A window
    // whose OPEN saw a non-empty buffer is only wrong if this happens inside it.
    if (carried && parts.length > 0) S.carriedFlushes += 1;
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
    if (typeof out === "string" && out.length > 0) {
      // Reassembly state belongs to THIS decoder. Whether this decoder is also the stream being
      // MEASURED is not known yet -- that is decided below, once the chunk has been looked at --
      // and promoting it here handed `active` to any decoder in the page. An unrelated one has an
      // empty buffer, so `wireIntegrity` then reported nothing outstanding while an SSE decoder
      // held half a frame: the window closing there published a denominator short by that frame
      // with a clean bill of health, and the window the suffix landed in was handed the whole
      // frame and accepted it. A response that was aborted mid-frame keeps its half frame to
      // itself.
      const st = stateFor(this);
      // The fragment the previous chunk ended on, but only if THIS chunk continues it. A split
      // inside the marker is repaired here rather than in the buffer, so a fragment that turns out
      // to be ordinary text ending in "d" is dropped instead of corrupting the frame behind it.
      // `markerTail` is only ever set when `pending` is empty, so the two can never both hold a
      // half of the same frame.
      const carriedMarker = Boolean(st.markerTail && continuesMarker(st.markerTail, out));
      const chunk = carriedMarker ? st.markerTail + out : out;
      setMarkerTail(st, "");
      // THE BOUND IS ON THE SCAN, NOT ON THE PAYLOAD. A chunk at or under the cap is its own head,
      // so nothing is allocated on the ordinary path and the ordinary path is unchanged. Over the
      // cap, v8 slices a string by reference rather than by copy, so the head costs no walk of the
      // payload either. Both things the cap was for survive: the search stays bounded by a constant
      // that grows with neither the payload nor the rung, and a bundle, a blob or a paste still has
      // to put the marker in its first MAX_SSE_CHUNK_CHARS characters to be mistaken for a stream.
      // What does NOT survive is the old rejection: a read carries everything the browser buffered
      // since the last one, so a stall past about two seconds at fast cadence hands the app one
      // well-formed 97,500 character batch, and dropping it took the largest burst of the stream
      // out of `sseChunks`, `lastSseAt` and the `deltaTaskMs` numerator at the moment a stall makes
      // it largest.
      const head = chunk.length <= MAX_SSE_CHUNK_CHARS ? chunk : chunk.slice(0, MAX_SSE_CHUNK_CHARS);
      const looksSse = head.indexOf(SSE_MARKER) >= 0;
      // A CONTINUATION IS STREAM TRAFFIC, AND IT STARTS A TASK CHAIN LIKE ANY OTHER CHUNK.
      // `noteSse` was gated on the marker alone while the counter below was gated on
      // `looksSse || pending`, so the tail of a frame the socket cut inside its JSON body was
      // COUNTED in the denominator and charged to nothing. The three quantities it skipped are
      // the three the batched-read note above names: `sseChunks`, `lastSseAt` and the
      // `deltaTaskMs` numerator. When the suffix lands on a later task the first half rendered
      // nothing and its chain has already closed, so the render the suffix does start is charged
      // to no window at all, and `stream_delta_cost_ms_per_kchar` is biased DOWNWARD exactly as
      // fragmentation rises -- the mirror of the denominator defect the comment below describes,
      // and in the flattering direction. A stale `lastSseAt` also lets `replyChars` decide the
      // stream has gone idle and return `null` for a window still carrying it.
      const continuesFrame = st.pending.length > 0;
      if (looksSse || continuesFrame) noteSse();
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
      // The whole chunk is fed to the counter and only the SCAN was bounded above: the denominator
      // is characters delivered, so counting a batched read's head would understate it by exactly
      // the amount a stall made it large.
      // AND ONLY NOW IS THIS DECODER THE ONE A WINDOW IS MEASURING: it either carries the relay's
      // framing or is completing a frame of its own. A decoder that is neither cannot take
      // `active` away from one that is.
      if (looksSse || continuesFrame) {
        // A fragment held by a DIFFERENT decoder cannot be part of this stream, and this one is
        // the stream. Dropped rather than carried, so unrelated traffic that ended in "data"
        // cannot report an outstanding frame for the rest of the cell.
        if (markerHold && markerHold !== st) setMarkerTail(markerHold, "");
        active = st;
        countDeltaChars(st, chunk, carriedMarker);
      } else {
        // A speculative fragment is kept on the decoder that produced it, and it is reported
        // through `wireIntegrity` only if that decoder is the active stream. Unrelated traffic
        // ending in "d" is therefore held without ever claiming to be an outstanding frame.
        setMarkerTail(st, partialMarkerTail(chunk));
      }
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
      // BEFORE the snapshot, because a chain still in flight belongs to the window that started
      // it and `reset()` below is about to throw it away. See `closeChain`.
      closeChain();
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
      return {
        failures: S.wireParseFailures,
        pending_chars: active.pending.length + heldMarkerChars(),
        // Read as a DELTA across the window by `StreamCostInstrument.close`, so a buffer that was
        // pending at the open refuses the window only when its frame was completed inside it.
        carried_flushes: S.carriedFlushes,
      };
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
    // synthetic injection can be measured on exactly the accumulators a real stream uses.
    __markStreaming() {
      noteSse();
    },
  };
})();
