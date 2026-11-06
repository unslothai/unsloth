// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// Streaming-phase cost accumulator, detected from SSE traffic since window kinds cannot separate it.
// Three accumulators: deltaTaskMs (task chains SSE chunks start, first decode to next macrotask),
// blockedMs (frames.js-style blocked time while streaming), streamingMs (time the stream ran).
// Chain end timed with MessageChannel: nested setTimeout(0) clamps to 4 ms.

(() => {
  if (window.__sb && window.__sb.streamcost) return;
  window.__sb = window.__sb || {};

  // About twenty cadence gaps: long enough not to chop a jam, short enough to exclude post-stream.
  const IDLE_GAP_MS = 1500;

  // Bounds the scan, not the payload: a stall can deliver a large valid batched read.
  const MAX_SSE_CHUNK_CHARS = 65536;

  const S = {
    sseChunks: 0,
    sseBursts: 0,
    deltaTaskMs: 0,
    blockedMs: 0,
    streamingMs: 0,
    lastSseAt: 0,
    // Measured own cost, checked by the overhead_growth_with_length gate.
    overheadMs: 0,
    decodeCalls: 0,
    everStreamed: false,
    // Cumulative, never reset by reset(): windows take the difference of two readings.
    wireChars: 0,
    wireFrames: 0,
    wireParseFailures: 0,
  };
  // Per decoder, not per page: an aborted response's partial frame must not glue onto the next.
  const DECODER_STATE = new WeakMap();
  // Per decoder, cumulative count of completions of frames buffered before the decode began.
  const CARRIED_BY_ID = new Map();
  // close() may ask about a decoder no longer alive, so history lives outside the WeakMap.
  const MAX_DECODER_HISTORY = 64;
  let DECODER_SEQ = 0;
  const noteCarried = (st) => {
    st.carriedFlushes += 1;
    CARRIED_BY_ID.set(st.id, st.carriedFlushes);
    while (CARRIED_BY_ID.size > MAX_DECODER_HISTORY) {
      CARRIED_BY_ID.delete(CARRIED_BY_ID.keys().next().value);
    }
  };
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
  // The decoder that last delivered SSE, i.e. the measured stream. Built by newState() so a window
  // opening before the first stream reads zero, not undefined.
  let active = newState();
  // Marker fragment holder; reported even before its decoder is identified as the stream.
  let markerHold = null;
  const setMarkerTail = (st, frag) => {
    st.markerTail = frag;
    if (frag) markerHold = st;
    else if (markerHold === st) markerHold = null;
  };
  const heldMarkerChars = () =>
    active.markerTail.length +
    (markerHold && markerHold !== active ? markerHold.markerTail.length : 0);
  // Drop the buffer past this bound rather than grow without limit.
  const MAX_PENDING_CHARS = 262144;

  // The socket can split this marker across two decode() calls.
  const SSE_MARKER = "data:";

  const partialMarkerTail = (s) => {
    for (let n = Math.min(SSE_MARKER.length - 1, s.length); n > 0; n -= 1) {
      if (s.endsWith(SSE_MARKER.slice(0, n))) return SSE_MARKER.slice(0, n);
    }
    return "";
  };

  // Without this, traffic ending in "d" would glue onto the next chunk as "ddata:".
  const continuesMarker = (frag, s) => {
    const rest = SSE_MARKER.slice(frag.length);
    const n = Math.min(rest.length, s.length);
    return n > 0 && s.slice(0, n) === rest.slice(0, n);
  };

  const now = () => performance.now();
  // Takes the instant so a caller can ask about the start of an interval.
  const streamingAt = (t) => S.lastSseAt > 0 && t - S.lastSseAt < IDLE_GAP_MS;

  // One pending chain: a burst delivered in one task must be charged once.
  let chainStart = null;
  // Also called from read(), which can run between a burst's decode and its chain macrotask.
  const closeChain = () => {
    if (chainStart === null) return;
    S.deltaTaskMs += now() - chainStart;
    chainStart = null;
  };
  const chan = new MessageChannel();
  chan.port1.onmessage = closeChain;

  // Counted off the wire, not a DOM read whose O(document) cost biased the virtualised arm.
  const countDeltaChars = (st, text, carriedMarker) => {
    const carried = st.pending.length > 0 || Boolean(carriedMarker);
    st.pending += text;
    if (st.pending.length > MAX_PENDING_CHARS) {
      S.wireParseFailures += 1;
      st.pending = "";
      return;
    }
    const parts = st.pending.split("\n\n");
    st.pending = parts.pop();
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
        // Reasoning arrives as reasoning_content beside empty content, so summing is not double counting.
        const content = typeof delta.content === "string" ? delta.content.length : 0;
        const reasoning =
          typeof delta.reasoning_content === "string" ? delta.reasoning_content.length : 0;
        S.wireChars += content + reasoning;
        S.wireFrames += 1;
      } catch (err) {
        // Counted, not swallowed: a short denominator inflates every cost per character.
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

  // decode() is the first main-thread code to see a chunk: O(1) in thread size.
  const nativeDecode = TextDecoder.prototype.decode;
  TextDecoder.prototype.decode = function (input, options) {
    const out = nativeDecode.call(this, input, options);
    const t = now();
    S.decodeCalls += 1;
    if (typeof out === "string" && out.length > 0) {
      // Only promoted to active below, once the chunk is known to be SSE.
      const st = stateFor(this);
      // Repair a split marker only if this chunk continues it; markerTail implies pending is empty.
      const carriedMarker = Boolean(st.markerTail && continuesMarker(st.markerTail, out));
      const chunk = carriedMarker ? st.markerTail + out : out;
      setMarkerTail(st, "");
      const head = chunk.length <= MAX_SSE_CHUNK_CHARS ? chunk : chunk.slice(0, MAX_SSE_CHUNK_CHARS);
      const looksSse = head.indexOf(SSE_MARKER) >= 0;
      // A frame continuation is stream traffic and starts a task chain like any other chunk.
      const continuesFrame = st.pending.length > 0;
      if (looksSse || continuesFrame) noteSse();
      // looksSse || pending: the second half of a split frame has no marker.
      // Feed the whole chunk (only the scan is bounded); only now does this decoder become active.
      if (looksSse || continuesFrame) {
        // A fragment on a different decoder cannot be part of this stream.
        if (markerHold && markerHold !== st) setMarkerTail(markerHold, "");
        active = st;
        countDeltaChars(st, chunk, carriedMarker);
      } else {
        setMarkerTail(st, partialMarkerTail(chunk));
      }
    }
    S.overheadMs += now() - t;
    return out;
  };

  // Uses frames.js's calibrated clamp so both instruments subtract the same idle floor.
  let lastTick = now();
  const tick = () => {
    const t = now();
    const gap = t - lastTick;
    // Attribute by state at interval START: a stall is only seen after it ends.
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

  // Last assistant message only; skipped once idle past the gap since the read is O(DOM).
  const replyChars = (force) => {
    if (!force && S.lastSseAt > 0 && now() - S.lastSseAt >= IDLE_GAP_MS) return null;
    const all = document.querySelectorAll('[data-role="assistant"]');
    if (all.length === 0) return null;
    const el = all[all.length - 1];
    return (el.textContent || "").length;
  };

  window.__sb.streamcost = {
    read(elapsedMs) {
      const t = now();
      // Before the snapshot: an in-flight chain belongs to the window that started it.
      closeChain();
      const f = window.__sb.frames;
      const clampInfo = f && f.clamp ? f.clamp() : { clampMs: null, reason: "frames.js absent" };
      const out = {
        sse_chunks: S.sseChunks,
        sse_chunks_attempted: true,
        sse_bursts: S.sseBursts,
        decode_calls: S.decodeCalls,
        // Never a bare zero: no-stream windows are flagged and skipped by scoring.
        streaming_observed: S.sseChunks > 0,
        streaming_ms: Math.round(S.streamingMs * 10) / 10,
        streaming_ms_attempted: true,
        delta_task_ms: Math.round(S.deltaTaskMs * 10) / 10,
        delta_task_ms_attempted: true,
        driver_elapsed_ms: elapsedMs === null || elapsedMs === undefined ? null : elapsedMs,
        clamp_ms: clampInfo.clampMs === null ? null : clampInfo.clampMs,
      };
      if (clampInfo.clampMs === null || clampInfo.clampMs === undefined) {
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

    // forId names the decoder: buffer and flush must belong to the same decoder.
    wireIntegrity(forId) {
      // A marker fragment counts as buffered; erring toward unscoreable is the right way.
      return {
        failures: S.wireParseFailures,
        pending_chars: active.pending.length + heldMarkerChars(),
        decoder_id: active.id,
        // Read as a delta across the window by StreamCostInstrument.close.
        carried_flushes: carriedFor(forId),
      };
    },
    replyChars() {
      return S.wireChars;
    },

    // Old DOM reading kept as a cross-check, never called inside a measured window.
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

    __markStreaming() {
      noteSse();
    },
  };
})();
