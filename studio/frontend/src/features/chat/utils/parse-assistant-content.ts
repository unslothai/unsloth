// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelRunResult } from "@assistant-ui/react";

type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";
/**
 * Invisible separator so literal think tags in reasoning do not close the panel
 * (#7066). U+2060 WORD JOINER, not U+200B ZERO WIDTH SPACE: U+200B is line-break
 * class ZW, so a neutralized tag could wrap mid-tag; WJ forbids that break.
 */
const THINK_NEUTRAL_ZW = "\u2060";

/**
 * Normalize streamed string or structured delta content to inline text.
 * Structured reasoning-only chunks remain distinguishable so their fallback
 * timer can span consecutive chunks even though each chunk carries closed tags.
 *
 * `closeOffsets` reports where each wrapper `</think>` starts so the caller can
 * register it as a known boundary. They are ours, not model markers: left to the
 * raw-marker heuristics they kept answer deltas in the drawer (#7334).
 */
export function extractDeltaText(delta: unknown): {
  text: string;
  structuredReasoningContinues: boolean;
  closeOffsets: number[];
} {
  const extractReasoningText = (payload: unknown): string => {
    if (typeof payload === "string") return payload;
    if (Array.isArray(payload)) {
      return payload.map((item) => extractReasoningText(item)).join("");
    }
    if (!payload || typeof payload !== "object") return "";

    const obj = payload as Record<string, unknown>;
    for (const key of ["thinking", "text", "content", "reasoning", "summary"]) {
      if (key in obj) {
        const text = extractReasoningText(obj[key]);
        if (text) return text;
      }
    }
    return "";
  };

  if (typeof delta === "string") {
    return { text: delta, structuredReasoningContinues: false, closeOffsets: [] };
  }
  if (!Array.isArray(delta)) {
    return { text: "", structuredReasoningContinues: false, closeOffsets: [] };
  }

  let text = "";
  let structuredReasoningContinues = false;
  const closeOffsets: number[] = [];
  for (const part of delta) {
    if (typeof part === "string") {
      text += part;
      if (part) {
        structuredReasoningContinues = false;
      }
      continue;
    }
    if (!part || typeof part !== "object") continue;
    const obj = part as {
      type?: string;
      text?: string;
      content?: string;
      thinking?: string;
    };
    if (obj.type === "text" || obj.type === "output_text") {
      const visibleText =
        typeof obj.text === "string"
          ? obj.text
          : typeof obj.content === "string"
            ? obj.content
            : "";
      text += visibleText;
      if (visibleText) {
        structuredReasoningContinues = false;
      }
    } else if (obj.type === "thinking" || obj.type === "reasoning") {
      const thinking = extractReasoningText(obj);
      // A literal </think> here must not close the wrapper early (#7066).
      if (thinking) {
        text += `${THINK_OPEN_TAG}${neutralizeThinkMarkup(thinking)}`;
        closeOffsets.push(text.length);
        text += THINK_CLOSE_TAG;
        structuredReasoningContinues = true;
      }
    }
  }
  return { text, structuredReasoningContinues, closeOffsets };
}

// ContentPart from @assistant-ui/react has readonly fields, so coalescing via
// `last.text += text` fails (TS2540). Instead replace the last element with a
// fresh merged object: same allocation cost as mutation but type-safe.

function appendTextPart(parts: ContentPart[], text: string): void {
  if (!text) return;
  const last = parts.at(-1);
  if (last?.type === "text") {
    parts[parts.length - 1] = { type: "text", text: last.text + text };
    return;
  }
  parts.push({ type: "text", text });
}

function appendReasoningPart(parts: ContentPart[], text: string): void {
  if (!text) return;
  const last = parts.at(-1);
  if (last?.type === "reasoning") {
    parts[parts.length - 1] = { type: "reasoning", text: last.text + text };
    return;
  }
  parts.push({ type: "reasoning", text });
}

/** Neutralize `<think>` / `</think>` in free text so a literal close tag in
 * reasoning cannot end the thinking block early (#7066). */
export function neutralizeThinkMarkup(text: string): string {
  if (!text) return text;
  if (!text.includes(THINK_OPEN_TAG) && !text.includes(THINK_CLOSE_TAG)) {
    return text;
  }
  return text
    .replaceAll(THINK_CLOSE_TAG, `</${THINK_NEUTRAL_ZW}think>`)
    .replaceAll(THINK_OPEN_TAG, `<${THINK_NEUTRAL_ZW}think>`);
}

/** Trailing chars that may be a prefix of a think marker (split-chunk safe). */
export function thinkMarkupHoldback(text: string): number {
  const markers = [THINK_CLOSE_TAG, THINK_OPEN_TAG];
  const maxLen = Math.max(...markers.map((marker) => marker.length));
  for (let size = Math.min(text.length, maxLen - 1); size > 0; size -= 1) {
    const suffix = text.slice(-size);
    if (markers.some((marker) => marker.startsWith(suffix))) {
      return size;
    }
  }
  return 0;
}

/** Neutralize complete think markers in a streaming buffer (#7066). */
export function drainThinkMarkupBuffer(
  buffer: string,
  options?: { finalize?: boolean },
): { emit: string; buffer: string } {
  if (!buffer) return { emit: "", buffer: "" };
  if (options?.finalize) {
    return { emit: neutralizeThinkMarkup(buffer), buffer: "" };
  }
  const keep = thinkMarkupHoldback(buffer);
  if (keep === buffer.length) return { emit: "", buffer };
  const rawEmit = keep ? buffer.slice(0, -keep) : buffer;
  return {
    emit: neutralizeThinkMarkup(rawEmit),
    buffer: keep ? buffer.slice(-keep) : "",
  };
}

/** Letters and digits, so "l'annee" and "It's" read as one word. */
const WORD_CHAR = /[\p{L}\p{N}]/u;

// Indexing a JS string yields UTF-16 code units, so a non-BMP letter reads as a
// lone surrogate that `\p{L}` misses. The backend indexes by CODE POINT, so
// "𝑥'𝑥" was intra-word there and a delimiter here, flipping the quote parity
// and leaking the rest of a quoted thought into the answer (#7334).

/** The whole code point ending just before `end`, or "" at the start. */
const codePointBefore = (text: string, end: number): string => {
  if (end <= 0) return "";
  const low = text.charCodeAt(end - 1);
  if (low >= 0xdc00 && low <= 0xdfff && end >= 2) {
    const high = text.charCodeAt(end - 2);
    if (high >= 0xd800 && high <= 0xdbff) return text.slice(end - 2, end);
  }
  return text[end - 1] ?? "";
};

/** The whole code point starting at `at`, or "" past the end. */
const codePointAt = (text: string, at: number): string => {
  const point = at >= 0 ? text.codePointAt(at) : undefined;
  return point === undefined ? "" : String.fromCodePoint(point);
};

const isIntraWordApostrophe = (text: string, at: number): boolean =>
  at > 0 &&
  WORD_CHAR.test(codePointBefore(text, at)) &&
  // An apostrophe is a single code unit, so the next code point starts at at+1.
  WORD_CHAR.test(codePointAt(text, at + 1));

/** A quote behind an odd backslash run sits inside a string literal. */
const isEscaped = (text: string, at: number): boolean => {
  let run = 0;
  for (let j = at - 1; j >= 0 && text[j] === "\\"; j -= 1) run += 1;
  return run % 2 === 1;
};

export type ParseOptions = {
  /** The response is still streaming, so `raw` can still grow. */
  streaming?: boolean;
  /** True for a `</think>` the caller inserted itself, at that index. */
  isKnownClose?: (index: number) => boolean;
  /**
   * Mid-stream only: a close tag at that index whose fence decision was
   * deferred. It may still be literal, so act on it only once the final parse
   * confirms it; recording when it arrived lets the reasoning timer stop there
   * rather than at end of stream (#7334). Reported once per index.
   */
  onDeferredClose?: (index: number) => void;
  /**
   * Scratch letting the scan resume where the previous delta left off, so a
   * streaming parse costs O(new text) instead of O(buffer) (#7334).
   *
   * Reuse a cache only while `raw` is APPEND-ONLY apart from truncation at the
   * end, the one shape the scan can verify; text rewritten in place would
   * resume from a stale boundary. Omitting it is always correct, just O(buffer).
   */
  resume?: ScanResumeCache;
};

const FENCE = "```";
/** `nextFence` sentinel: the next marker has not been looked up yet. */
const FENCE_UNSCANNED = -2;

/**
 * Monotone cursors summarizing the prefix a previous scan inspected. The parser
 * re-runs over the whole buffer on every SSE delta, so restarting at `spanStart`
 * re-walked the same fences and quotes: O(n^2) over reasoning that repeatedly
 * quotes `</think>` (#7334).
 */
type ScanResume = {
  /** Length of the buffer this state was built from. */
  rawLen: number;
  /** Use stamp; the lowest is evicted when the table is full. */
  used: number;
  spanStart: number;
  from: number;
  streaming: boolean;
  isKnownClose: ((index: number) => boolean) | undefined;
  onDeferredClose: ((index: number) => void) | undefined;
  /** Where the next candidate search starts; earlier verdicts are settled. */
  resumeFrom: number;
  fences: number;
  nextFence: number;
  fenceFrom: number;
  dq: number;
  dqFrom: number;
  sq: number;
  sqFrom: number;
  bt: number;
  btFrom: number;
};

/** Opaque per-stream scratch; see `ParseOptions.resume`. */
export type ScanResumeCache = { slots: ScanResume[] };

/** Scratch for one append-only buffer. */
export function createScanResumeCache(): ScanResumeCache {
  return { slots: [] };
}

// One slot per (call site, reasoning span): every delta is parsed, polled and
// rebuilt into content parts, each with its own options.
const RESUME_SLOTS = 8;
let resumeClock = 0;

function resetResume(slot: ScanResume): void {
  slot.rawLen = 0;
  slot.resumeFrom = slot.from;
  slot.fences = 0;
  slot.nextFence = FENCE_UNSCANNED;
  slot.fenceFrom = slot.spanStart;
  slot.dq = 0;
  slot.dqFrom = slot.spanStart;
  slot.sq = 0;
  slot.sqFrom = slot.spanStart;
  slot.bt = 0;
  slot.btFrom = slot.spanStart;
}

/**
 * Slot for this call's options, least recently used evicted. Callbacks match by
 * identity, so a caller minting a fresh arrow per delta just starts cold, as
 * does a call with no cache. Eviction and a cold start only cost a rescan.
 */
function resumeSlotFor(
  cache: ScanResumeCache | undefined,
  spanStart: number,
  from: number,
  streaming: boolean,
  isKnownClose: ((index: number) => boolean) | undefined,
  onDeferredClose: ((index: number) => void) | undefined,
): ScanResume {
  resumeClock += 1;
  const slots = cache?.slots;
  if (slots) {
    for (const slot of slots) {
      if (
        slot.spanStart === spanStart &&
        slot.from === from &&
        slot.streaming === streaming &&
        slot.isKnownClose === isKnownClose &&
        slot.onDeferredClose === onDeferredClose
      ) {
        slot.used = resumeClock;
        return slot;
      }
    }
  }
  const slot: ScanResume = {
    rawLen: 0,
    used: resumeClock,
    spanStart,
    from,
    streaming,
    isKnownClose,
    onDeferredClose,
    resumeFrom: from,
    fences: 0,
    nextFence: FENCE_UNSCANNED,
    fenceFrom: spanStart,
    dq: 0,
    dqFrom: spanStart,
    sq: 0,
    sqFrom: spanStart,
    bt: 0,
    btFrom: spanStart,
  };
  if (slots) {
    if (slots.length < RESUME_SLOTS) {
      slots.push(slot);
    } else {
      let lru = 0;
      for (let i = 1; i < slots.length; i += 1) {
        if (slots[i].used < slots[lru].used) lru = i;
      }
      slots[lru] = slot;
    }
  }
  return slot;
}

/**
 * First structural (non-quoted, non-fenced) close tag at or after `from`.
 *
 * A close tag is *literal* content rather than a block end (#7066) when it sits
 * inside a ``` fence that actually closes, or when it is flanked by quote chars
 * whose leading quote OPENS a span (odd count of that char since `spanStart`).
 *
 * One forward pass: the fence count, quote counts and "is there a later fence"
 * answer carry across candidates, so a call costs O(raw.length) even with many
 * literal `"</think>"` mentions, and across deltas the pass resumes from
 * `ScanResume` so a delta costs O(added text). Only a verdict the inspected
 * prefix settles may be resumed past; a tag whose trailing flank sits at the
 * buffer edge, or whose fenced verdict reads to end of stream, is re-examined
 * every delta. `onDeferredClose` therefore fires when the scan first reaches a
 * candidate, which is the report callers time the thought from.
 *
 * `streaming` marks a mid-stream parse, where an enclosing ``` fence may still
 * close later, so the unclosed-fence fallback is deferred to the final parse.
 * `isKnownClose` reports delimiters the caller inserted itself (closing a
 * synthetic `reasoning_content` wrapper); those are authoritative, so the
 * heuristics below, which only interpret RAW model markers, must not apply.
 */
function findStructuralThinkClose(
  raw: string,
  spanStart: number,
  from: number,
  streaming = false,
  isKnownClose?: (index: number) => boolean,
  onDeferredClose?: (index: number) => void,
  resume?: ScanResumeCache,
): number {
  const slot = resumeSlotFor(
    resume,
    spanStart,
    from,
    streaming,
    isKnownClose,
    onDeferredClose,
  );
  // The cache only promises an append-only buffer, so a shorter one was
  // truncated and nothing inspected past its end holds. Comparing the text
  // instead would cost the O(buffer) per delta this exists to avoid.
  if (raw.length < slot.rawLen) resetResume(slot);

  // Greedy non-overlapping fence scan (matches Python str.count): `fences`
  // counts markers starting strictly before `nextFence`, looked up from
  // `fenceFrom` on first use so a span with no candidate never pays for it.
  let fences = slot.fences;
  let nextFence = slot.nextFence;
  let fenceFrom = slot.fenceFrom;
  // Last fence marker in `raw`; only the odd-parity branch needs it.
  let lastFence: number | undefined;
  // Running quote counts over [spanStart, cursor) per char, advanced lazily
  // with indexOf rather than a char loop (same answer, far less work on prose).
  let dq = slot.dq;
  let dqFrom = slot.dqFrom;
  let sq = slot.sq;
  let sqFrom = slot.sqFrom;
  let bt = slot.bt;
  let btFrom = slot.btFrom;
  const quoteCount = (ch: string, end: number): number => {
    let n = ch === '"' ? dq : ch === "'" ? sq : bt;
    const cursor = ch === '"' ? dqFrom : ch === "'" ? sqFrom : btFrom;
    for (let at = raw.indexOf(ch, cursor); at !== -1 && at < end; ) {
      // Not delimiters: an apostrophe inside a word ("It's"), and a quote
      // escaped by an odd backslash run, which sits inside a string literal.
      // Counting either flipped the parity of a quoted tag (#7334).
      if (
        (ch !== "'" || !isIntraWordApostrophe(raw, at)) &&
        !isEscaped(raw, at)
      ) {
        n += 1;
      }
      at = raw.indexOf(ch, at + 1);
    }
    if (ch === '"') {
      dq = n;
      dqFrom = end;
    } else if (ch === "'") {
      sq = n;
      sqFrom = end;
    } else {
      bt = n;
      btFrom = end;
    }
    return n;
  };
  // Memoized "is there a close tag at or after `at`" so the fence look-ahead
  // below stays amortized O(raw.length) instead of one indexOf per candidate
  // (#7334). Monotone: none found from an offset means none from a later one.
  let seekFrom = -1;
  let seekHit = -1;
  const hasCloseTagFrom = (at: number): boolean => {
    if (seekFrom !== -1) {
      if (seekHit >= at) return true;
      if (seekHit === -1 && at >= seekFrom) return false;
    }
    seekFrom = at;
    seekHit = raw.indexOf(THINK_CLOSE_TAG, at);
    return seekHit !== -1;
  };

  let searchFrom = slot.resumeFrom;
  let closeIndex = raw.indexOf(THINK_CLOSE_TAG, searchFrom);
  // Cleared once a verdict rests on text that has not arrived, so the resume
  // point never passes a tag a later delta could reclassify.
  let resumable = true;
  // First structural close, or -1. Never cached: a tag at the very end reads as
  // unflanked now and may read as quoted next delta.
  let structural = -1;

  while (closeIndex !== -1) {
    if (nextFence === FENCE_UNSCANNED) {
      nextFence = raw.indexOf(FENCE, fenceFrom);
    }
    while (nextFence !== -1 && nextFence < closeIndex) {
      fences += 1;
      fenceFrom = nextFence + FENCE.length;
      nextFence = raw.indexOf(FENCE, fenceFrom);
    }

    let literal: boolean;
    if (isKnownClose?.(closeIndex)) {
      // Our own delimiter: the boundary is already known, not inferred.
      literal = false;
    } else if (fences % 2 === 1) {
      // The close sits inside an open ``` fence. Global parity over the span is
      // wrong: a separate later unclosed fence would misflag an earlier close
      // whose own fence already closed (#7334).
      if (streaming) {
        // "Not closed yet" is not "never closes", so defer like the backend
        // extractor: calling it structural and reversing it later would bounce
        // text out of the drawer and latch reasoningDuration on a tag that was
        // never the close (#7334). Report it so the caller can timestamp it and
        // use that instant only if the final parse agrees.
        onDeferredClose?.(closeIndex);
        literal = true;
      } else {
        // The look-ahead below reads to the end of `raw`, so the prefix does
        // not settle this verdict and cannot be resumed past.
        resumable = false;
        // Where the enclosing fence would close: the greedy cursor answers
        // directly, falling back to the O(n) scan when exhausted, since
        // overlapping runs such as "````" can hide a marker from it.
        let fenceClose = nextFence;
        if (fenceClose === -1) {
          if (lastFence === undefined) lastFence = raw.lastIndexOf(FENCE);
          if (lastFence >= closeIndex) fenceClose = lastFence;
        }
        // No closing ``` at all means the fence never closes, so this tag is
        // the genuine close: an unclosed fence must not swallow the answer. A
        // ``` that does follow only proves the reasoning-side fence closed when
        // reasoning continues past it to a further close tag; otherwise that
        // marker opens a fenced block in the ANSWER, which hid the answer for
        // "draft ```</think>Answer: ```js ... ```" (#7334). Mirrors the backend
        // extractor's _fence_unresolved_at_close.
        literal =
          fenceClose !== -1 && hasCloseTagFrom(fenceClose + FENCE.length);
      }
    } else {
      const before = closeIndex > spanStart ? raw[closeIndex - 1] : "";
      const closeEnd = closeIndex + THINK_CLOSE_TAG.length;
      // Skip an escaping backslash so a mention quoted inside a string literal
      // ( \"</think>\" ) still reads as symmetrically quoted (#7334).
      const after =
        (raw[closeEnd] === "\\" ? raw[closeEnd + 1] : raw[closeEnd]) ?? "";
      // A quoted mention is symmetric: accepting ANY two delimiters called
      // "`</think>\"yes\"" quoted and hid the whole answer in the drawer (#7334).
      if (streaming && !after && before && `"'\``.includes(before)) {
        // Mid-stream an ABSENT trailing flank is not an empty one: providers
        // emit `</think>` as one token, so `<think>echo "</think>` ends on the
        // tag and its closing quote lands in the NEXT delta. Reading the gap as
        // "not quoted" calls the mention structural for one delta, and
        // chat-adapter latches reasoningDuration on that instant and never
        // lowers it (#7334). Defer like the fence branch above (backend:
        // _should_hold_quoted_think_close); the next delta settles this tag, so
        // nothing may resume past it.
        onDeferredClose?.(closeIndex);
        resumable = false;
        literal = true;
      } else if (!before || before !== after || !`"'\``.includes(before)) {
        literal = false;
      } else {
        // A prose mention closes its quote and reads on as prose, so a closing
        // quote running straight into a word char is the ANSWER's own opening
        // quote and the tag WAS the close; reading it as a mention hid the whole
        // answer for '"</think>"The answer is 42.' (#7334). Mirrors the
        // backend's _quoted_close_opens_answer.
        const quoteAt = raw[closeEnd] === "\\" ? closeEnd + 1 : closeEnd;
        // A quoted mention pairs delimiter RUNS of EQUAL length: CommonMark
        // closes a code span with "a backtick string of equal length", so
        // "`</think>```python" pairs a 1-run against a 3-run and is no span at
        // all -- that ``` opens the ANSWER's fence and the tag was the close.
        // Raw-char parity alone cannot decide this: well-formed markdown
        // reaches an ODD backtick count via a nested span (``a ` b``) or a
        // closing fence longer than its opener, both legal (#7334).
        let runBefore = 0;
        for (let i = closeIndex - 1; i >= spanStart && raw[i] === before; i -= 1) {
          runBefore += 1;
        }
        let runAfter = 0;
        for (let i = quoteAt; i < raw.length && raw[i] === before; i += 1) {
          runAfter += 1;
        }
        // The deciding char sits after the WHOLE trailing run, and the next
        // delta may still supply either, so reading one as absent flips the
        // verdict: nothing may resume past this tag until they land (#7334).
        if (quoteAt + runAfter >= raw.length) resumable = false;
        // Literal only when the leading quote OPENS a span: an odd count of
        // that char since the reasoning start.
        literal =
          runBefore === runAfter &&
          !WORD_CHAR.test(codePointAt(raw, quoteAt + runAfter)) &&
          quoteCount(before, closeIndex) % 2 === 1;
      }
    }

    if (!literal) {
      structural = closeIndex;
      break;
    }
    searchFrom = closeIndex + THINK_CLOSE_TAG.length;
    // Settled only once the trailing flank (the char after the tag, or after
    // its escaping backslash) AND the char after it are inside the inspected
    // text: the latter separates a mention from an answer opening with a
    // quote, so a verdict without it can still change (#7334).
    const flankEnd = raw[searchFrom] === "\\" ? searchFrom + 2 : searchFrom + 1;
    if (resumable && flankEnd < raw.length) {
      slot.resumeFrom = searchFrom;
      slot.fences = fences;
      // A -1 lookup only proves there is no marker before the last 2 chars,
      // where the next delta could complete one.
      slot.nextFence = nextFence === -1 ? FENCE_UNSCANNED : nextFence;
      slot.fenceFrom =
        nextFence === -1
          ? Math.max(fenceFrom, raw.length - (FENCE.length - 1))
          : fenceFrom;
      slot.dq = dq;
      slot.dqFrom = dqFrom;
      slot.sq = sq;
      slot.sqFrom = sqFrom;
      slot.bt = bt;
      slot.btFrom = btFrom;
    }
    closeIndex = raw.indexOf(THINK_CLOSE_TAG, searchFrom);
  }

  if (structural === -1 && resumable) {
    // No tag starts in the text just searched, so the next delta re-reads only
    // the tail one straddling the end could start in. Leaving the fence and
    // quote cursors behind is safe: they catch up on the next candidate.
    const tail = raw.length - (THINK_CLOSE_TAG.length - 1);
    if (searchFrom > slot.resumeFrom) slot.resumeFrom = searchFrom;
    if (tail > slot.resumeFrom) slot.resumeFrom = tail;
  }
  slot.rawLen = raw.length;
  return structural;
}

/**
 * Split raw assistant text into reasoning / text parts.
 *
 * Pass `{ streaming: true }` while the response is still arriving so an
 * as-yet-unclosed ``` fence is not resolved early; the default (stream
 * complete) applies the structural fallback (#7334).
 */
export function parseAssistantContent(
  raw: string,
  options?: ParseOptions,
): ContentPart[] {
  const parts: ContentPart[] = [];
  if (!raw) {
    return parts;
  }
  const streaming = options?.streaming ?? false;

  let cursor = 0;
  while (cursor < raw.length) {
    const openIndex = raw.indexOf(THINK_OPEN_TAG, cursor);
    if (openIndex === -1) {
      appendTextPart(parts, raw.slice(cursor));
      break;
    }

    appendTextPart(parts, raw.slice(cursor, openIndex));

    const reasoningStart = openIndex + THINK_OPEN_TAG.length;
    const closeIndex = findStructuralThinkClose(
      raw,
      reasoningStart,
      reasoningStart,
      streaming,
      options?.isKnownClose,
      options?.onDeferredClose,
      options?.resume,
    );
    if (closeIndex === -1) {
      appendReasoningPart(parts, raw.slice(reasoningStart));
      break;
    }

    appendReasoningPart(parts, raw.slice(reasoningStart, closeIndex));
    cursor = closeIndex + THINK_CLOSE_TAG.length;
  }

  return parts;
}

/**
 * True once the reasoning block has *structurally* closed, using the same
 * literal classification as `parseAssistantContent`. A raw substring check would
 * latch the reasoning-duration timer on a literal `</think>` and never correct
 * it when the real close arrives (#7334). Callers polling mid-stream must pass
 * `{ streaming: true }`: a tag inside a fence not closed *yet* is not a close.
 */
export function hasClosedThinkTag(
  raw: string,
  options?: ParseOptions,
): boolean {
  return structuralThinkCloseIndex(raw, options) !== -1;
}

/**
 * Index of the structural close tag ending the first reasoning block, or -1.
 * Same classification as `hasClosedThinkTag`; callers that recorded deferred
 * candidates mid-stream match the confirmed index against them (#7334).
 */
export function structuralThinkCloseIndex(
  raw: string,
  options?: ParseOptions,
): number {
  const openIndex = raw.indexOf(THINK_OPEN_TAG);
  const spanStart = openIndex === -1 ? 0 : openIndex + THINK_OPEN_TAG.length;
  return findStructuralThinkClose(
    raw,
    spanStart,
    spanStart,
    options?.streaming ?? false,
    options?.isKnownClose,
    options?.onDeferredClose,
    options?.resume,
  );
}

/**
 * Structural close of the LAST reasoning block, or -1 when the block is still
 * open (or there is none). Walks every block, so the reasoning timer reads the
 * close that ended the group it is still timing (#7334).
 */
export function lastStructuralThinkCloseIndex(
  raw: string,
  options?: ParseOptions,
): number {
  let cursor = 0;
  let lastClose = -1;
  for (;;) {
    const openIndex = raw.indexOf(THINK_OPEN_TAG, cursor);
    if (openIndex === -1) {
      return lastClose;
    }
    const spanStart = openIndex + THINK_OPEN_TAG.length;
    const closeIndex = findStructuralThinkClose(
      raw,
      spanStart,
      spanStart,
      options?.streaming ?? false,
      options?.isKnownClose,
      options?.onDeferredClose,
      options?.resume,
    );
    if (closeIndex === -1) {
      return -1;
    }
    lastClose = closeIndex;
    cursor = closeIndex + THINK_CLOSE_TAG.length;
  }
}

/**
 * True while a reasoning block is open, i.e. the last `<think>` has no close
 * after it. Structural, not `lastIndexOf`: a literal `</think>` quoted inside
 * the thought would otherwise read as the block end and stop the timer (#7066).
 */
export function hasUnclosedThinkTag(
  raw: string,
  options?: ParseOptions,
): boolean {
  return (
    raw.includes(THINK_OPEN_TAG) &&
    lastStructuralThinkCloseIndex(raw, options) === -1
  );
}
