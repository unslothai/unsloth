// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelRunResult } from "@assistant-ui/react";

type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";
/** Invisible joiner so literal think tags in reasoning text do not close the panel (#7066). */
const THINK_NEUTRAL_ZW = "\u200b";

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

/**
 * Neutralize structural `<think>` / `</think>` markers inside free text so a
 * literal close tag in reasoning (or a user quote) cannot prematurely end the
 * thinking block (#7066).
 */
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

export type ParseOptions = {
  /** The response is still streaming, so `raw` can still grow. */
  streaming?: boolean;
  /** True for a `</think>` the caller inserted itself, at that index. */
  isKnownClose?: (index: number) => boolean;
};

/**
 * First structural (non-quoted, non-fenced) close tag at or after `from`.
 *
 * A close tag is *literal* content rather than a block end (#7066) when it sits
 * inside a ``` fence that actually closes, or when it is flanked by quote chars
 * whose leading quote OPENS a span (odd count of that char since `spanStart`).
 *
 * One forward pass: the fence count, the quote counts and the "is there a later
 * fence" answer carry across candidate tags, so a call costs O(raw.length) even
 * with many literal `"</think>"` mentions. Restarting the quote scan at
 * `spanStart` per candidate was O(candidates x length), and this runs on the
 * cumulative string for every SSE delta (#7334).
 *
 * `streaming` marks a mid-stream parse, where `raw` can still grow: an
 * enclosing ``` fence that has not closed yet may still close in a later
 * delta, so the unclosed-fence fallback is deferred to the final parse.
 *
 * `isKnownClose` reports delimiters the caller inserted itself (closing a
 * synthetic `reasoning_content` wrapper). Those positions are authoritative,
 * so the heuristics below - which only exist to interpret RAW model markers -
 * must not second-guess them.
 */
function findStructuralThinkClose(
  raw: string,
  spanStart: number,
  from: number,
  streaming = false,
  isKnownClose?: (index: number) => boolean,
): number {
  const FENCE = "```";
  let closeIndex = raw.indexOf(THINK_CLOSE_TAG, from);
  if (closeIndex === -1) return -1;

  // Greedy non-overlapping fence scan (matches Python str.count): `fences` is
  // the number of fence markers starting strictly before `nextFence`.
  let fences = 0;
  let nextFence = raw.indexOf(FENCE, spanStart);
  // Last fence marker in `raw`; only the odd-parity branch needs it, so it is
  // computed at most once and reused.
  let lastFence: number | undefined;
  // Running quote counts over [spanStart, cursor) per quote char, advanced
  // lazily with indexOf rather than a char-by-char loop (same answer, far less
  // work on ordinary prose, which is mostly quote-free).
  let dq = 0;
  let dqFrom = spanStart;
  let sq = 0;
  let sqFrom = spanStart;
  let bt = 0;
  let btFrom = spanStart;
  const quoteCount = (ch: string, end: number): number => {
    let n = ch === '"' ? dq : ch === "'" ? sq : bt;
    const cursor = ch === '"' ? dqFrom : ch === "'" ? sqFrom : btFrom;
    for (let at = raw.indexOf(ch, cursor); at !== -1 && at < end; ) {
      n += 1;
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
  // Memoized "is there a close tag at or after `at`", so the fence look-ahead
  // below stays amortized O(raw.length) across candidates instead of one full
  // indexOf each (#7334). `hit` is monotone: once none is found from some
  // offset, none is found from any later one.
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

  while (closeIndex !== -1) {
    while (nextFence !== -1 && nextFence < closeIndex) {
      fences += 1;
      nextFence = raw.indexOf(FENCE, nextFence + FENCE.length);
    }

    let literal: boolean;
    if (isKnownClose?.(closeIndex)) {
      // Our own delimiter: the boundary is already known, not inferred.
      literal = false;
    } else if (fences % 2 === 1) {
      // The close sits inside an open ``` fence. Global parity over the rest of
      // the span is wrong here, since a separate later unclosed fence would
      // misflag an earlier close whose own fence already closed (#7334).
      if (streaming) {
        // More deltas are coming, so "not closed yet" is not "never closes".
        // Defer like the backend extractor's hold: calling it structural now
        // and reversing it when the fence closes would bounce text out of the
        // thinking drawer and back, and latch reasoningDuration on a tag that
        // was never the real close (#7334).
        literal = true;
      } else {
        // Where the enclosing fence would close. The greedy cursor answers this
        // directly; only fall back to the O(n) scan when it is exhausted, since
        // overlapping runs such as "````" can hide a marker from it.
        let fenceClose = nextFence;
        if (fenceClose === -1) {
          if (lastFence === undefined) lastFence = raw.lastIndexOf(FENCE);
          if (lastFence >= closeIndex) fenceClose = lastFence;
        }
        // No closing ``` at all: the enclosing fence never closes, so this tag
        // is the genuine structural close -- an unclosed fence in the reasoning
        // must not swallow the visible answer. And a ``` that does follow only
        // proves the reasoning-side fence closed when reasoning continues past
        // it to a further close tag; otherwise that marker opens a fenced block
        // in the ANSWER, which used to hide the whole answer in the drawer for
        // "draft ```</think>Answer: ```js ... ```" (#7334). Mirrors the backend
        // extractor's _fence_unresolved_at_close.
        literal =
          fenceClose !== -1 && hasCloseTagFrom(fenceClose + FENCE.length);
      }
    } else {
      const before = closeIndex > spanStart ? raw[closeIndex - 1] : "";
      const after = raw[closeIndex + THINK_CLOSE_TAG.length] ?? "";
      if (
        !before ||
        !after ||
        !`"'\``.includes(before) ||
        !`"'\``.includes(after)
      ) {
        literal = false;
      } else {
        // The leading quote is literal only when it OPENS a span, i.e. an odd
        // count of that char since the reasoning start.
        literal = quoteCount(before, closeIndex) % 2 === 1;
      }
    }

    if (!literal) return closeIndex;
    closeIndex = raw.indexOf(
      THINK_CLOSE_TAG,
      closeIndex + THINK_CLOSE_TAG.length,
    );
  }
  return -1;
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
 * True once the reasoning block has *structurally* closed. Uses the same
 * quoted/fenced-literal classification as `parseAssistantContent`, so a literal
 * `</think>` inside reasoning (a quote or fenced example) does not count as the
 * end of thinking. A raw substring check would latch the reasoning-duration
 * timer on that literal tag and never correct it when the real close arrives,
 * underreporting the thought time (#7334).
 *
 * Callers polling mid-stream must pass `{ streaming: true }` for the same
 * reason: a tag inside a fence that has not closed *yet* is not a close.
 */
export function hasClosedThinkTag(
  raw: string,
  options?: ParseOptions,
): boolean {
  const openIndex = raw.indexOf(THINK_OPEN_TAG);
  const spanStart = openIndex === -1 ? 0 : openIndex + THINK_OPEN_TAG.length;
  return (
    findStructuralThinkClose(
      raw,
      spanStart,
      spanStart,
      options?.streaming ?? false,
      options?.isKnownClose,
    ) !== -1
  );
}
