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

/**
 * True when a close tag looks like quoted/code content rather than a block
 * end (#7066): flanked by quote chars, with the leading quote OPENING a span
 * (odd count of that quote char since the reasoning start).
 */
function isLiteralThinkClose(
  raw: string,
  spanStart: number,
  closeIndex: number,
): boolean {
  const before = closeIndex > spanStart ? raw[closeIndex - 1] : "";
  const after = raw[closeIndex + THINK_CLOSE_TAG.length] ?? "";
  if (!before || !after) return false;
  if (!`"'\``.includes(before) || !`"'\``.includes(after)) return false;
  let count = 0;
  for (let i = spanStart; i < closeIndex; i++) {
    if (raw[i] === before) count++;
  }
  return count % 2 === 1;
}

/** First structural (non-quoted) close tag at or after `from`. */
function findStructuralThinkClose(
  raw: string,
  spanStart: number,
  from: number,
): number {
  let closeIndex = raw.indexOf(THINK_CLOSE_TAG, from);
  while (
    closeIndex !== -1 &&
    isLiteralThinkClose(raw, spanStart, closeIndex)
  ) {
    closeIndex = raw.indexOf(
      THINK_CLOSE_TAG,
      closeIndex + THINK_CLOSE_TAG.length,
    );
  }
  return closeIndex;
}

export function parseAssistantContent(raw: string): ContentPart[] {
  const parts: ContentPart[] = [];
  if (!raw) {
    return parts;
  }

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

export function hasClosedThinkTag(raw: string): boolean {
  return raw.includes(THINK_CLOSE_TAG);
}
