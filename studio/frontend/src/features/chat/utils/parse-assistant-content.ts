// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelRunResult } from "@assistant-ui/react";

type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";

/**
 * Normalize streamed string or structured delta content to inline text.
 * Structured reasoning-only chunks remain distinguishable so their fallback
 * timer can span consecutive chunks even though each chunk carries closed tags.
 */
export function extractDeltaText(delta: unknown): {
  text: string;
  structuredReasoningContinues: boolean;
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
    return { text: delta, structuredReasoningContinues: false };
  }
  if (!Array.isArray(delta)) {
    return { text: "", structuredReasoningContinues: false };
  }

  let text = "";
  let structuredReasoningContinues = false;
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
      if (thinking) {
        text += `${THINK_OPEN_TAG}${thinking}${THINK_CLOSE_TAG}`;
        structuredReasoningContinues = true;
      }
    }
  }
  return { text, structuredReasoningContinues };
}

// ContentPart from @assistant-ui/react has readonly fields, so `last.text +=
// text` fails (TS2540). Replace the last element with a merged object instead.

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

export function parseAssistantContent(
  raw: string,
): ContentPart[] {
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
    const closeIndex = raw.indexOf(THINK_CLOSE_TAG, reasoningStart);
    if (closeIndex === -1) {
      appendReasoningPart(parts, raw.slice(reasoningStart));
      break;
    }

    appendReasoningPart(parts, raw.slice(reasoningStart, closeIndex));
    cursor = closeIndex + THINK_CLOSE_TAG.length;
  }

  return parts;
}

export function hasUnclosedThinkTag(raw: string): boolean {
  return raw.lastIndexOf(THINK_OPEN_TAG) > raw.lastIndexOf(THINK_CLOSE_TAG);
}

/**
 * The most characters of a `<think>` or `</think>` that can sit before the point
 * a previous call stopped at: one short of the longer tag, since a tag
 * straddling the boundary has at least one character on the new side.
 */
const THINK_TAG_OVERLAP =
  Math.max(THINK_OPEN_TAG.length, THINK_CLOSE_TAG.length) - 1;

export type ThinkTagTracker = {
  /**
   * Whether `text` ends inside a `<think>` block, that is, what
   * `hasUnclosedThinkTag(text)` returns.
   *
   * `text` must be the previous call's text with characters appended and then,
   * at most once, a suffix removed. That is the only way the streaming adapter's
   * buffer changes: deltas are appended, and the trailing template-literal strip
   * takes a suffix off the end. An unrelated string, or one whose committed
   * prefix was rewritten, gets an answer based on stale tag positions.
   */
  update(text: string): boolean;
};

/**
 * Track `hasUnclosedThinkTag` across a stream without rereading the buffer.
 *
 * `hasUnclosedThinkTag` walks the whole reply, so calling it once per SSE
 * arrival costs O(reply^2) over a reply. This keeps the index of the last open
 * and close tag and looks only at what arrived since the previous call, plus the
 * few characters in front of it that a tag split across arrivals can occupy. A
 * `<think>` delivered one character at a time over seven arrivals is found on
 * the arrival that completes it.
 */
export function createThinkTagTracker(): ThinkTagTracker {
  // Everything before this offset has been scanned; the indices are the last
  // position of each tag in it, or -1, exactly as `lastIndexOf` reports.
  let scanned = 0;
  let lastOpen = -1;
  let lastClose = -1;

  return {
    update(text: string): boolean {
      if (text.length >= scanned) {
        const from = Math.max(0, scanned - THINK_TAG_OVERLAP);
        const arrived = text.slice(from);
        // A tag re-found inside the overlap is one already recorded, so it can
        // only reproduce the index it gave last time.
        const openAt = arrived.lastIndexOf(THINK_OPEN_TAG);
        if (openAt !== -1) {
          lastOpen = Math.max(lastOpen, from + openAt);
        }
        const closeAt = arrived.lastIndexOf(THINK_CLOSE_TAG);
        if (closeAt !== -1) {
          lastClose = Math.max(lastClose, from + closeAt);
        }
      } else {
        // A suffix went away. Re-find only the tags that left with it; the
        // ordinary case, a strip of plain text, costs nothing here.
        if (lastOpen >= 0 && lastOpen + THINK_OPEN_TAG.length > text.length) {
          lastOpen = text.lastIndexOf(THINK_OPEN_TAG);
        }
        if (lastClose >= 0 && lastClose + THINK_CLOSE_TAG.length > text.length) {
          lastClose = text.lastIndexOf(THINK_CLOSE_TAG);
        }
      }
      scanned = text.length;
      return lastOpen > lastClose;
    },
  };
}
