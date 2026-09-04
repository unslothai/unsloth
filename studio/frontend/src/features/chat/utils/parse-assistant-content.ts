// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelRunResult } from "@assistant-ui/react";

type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";

/** Normalize streamed string or structured delta content to inline text. Structured
 *  reasoning-only chunks stay distinguishable so their fallback timer can span consecutive
 *  chunks even though each chunk carries closed tags. */
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

// ContentPart from @assistant-ui/react has readonly fields, so `last.text += text` fails
// (TS2540). Replace the last element with a merged object instead.

export function appendTextPart(parts: ContentPart[], text: string): void {
  if (!text) return;
  const last = parts.at(-1);
  if (last?.type === "text") {
    parts[parts.length - 1] = { type: "text", text: last.text + text };
    return;
  }
  parts.push({ type: "text", text });
}

export function appendReasoningPart(parts: ContentPart[], text: string): void {
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

/** The most characters of a `<think>` or `</think>` that can sit before the point a previous
 *  call stopped at: one short of the longer tag, since a tag straddling the boundary has at
 *  least one character on the new side. */
const THINK_TAG_OVERLAP =
  Math.max(THINK_OPEN_TAG.length, THINK_CLOSE_TAG.length) - 1;

export type ThinkTagTracker = {
  /** Take the characters an arrival added. Reads `delta` and the seven characters in front of it,
   *  never the accumulated reply. */
  append(delta: string): void;
  /** Take back the suffix a rewrite removed. `text` is the buffer AFTER the removal, and is read
   *  only when a tag left with the suffix or the retained overlap has to be refilled. */
  retract(text: string): void;
  /** Whether the accumulated text ends inside a `<think>` block, that is, what
   *  `hasUnclosedThinkTag` returns for the same text. */
  endsInsideThink(): boolean;
};

/** Track `hasUnclosedThinkTag` across a stream without reading the buffer. That helper walks the
 *  whole reply, so calling it per SSE arrival costs O(reply^2). Reading the buffer at all
 *  costs that much even for one character, since `text += delta` builds a cons string the next
 *  read flattens. So this takes the delta and keeps the last seven characters itself, the most
 *  a tag split across arrivals can hide in front of one. */
export function createThinkTagTracker(): ThinkTagTracker {
  // Characters taken so far; the two indices are the last position of each tag within them, or
  // -1, exactly as `lastIndexOf` reports.
  let length = 0;
  let lastOpen = -1;
  let lastClose = -1;
  // The final `THINK_TAG_OVERLAP` characters, held flat so the next arrival can be searched
  // together with the tag prefix it may complete.
  let overlap = "";

  const refindWithin = (text: string): void => {
    if (lastOpen >= 0 && lastOpen + THINK_OPEN_TAG.length > text.length) {
      lastOpen = text.lastIndexOf(THINK_OPEN_TAG);
    }
    if (lastClose >= 0 && lastClose + THINK_CLOSE_TAG.length > text.length) {
      lastClose = text.lastIndexOf(THINK_CLOSE_TAG);
    }
  };

  return {
    append(delta: string): void {
      if (!delta) {
        return;
      }
      const window = overlap + delta;
      const from = length - overlap.length;
      // A tag re-found inside the overlap is the one already recorded, so it can only reproduce the
      // index it produced last time.
      const openAt = window.lastIndexOf(THINK_OPEN_TAG);
      if (openAt !== -1) {
        lastOpen = Math.max(lastOpen, from + openAt);
      }
      const closeAt = window.lastIndexOf(THINK_CLOSE_TAG);
      if (closeAt !== -1) {
        lastClose = Math.max(lastClose, from + closeAt);
      }
      length += delta.length;
      overlap =
        window.length <= THINK_TAG_OVERLAP
          ? window
          : window.slice(window.length - THINK_TAG_OVERLAP);
    },
    retract(text: string): void {
      refindWithin(text);
      length = text.length;
      overlap =
        text.length <= THINK_TAG_OVERLAP
          ? text
          : text.slice(text.length - THINK_TAG_OVERLAP);
    },
    endsInsideThink(): boolean {
      return lastOpen > lastClose;
    },
  };
}
