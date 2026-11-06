// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelRunResult } from "@assistant-ui/react";

type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";

export function extractDeltaText(delta: unknown): {
  text: string;
  structuredReasoningContinues: boolean;
  hasStructuredReasoning: boolean;
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
    return {
      text: delta,
      structuredReasoningContinues: false,
      hasStructuredReasoning: false,
    };
  }
  if (!Array.isArray(delta)) {
    return {
      text: "",
      structuredReasoningContinues: false,
      hasStructuredReasoning: false,
    };
  }

  let text = "";
  let structuredReasoningContinues = false;
  let hasStructuredReasoning = false;
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
        hasStructuredReasoning = true;
      }
    }
  }
  return { text, structuredReasoningContinues, hasStructuredReasoning };
}

// ContentPart fields are readonly (TS2540), so replace the last element instead.

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
  { parseThink = true }: { parseThink?: boolean } = {},
): ContentPart[] {
  const parts: ContentPart[] = [];
  if (!raw) {
    return parts;
  }
  if (!parseThink) {
    appendTextPart(parts, raw);
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

const THINK_TAG_OVERLAP =
  Math.max(THINK_OPEN_TAG.length, THINK_CLOSE_TAG.length) - 1;

export type ThinkTagTracker = {
  append(delta: string): void;
  retract(text: string): void;
  endsInsideThink(): boolean;
};

/** Tracks `hasUnclosedThinkTag` incrementally; reading the buffer per arrival is O(reply^2). */
export function createThinkTagTracker(): ThinkTagTracker {
  let length = 0;
  let lastOpen = -1;
  let lastClose = -1;
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
