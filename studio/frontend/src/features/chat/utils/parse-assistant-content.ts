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
