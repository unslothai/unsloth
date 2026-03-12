// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelRunResult } from "@assistant-ui/react";

type ContentPart = NonNullable<ChatModelRunResult["content"]>[number];

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";

function appendTextPart(parts: ContentPart[], text: string): void {
  if (text) {
    parts.push({ type: "text", text });
  }
}

function appendReasoningPart(parts: ContentPart[], text: string): void {
  if (text) {
    parts.push({ type: "reasoning", text });
  }
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

export function hasClosedThinkTag(raw: string): boolean {
  return raw.includes(THINK_CLOSE_TAG);
}
