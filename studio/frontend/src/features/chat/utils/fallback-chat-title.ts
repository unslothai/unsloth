// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Match the existing inline-rename limit; CSS owns visual truncation. */
export const FALLBACK_CHAT_TITLE_MAX_LENGTH = 120;

const FIRST_LINE_PATTERN = /\r?\n/;
const WHITESPACE_PATTERN = /\s+/g;

function cleanFirstLine(userText: string): string {
  const firstLine = (userText || "").split(FIRST_LINE_PATTERN, 1)[0] ?? "";
  return firstLine.replace(WHITESPACE_PATTERN, " ").trim();
}

export function fallbackTitleFromUserText(userText: string): string {
  const cleaned = cleanFirstLine(userText);
  return cleaned
    ? cleaned.slice(0, FALLBACK_CHAT_TITLE_MAX_LENGTH)
    : "New Chat";
}
