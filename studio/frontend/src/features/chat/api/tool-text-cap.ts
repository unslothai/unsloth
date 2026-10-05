// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MAX_TOOL_TEXT_CHARS = 256_000;

// Shared prefix of every cap notice, including the backend's spill notice that names the saved file.
const TOOL_TEXT_TRUNCATION_MARKER =
  "\n\n... (tool result truncated to 256,000 chars for the model;";
const TOOL_TEXT_TRUNCATION_NOTICE = `${TOOL_TEXT_TRUNCATION_MARKER} the full output is not retained in model context.)`;

export function capToolText(text: string): string {
  if (text.length <= MAX_TOOL_TEXT_CHARS) {
    return text;
  }
  if (
    text.lastIndexOf(TOOL_TEXT_TRUNCATION_MARKER) >=
    Math.max(0, text.length - 2_000)
  ) {
    return text;
  }
  // Never end on a lone high surrogate: the backend cannot UTF-8 encode it.
  const last = text.charCodeAt(MAX_TOOL_TEXT_CHARS - 1);
  const end =
    last >= 0xd800 && last <= 0xdbff
      ? MAX_TOOL_TEXT_CHARS - 1
      : MAX_TOOL_TEXT_CHARS;
  const head = text.slice(0, end);
  const cut = head.lastIndexOf("\n");
  const onBoundary = cut > 0 && cut >= Math.floor(MAX_TOOL_TEXT_CHARS / 2);
  return (onBoundary ? head.slice(0, cut) : head) + TOOL_TEXT_TRUNCATION_NOTICE;
}
