// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MAX_TOOL_TEXT_CHARS = 256_000;

const TOOL_TEXT_TRUNCATION_NOTICE =
  "\n\n... (tool result truncated to 256,000 chars for the model; the full output is not retained in model context.)";

export function capToolText(text: string): string {
  if (text.length <= MAX_TOOL_TEXT_CHARS) {
    return text;
  }
  if (text.endsWith(TOOL_TEXT_TRUNCATION_NOTICE)) {
    return text;
  }
  const head = text.slice(0, MAX_TOOL_TEXT_CHARS);
  const cut = head.lastIndexOf("\n");
  const onBoundary = cut > 0 && cut >= Math.floor(MAX_TOOL_TEXT_CHARS / 2);
  return (onBoundary ? head.slice(0, cut) : head) + TOOL_TEXT_TRUNCATION_NOTICE;
}
