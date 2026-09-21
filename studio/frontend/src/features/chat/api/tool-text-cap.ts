// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Mirrors `cap_tool_text` in studio/backend/core/inference/tools.py, so the frontend's
 * model-bound replay path applies the same 256 KB floor as the backend's live loop.
 * Applied only where text is prepared for the model: the tool-card keeps the full result,
 * and MCP image envelopes ride below the cap because the body is cut before re-appending.
 */
export const MAX_TOOL_TEXT_CHARS = 256_000;

const TOOL_TEXT_TRUNCATION_NOTICE =
  "\n\n... (tool result truncated to 256,000 chars for the model; the full output is not retained in model context.)";

/** Cap tool-result text for the model at `MAX_TOOL_TEXT_CHARS` characters, unconditionally.
 *  Truncates at a nearby line break when there is one inside the room, then appends the
 *  truncation notice. Idempotent: a result that already carries the notice passes through
 *  uncut, so capping twice (once here, once on the backend) is a no-op. */
export function capToolText(text: string): string {
  if (text.length <= MAX_TOOL_TEXT_CHARS) {
    return text;
  }
  if (text.endsWith(TOOL_TEXT_TRUNCATION_NOTICE)) {
    return text;
  }
  const head = text.slice(0, MAX_TOOL_TEXT_CHARS);
  const cut = head.lastIndexOf("\n");
  const onBoundary = cut > 0 && cut >= MAX_TOOL_TEXT_CHARS / 2;
  return (onBoundary ? head.slice(0, cut) : head) + TOOL_TEXT_TRUNCATION_NOTICE;
}