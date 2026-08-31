// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Last-resort scrub for OpenAI's private-use citation markers
 * (`\uE200cite\uE202SOURCE_ID[\uE202...]\uE201`, see
 * https://developers.openai.com/api/docs/guides/citation-formatting).
 *
 * The backend rewrites them into real links, but a message persisted from an
 * interrupted stream -- or imported from another client -- can still carry one,
 * and Streamdown renders whatever it is handed.
 */

const CITE_MARKER_RE = /\uE200cite\uE202[^\uE201]*\uE201/g;
// An unterminated marker keeps its payload, so stripping the delimiters alone
// leaves "citeturn0search0" on screen. Anchored at the end because that is the
// only place a truncation can leave one. Mirrors the backend rule in
// external_provider._flush_pending_marker_tail, which drops an unclosed tail.
const CITE_PARTIAL_RE = /\uE200[^\uE201]*$/;
const PUA_ORPHAN_RE = /[\uE200\uE201\uE202]/g;

export function scrubOpenAICitationMarkers(text: string): string {
  if (!text) return text;
  if (
    !text.includes("\uE200") &&
    !text.includes("\uE201") &&
    !text.includes("\uE202")
  ) {
    return text;
  }
  return text
    .replace(CITE_MARKER_RE, "")
    .replace(CITE_PARTIAL_RE, "")
    .replace(PUA_ORPHAN_RE, "");
}
