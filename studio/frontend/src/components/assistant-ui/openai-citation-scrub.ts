// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Strips OpenAI's private-use citation markers, which survive in persisted or
 * imported messages. developers.openai.com/api/docs/guides/citation-formatting */

const CITE_MARKER_RE = /\uE200cite\uE202[^\uE201]*\uE201/g;
// An unterminated marker keeps its payload, so stripping the delimiters alone leaves
// "citeturn0search0". Same rule as external_provider._flush_pending_marker_tail.
// Only what a truncation can leave: a lone open byte elsewhere in the text is one
// stray glyph for the sweep below, not a licence to drop the rest of the message.
const CITE_PARTIAL_RE = /\uE200(?:c|ci|cit|cite|cite\uE202[^\uE200\uE201]*)$/;
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
