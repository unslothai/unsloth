// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pictographs, regional-indicator flag pairs, and the modifiers that recolor an
// otherwise plain glyph (VS-16, ZWJ, skin tones, the tag characters used by
// England/Scotland/Wales-style flag sequences) all route through a color font.
const COLOR_FONT_TRIGGER_RE =
  /\p{Extended_Pictographic}|\p{Regional_Indicator}|\p{Emoji_Modifier}|[\u{FE0F}\u{200D}]|[\u{E0020}-\u{E007F}]/gu;

/**
 * Drop characters that render through a color font (emoji, flags, and their
 * modifiers), for hosts where that path is known to crash the renderer
 * (issue #9453: a Linux AppImage's bundled WebKitGTK/Skia asserts on a
 * COLRv1 color-stop table). READMEs are arbitrary Hugging Face content, so
 * this is the only place that can head it off before render.
 */
export function stripColorFontTriggers(markdown: string): string {
  return markdown.replace(COLOR_FONT_TRIGGER_RE, "");
}
