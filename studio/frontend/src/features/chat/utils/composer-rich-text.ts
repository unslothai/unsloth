// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// full: Markdown is styled as it is typed, and formatted pastes keep their formatting.
// paste: the composer stays a plain textarea; formatted pastes become Markdown.
// off: plain text in, plain text out.
export type ComposerRichText = "full" | "paste" | "off";

export const COMPOSER_RICH_TEXT_MODES: readonly ComposerRichText[] = [
  "full",
  "paste",
  "off",
];

export const DEFAULT_COMPOSER_RICH_TEXT: ComposerRichText = "paste";

// Records from before the three-way setting carry `richTextPaste: boolean` instead.
export function migrateComposerRichText(
  value: unknown,
  legacyRichTextPaste: unknown,
): ComposerRichText {
  if (COMPOSER_RICH_TEXT_MODES.includes(value as ComposerRichText))
    return value as ComposerRichText;
  return legacyRichTextPaste === false ? "off" : DEFAULT_COMPOSER_RICH_TEXT;
}
