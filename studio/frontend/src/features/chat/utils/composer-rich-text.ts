// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ComposerRichText = "full" | "paste" | "off";

export const COMPOSER_RICH_TEXT_MODES: readonly ComposerRichText[] = [
  "full",
  "paste",
  "off",
];

export const DEFAULT_COMPOSER_RICH_TEXT: ComposerRichText = "paste";

export function migrateComposerRichText(
  value: unknown,
  legacyRichTextPaste: unknown,
): ComposerRichText {
  if (COMPOSER_RICH_TEXT_MODES.includes(value as ComposerRichText))
    return value as ComposerRichText;
  return legacyRichTextPaste === false ? "off" : DEFAULT_COMPOSER_RICH_TEXT;
}
