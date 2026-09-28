// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export type SentAttachmentLayout = "list" | "chips";

export const SENT_ATTACHMENT_LIST_MAX = 6;

export const COMPOSER_ATTACHMENT_MAX_ROWS = 2;

export function sentAttachmentLayout(
  // The Appearance setting of the same name; spelled out so this module imports nothing.
  setting: "auto" | "list" | "chips",
  count: number,
): SentAttachmentLayout {
  if (setting === "auto") {
    return count > SENT_ATTACHMENT_LIST_MAX ? "chips" : "list";
  }
  return setting;
}

export function composerAttachmentsOverflow(
  count: number,
  width: number,
  cardWidth: number,
  gap: number,
): boolean {
  if (count === 0 || cardWidth <= 0) return false;
  // Cards sized to a fifth of the row divide it exactly; the slack absorbs subpixel rounding.
  const perRow = Math.max(1, Math.floor((width + gap) / (cardWidth + gap) + 0.01));
  return Math.ceil(count / perRow) > COMPOSER_ATTACHMENT_MAX_ROWS;
}
