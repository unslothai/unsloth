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
  cardWidths: readonly number[],
  width: number,
  gap: number,
  maxRows: number = COMPOSER_ATTACHMENT_MAX_ROWS,
): boolean {
  // Wraps as flex-wrap does; the slack absorbs subpixel rounding of cards that divide the row exactly.
  let rows = 1;
  let used = 0;
  for (const card of cardWidths) {
    if (card <= 0) continue;
    if (used > 0 && used + gap + card > width + 0.5) {
      rows += 1;
      used = card;
    } else {
      used += (used > 0 ? gap : 0) + card;
    }
  }
  return rows > maxRows;
}
