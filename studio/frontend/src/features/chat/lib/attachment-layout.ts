// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How attachments lay out in the composer and in a sent message. Pure, so the layout rules can be
// tested without a DOM.

export type SentAttachmentLayout = "list" | "chips";

/** Past this many attachments, Auto collapses a sent message's files to chips. */
export const SENT_ATTACHMENT_LIST_MAX = 6;

/** How many rows of cards the composer grows by before they become one scrolling strip. */
export const COMPOSER_ATTACHMENT_MAX_ROWS = 2;

/** Which layout a sent message's attachments take: the setting, with Auto turning a long list
 *  into chips. */
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

/** Whether `count` cards of `cardWidth` need more than the allowed rows in `width`. */
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
