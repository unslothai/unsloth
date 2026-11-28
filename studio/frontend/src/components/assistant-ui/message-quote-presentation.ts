// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure state rules for the floating quote pill.

export type QuotePillMode = "hidden" | "single" | "multiple";

/** Visible attribution only: the internal message id stays in the anchor and model envelope. */
export function quoteSourceLabel(
  turn: number,
  turnLabel: (turn: number) => string,
): string {
  return turn > 0 ? turnLabel(turn) : "";
}

export function quotePillMode(count: number): QuotePillMode {
  if (count <= 0) {
    return "hidden";
  }
  return count === 1 ? "single" : "multiple";
}

/** Removing the last quote closes the dropdown; otherwise its state survives. */
export function quotePillExpandedAfterCountChange(
  previousCount: number,
  nextCount: number,
  expanded: boolean,
): boolean {
  if (nextCount <= 0) {
    return false;
  }
  if (previousCount <= 0) {
    return false;
  }
  return expanded;
}

/**
 * A touch pin owns the single-quote preview. Radix requests a close on
 * pointerdown/click; ignore that temporary close while this exact quote is
 * pinned.
 */
export function shouldApplyQuotePreviewOpen(
  pinnedQuoteId: string | null,
  quoteId: string | null,
  requestedOpen: boolean,
): boolean {
  return requestedOpen || quoteId === null || pinnedQuoteId !== quoteId;
}

/** First tap pins the quote id; a second tap on the same quote unpins it. */
export function toggledPinnedQuoteId(
  pinnedQuoteId: string | null,
  quoteId: string,
): string | null {
  return pinnedQuoteId === quoteId ? null : quoteId;
}
