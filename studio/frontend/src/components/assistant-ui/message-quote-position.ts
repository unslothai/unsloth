// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure geometry for the draggable quote pill and its dropdown. No DOM types or
// React state here, so the rules stay testable without a browser.

export type QuotePillPosition = {
  readonly left: number;
  readonly top: number;
};

export type QuotePillSize = {
  readonly width: number;
  readonly height: number;
};

export type QuotePillViewport = {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
};

export type QuoteDropdownSide = "above" | "below";

export type QuoteDropdownPlacement = {
  readonly left: number;
  readonly top: number;
  readonly maxHeight: number;
  readonly side: QuoteDropdownSide;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

export function clampQuotePillPosition(
  position: QuotePillPosition,
  size: QuotePillSize,
  viewport: QuotePillViewport,
  margin = 8,
): QuotePillPosition {
  return {
    left: clamp(
      position.left,
      viewport.left + margin,
      viewport.left + viewport.width - margin - size.width,
    ),
    top: clamp(
      position.top,
      viewport.top + margin,
      viewport.top + viewport.height - margin - size.height,
    ),
  };
}

/** One shared outer width for the quote pill and its dropdown, capped by the chat viewport. */
export function clampQuoteCollectionWidth(
  width: number,
  viewportWidth: number,
  margin = 8,
): number {
  return Math.max(0, Math.min(width, viewportWidth - margin * 2));
}

export function quotePillDragExceeded(
  startX: number,
  startY: number,
  currentX: number,
  currentY: number,
  threshold = 4,
): boolean {
  const dx = currentX - startX;
  const dy = currentY - startY;
  return dx * dx + dy * dy >= threshold * threshold;
}

/**
 * Chooses the side with enough room and clamps the dropdown to the chat
 * viewport. The caller applies `maxHeight` with overflow scrolling.
 */
export function placeQuoteDropdown(
  pillPosition: QuotePillPosition,
  pillSize: QuotePillSize,
  dropdownSize: QuotePillSize,
  viewport: QuotePillViewport,
  margin = 8,
  gap = 6,
): QuoteDropdownPlacement {
  const minTop = viewport.top + margin;
  const maxBottom = viewport.top + viewport.height - margin;
  const belowTop = pillPosition.top + pillSize.height + gap;
  const spaceBelow = Math.max(0, maxBottom - belowTop);
  const spaceAbove = Math.max(0, pillPosition.top - gap - minTop);

  // Prefer below when the whole dropdown fits, otherwise use the roomier side.
  const side: QuoteDropdownSide =
    spaceBelow >= dropdownSize.height || spaceBelow >= spaceAbove
      ? "below"
      : "above";
  const maxHeight = side === "below" ? spaceBelow : spaceAbove;
  const height = Math.min(dropdownSize.height, maxHeight);
  const top =
    side === "below"
      ? belowTop
      : Math.max(minTop, pillPosition.top - gap - height);
  const left = clamp(
    pillPosition.left + pillSize.width / 2 - dropdownSize.width / 2,
    viewport.left + margin,
    viewport.left + viewport.width - margin - dropdownSize.width,
  );

  return { left, top, maxHeight, side };
}
