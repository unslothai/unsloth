// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Places the API monitor panel clear of other floating panels (the Live resource monitor shares
 * the corner). Avoidance, not z-index: both are windows whose controls must stay reachable.
 */

export interface PanelRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface PanelSize {
  width: number;
  height: number;
}

export interface PanelViewport {
  width: number;
  height: number;
}

export interface PanelAnchor {
  left: number;
  top: number;
}

/** The inset the panel shipped with (`bottom-4 right-4`), and the stack's. */
export const PANEL_MARGIN = 16;
export const PANEL_GAP = 8;
/** Keeps the panel below the navbar or Tauri titlebar, which publish no box. */
export const PANEL_TOP_MARGIN = 64;

/**
 * `clearRank` picks among free spots; `refugeRank` among covered ones, where right-hand corners
 * are worst because both panels keep their close button and resize grip there.
 */
interface Candidate {
  anchor: PanelAnchor;
  clearRank: number;
  refugeRank: number;
}

function overlapArea(
  anchor: PanelAnchor,
  size: PanelSize,
  box: PanelRect,
): number {
  const width =
    Math.min(anchor.left + size.width, box.right) -
    Math.max(anchor.left, box.left);
  const height =
    Math.min(anchor.top + size.height, box.bottom) -
    Math.max(anchor.top, box.top);
  return width > 0 && height > 0 ? width * height : 0;
}

/** Keep the panel on screen, top-left corner first; also for hand-placed panels on resize. */
export function clampPanelToViewport(
  anchor: PanelAnchor,
  size: PanelSize,
  viewport: PanelViewport,
): PanelAnchor {
  return {
    left: Math.max(
      PANEL_MARGIN,
      Math.min(anchor.left, viewport.width - PANEL_MARGIN - size.width),
    ),
    top: Math.max(
      PANEL_TOP_MARGIN,
      Math.min(anchor.top, viewport.height - PANEL_MARGIN - size.height),
    ),
  };
}

function candidates(
  size: PanelSize,
  obstacles: readonly PanelRect[],
  viewport: PanelViewport,
): Candidate[] {
  const right = viewport.width - PANEL_MARGIN - size.width;
  const bottom = viewport.height - PANEL_MARGIN - size.height;
  // Lowest obstacle first: the smallest move from where the user last saw the panel.
  const steps = [...obstacles]
    .sort((a, b) => b.top - a.top)
    .map((box) => ({ left: right, top: box.top - PANEL_GAP - size.height }));
  // Keep each candidate's corner explicitly; inferring the side from `left` misranks wide panels.
  const ordered: Array<PanelAnchor & { rightSide: boolean }> = [
    { left: right, top: bottom, rightSide: true },
    ...steps.map((step) => ({ ...step, rightSide: true })),
    { left: PANEL_MARGIN, top: bottom, rightSide: false },
    { left: PANEL_MARGIN, top: PANEL_TOP_MARGIN, rightSide: false },
    { left: right, top: PANEL_TOP_MARGIN, rightSide: true },
  ];
  return ordered.map((anchor, index) => {
    const placed = clampPanelToViewport(anchor, size, viewport);
    return {
      anchor: placed,
      clearRank: index,
      // Prefer left half, then bottom: panel and stack live along the bottom edge.
      refugeRank:
        (anchor.rightSide ? 2 : 0) +
        (placed.top > viewport.height / 2 ? 0 : 1),
    };
  });
}

/** A free spot wins; when all are covered the least covered wins, ties by refuge rank. */
export function placeFloatingPanel(
  size: PanelSize,
  obstacles: readonly PanelRect[],
  viewport: PanelViewport,
): PanelAnchor {
  const options = candidates(size, obstacles, viewport);
  let best = options[0];
  let bestOverlap = Number.POSITIVE_INFINITY;
  for (const option of options) {
    let overlap = 0;
    for (const box of obstacles) {
      overlap += overlapArea(option.anchor, size, box);
    }
    if (overlap === 0) {
      return option.anchor;
    }
    if (
      overlap < bestOverlap ||
      (overlap === bestOverlap && option.refugeRank < best.refugeRank)
    ) {
      best = option;
      bestOverlap = overlap;
    }
  }
  return best.anchor;
}

/** The panel opens itself, so if fully hidden the layer, not the geometry, has to give. */
export function isFullyCovered(
  anchor: PanelAnchor,
  size: PanelSize,
  obstacles: readonly PanelRect[],
): boolean {
  return obstacles.some(
    (box) =>
      box.left <= anchor.left &&
      box.top <= anchor.top &&
      box.right >= anchor.left + size.width &&
      box.bottom >= anchor.top + size.height,
  );
}
