// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Where the API monitor panel sits so that it does not land on top of another
 * floating panel.
 *
 * The Live resource monitor defaults to the bottom-right corner and so does
 * this one, which put this panel's header, its Close button and its drag
 * handle underneath a window the user can resize across the whole viewport.
 *
 * Avoidance rather than a z-index, because it is what Studio already does in
 * this corner: the notification stack does not outrank the monitor, it steps
 * over it (monitor-frame-store's stackGeometry, whose gap and inset this file
 * matches so the three surfaces line up on one grid). This panel reads the same
 * published boxes and does the same thing one rung further in.
 *
 * Pure, and separate from the component, because the interesting part is what
 * happens when there is nowhere clear to go.
 */

/** A box in viewport coordinates, as published to the monitor frame store. */
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

/** The panel's top-left corner, in viewport coordinates. */
export interface PanelAnchor {
  left: number;
  top: number;
}

/** The inset the panel shipped with (`bottom-4 right-4`), and the stack's. */
export const PANEL_MARGIN = 16;
/** Clearance left between the panel and a box it has stepped over. */
export const PANEL_GAP = 8;
/**
 * How far down the panel's own top edge may come.
 *
 * The app's top chrome -- the navbar at 48px, or the custom Tauri titlebar band
 * that carries the window's own minimise and close buttons -- publishes no box,
 * and this layer paints over all of it. Stepping over something tall would
 * otherwise park a floating panel on the window controls, which is the one
 * place a user must always be able to reach.
 */
export const PANEL_TOP_MARGIN = 64;

/**
 * A place to sit, and how much this panel wants it.
 *
 * Two orders, because they disagree. `clearRank` is which free spot to take:
 * the corner the panel shipped in, else the smallest step out of the way, else
 * anywhere. `refugeRank` is which spot to take when none of them are free, and
 * there the right-hand corners are the worst ones: both floating panels put
 * their close button and their resize grip on their right edge, so covering a
 * right-hand corner is what takes a covered window away from the user.
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

/**
 * Keep the whole panel on screen. The margin wins a viewport too small to hold
 * the panel at all, so its top-left corner -- the title, the drag handle --
 * stays reachable and the overflow goes off the far edge.
 *
 * Also applied to a panel the user has placed by hand, which stops being
 * re-placed but must not be left off the edge by a window the user then
 * shrinks: the panel keeps no position across reloads, so a panel stranded
 * outside the viewport has no way back.
 */
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
  // Stepping over the lowest box first: clearing that one may be enough, and
  // it is the smallest move away from where the user last saw the panel.
  const steps = [...obstacles]
    .sort((a, b) => b.top - a.top)
    .map((box) => ({ left: right, top: box.top - PANEL_GAP - size.height }));
  // The corner each candidate came from, kept rather than inferred back out of
  // its coordinates. A 400px panel in a 768px window is anchored right at
  // left=352, which is left of the midpoint, so reading the side off `left`
  // ranked the right-hand corner as a left-hand refuge. Being first, it then
  // won the tie and the panel stayed exactly where it was, over the Close
  // button and the resize grip this fallback exists to keep reachable.
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
      // Left half first, and within a half the bottom first: this panel and
      // the stack both live along the bottom edge, so a refuge up top is the
      // bigger surprise.
      refugeRank:
        (anchor.rightSide ? 2 : 0) +
        (placed.top > viewport.height / 2 ? 0 : 1),
    };
  });
}

/**
 * The best anchor for a panel of `size` given everything it must keep clear of.
 *
 * The first candidate that touches nothing wins, so a free corner keeps the
 * panel exactly where it has always been. If every candidate is covered -- a
 * monitor resized over the whole viewport -- the least covered one wins, and
 * an outright tie goes to the corner that leaves the covered window's own
 * controls reachable rather than back to the corner both panels want.
 */
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

/**
 * Whether one of these boxes hides the panel completely.
 *
 * The panel opens itself, so unlike the resource monitor it can be swallowed
 * without the user having asked for anything, and a panel with no pixel showing
 * has no way back to the front. This is the one case where the layer, not the
 * geometry, has to give.
 */
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
