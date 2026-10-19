// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const GAP = 6;
const EDGE = 8;

/** Where a tooltip of `size` goes for `anchor`: centred below, flipped above near the bottom edge,
 *  and kept inside the viewport. */
export function placeTitleTooltip(
  anchor: Pick<DOMRect, "left" | "top" | "bottom" | "width">,
  size: { width: number; height: number },
  viewport: { width: number; height: number },
): { left: number; top: number } {
  const below = anchor.bottom + GAP;
  const top =
    below + size.height + EDGE > viewport.height ? anchor.top - GAP - size.height : below;
  const centred = anchor.left + anchor.width / 2 - size.width / 2;
  const left = Math.min(Math.max(EDGE, centred), viewport.width - EDGE - size.width);
  return { left, top: Math.max(EDGE, top) };
}
