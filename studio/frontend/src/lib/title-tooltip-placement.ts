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

/** The titled element now under the pointer. `current` has had its title lifted, so `found` (the
 *  nearest titled ancestor of the hit) skips past it: while the pointer is still over `current`,
 *  it stays the target unless a titled element inside it was hit. */
export function resolveTitleTarget<T extends { contains(other: T | null): boolean }>(
  current: T | null,
  found: T | null,
  stillOverCurrent: boolean,
): T | null {
  if (!current || !stillOverCurrent) return found;
  return found && found !== current && current.contains(found) ? found : current;
}

export type TitleStandIn = "aria-label" | "aria-description" | null;

/** What keeps a lifted title's meaning for assistive tech: the name of an icon-only control, the
 *  description of one already named by its text or alt, nothing when an aria label names it. */
export function titleStandIn({ labelled, named }: { labelled: boolean; named: boolean }): TitleStandIn {
  if (labelled) return null;
  return named ? "aria-description" : "aria-label";
}
