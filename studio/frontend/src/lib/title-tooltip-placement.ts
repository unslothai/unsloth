// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const GAP = 6;
const EDGE = 8;

/** Centred below `anchor`, flipped above near the bottom edge, clamped to the viewport. */
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

/** `current`'s title is held empty, so `found` skips past it: keep `current` while still over it. */
export function resolveTitleTarget<T extends { contains(other: T | null): boolean }>(
  current: T | null,
  found: T | null,
  stillOverCurrent: boolean,
): T | null {
  if (!current || !stillOverCurrent) return found;
  return found && found !== current && current.contains(found) ? found : current;
}

export type TitleStandIn = "aria-label" | "aria-description" | null;

/** A lifted title is an unnamed control's name, else its description unless it already has one. */
export function titleStandIn({
  labelled,
  named,
  described,
}: { labelled: boolean; named: boolean; described: boolean }): TitleStandIn {
  if (!labelled && !named) return "aria-label";
  return described ? null : "aria-description";
}
