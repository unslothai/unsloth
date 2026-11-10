// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * WebKit folds window and document pointer listeners into the style of every element (event listener
 * regions), so the first listener of a type added there, or the last one removed, restyles the whole
 * page. react-resizable-panels adds and removes its document pointer listeners each time a panel group
 * re-registers, which the browser pane does several times while it opens and closes: about 2 s per
 * step on a 100-turn chat. One no-op listener of each type held for the page's lifetime keeps that set
 * constant, so those adds and removes change nothing. Chromium is unaffected either way.
 */

export const PINNED_POINTER_EVENTS = [
  "pointerdown",
  "pointermove",
  "pointerup",
  "pointercancel",
  "pointerover",
  "pointerout",
  "pointerenter",
  "pointerleave",
] as const;

function keepRegion(): void {}

/** Idempotent: the same listener is only ever registered once per type. */
export function pinPointerListenerRegions(
  target: Pick<EventTarget, "addEventListener">,
): void {
  for (const type of PINNED_POINTER_EVENTS) target.addEventListener(type, keepRegion);
}
