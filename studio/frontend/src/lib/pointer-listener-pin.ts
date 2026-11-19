// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Pins one no-op pointer listener per type: WebKit restyles only when the set of listeners changes.
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
