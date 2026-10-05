// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The browser interface scale as a factor on layout px, for widths held in JS.
 * Always 1 on desktop, where webview zoom already scales every px. No imports,
 * so width stores stay loadable without the app.
 */
let factor = 1;
const listeners = new Set<() => void>();

export function layoutScale(): number {
  return factor;
}

export function setLayoutScale(next: number): void {
  if (next === factor) return;
  factor = next;
  for (const listener of listeners) listener();
}

export function subscribeLayoutScale(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
