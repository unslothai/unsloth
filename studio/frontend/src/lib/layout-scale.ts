// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Always 1 on desktop, where webview zoom scales every px. No imports, so width stores load without the app. */
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
