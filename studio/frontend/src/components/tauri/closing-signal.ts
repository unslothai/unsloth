// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Rust emits this from the quit thread in main.rs once quit confirmations pass (Windows only). */
export const APP_CLOSING_EVENT = "app-closing";

/** And takes it back down with this, for a quit that never reaches the exit. */
export const APP_CLOSING_CANCELLED_EVENT = "app-closing-cancelled";

// Module state: events arrive on the backend hook, the overlay renders from the provider.
let closing = false;
const listeners = new Set<(closing: boolean) => void>();

function setAppClosing(next: boolean): void {
  if (closing === next) {
    return;
  }
  closing = next;
  // Over a copy: a listener that subscribes or throws must not decide who else hears.
  for (const listener of [...listeners]) {
    listener(next);
  }
}

export function isAppClosing(): boolean {
  return closing;
}

/** Paint the overlay. Idempotent, so a re-emitted app-closing costs no re-render. */
export function markAppClosing(): void {
  setAppClosing(true);
}

/** The quit never reached the exit, so the app stays and the overlay has to go. */
export function clearAppClosing(): void {
  setAppClosing(false);
}

export function subscribeAppClosing(
  listener: (closing: boolean) => void,
): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
