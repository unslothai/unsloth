// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Rust asks for the closing overlay with this, from the quit thread in main.rs. */
export const APP_CLOSING_EVENT = "app-closing";

/** And takes it back down with this, when a quit confirmation declines. */
export const APP_CLOSING_CANCELLED_EVENT = "app-closing-cancelled";

// Module state rather than React state: the close button lives in the titlebar and the
// overlay in the provider, and neither is an ancestor of the other. A webview reload
// resets it, which fails open: no quit survives the reload either.
let closing = false;
const listeners = new Set<(closing: boolean) => void>();

function setAppClosing(next: boolean): void {
  if (closing === next) {
    return;
  }
  closing = next;
  // Over a copy: a listener that subscribes or throws mid-notify must not decide which
  // of the others hear about the quit.
  for (const listener of [...listeners]) {
    listener(next);
  }
}

export function isAppClosing(): boolean {
  return closing;
}

/**
 * Paint the overlay. Called twice per quit on Windows and Linux: the close button calls
 * it optimistically so the first paint does not wait on the IPC round trip, then Rust's
 * own app-closing arrives. The second call is a no-op.
 */
export function markAppClosing(): void {
  setAppClosing(true);
}

/** The quit was declined, so the app stays and the overlay has to go. */
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
