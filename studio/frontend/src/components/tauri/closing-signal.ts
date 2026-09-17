// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Rust asks for the closing overlay with this, from the quit thread in main.rs, once the
 * quit confirmations have passed and only the backend reap is left. Windows is the only
 * platform that asks: macOS closes to the tray, and the Linux reap never showed the freeze
 * this covers. Nothing here is platform-aware, so an overlay is whatever Rust asks for.
 */
export const APP_CLOSING_EVENT = "app-closing";

/** And takes it back down with this, for a quit that never reaches the exit. */
export const APP_CLOSING_CANCELLED_EVENT = "app-closing-cancelled";

// Module state rather than React state: the events arrive on the backend hook and the
// overlay renders from the provider. A webview reload resets it, which fails open: no
// quit survives the reload either.
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
