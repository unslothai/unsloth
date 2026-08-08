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

/**
 * How long the reap may run before the overlay offers a way out. Windows is the platform
 * that gets here, and its bounded worst case is about 18s: `stop_spawned_backend` spends up
 * to 2 liveness requests and 2 shutdown requests at `LOCAL_HTTP_TIMEOUT` (2s each), then
 * waits twice for the child to exit (5s each) either side of the CTRL_BREAK. The 5s
 * graceful waits the installer and the updater take are `#[cfg(unix)]`, so Windows skips
 * them and force kills instead.
 *
 * This sits well past that 18s rather than just over it, because the button abandons the
 * reap: offered during a shutdown that is merely slow and about to succeed, it turns a wait
 * into a half-finished teardown. Anything still going at 35s is wedged, not slow. The
 * overlay covers the titlebar, so without the button the window has no controls and nothing
 * to click.
 */
export const FORCE_QUIT_AFTER_MS = 35_000;

/**
 * Last resort for that wedge. Rust exits the process without finishing the reap, so this
 * normally never settles. Swallows a failure rather than rejecting: there is nothing left
 * to fall back to, and an unhandled rejection is noise in the one moment the user is
 * already stuck.
 */
export async function forceQuit(): Promise<void> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("force_quit");
  } catch (error) {
    console.error("Force quit failed:", error);
  }
}

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
