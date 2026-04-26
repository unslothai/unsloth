// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Global gate flipped on while the user-initiated server restart is in
 * flight (between `POST /api/settings/server/restart` and the new server
 * answering /api/health).
 *
 * Why this exists: every in-flight `authFetch` during the ~5–15s window
 * fails with TypeError, which the auth layer translates into the loud
 * "Studio isn't running -- please relaunch it." banner + toasts, and a
 * follow-up 401 can trigger `redirectToAuth()` which hard-navigates to
 * /login and tears down the settings modal.
 *
 * With this gate set, callers can park their requests on `awaitRestart()`
 * and resume against the new server, and the auth layer can suppress
 * navigation while the server is intentionally bouncing.
 */

let pending: Promise<void> | null = null;
let release: (() => void) | null = null;

export function beginRestart(): () => void {
  if (pending) {
    return release ?? (() => {});
  }
  pending = new Promise<void>((resolve) => {
    release = () => {
      release = null;
      pending = null;
      resolve();
    };
  });
  // release is set above by the executor (synchronous), but TS can't
  // narrow that — assert here so the call site gets a non-null return.
  // biome-ignore lint/style/noNonNullAssertion: executor is synchronous
  return release!;
}

export function isRestarting(): boolean {
  return pending !== null;
}

/**
 * Park until the restart gate releases, or until the safety timeout
 * elapses. The timeout exists purely as a deadlock guard: if the gate
 * is never released (e.g. an unexpected throw past the `finally` that
 * owns it), callers still recover instead of spinning forever.
 */
export async function awaitRestart(timeoutMs = 120_000): Promise<void> {
  if (!pending) {
    return;
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  const safety = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, timeoutMs);
  });
  try {
    await Promise.race([pending, safety]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}
