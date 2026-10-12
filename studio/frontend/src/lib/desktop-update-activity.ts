// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The desktop update stops the backend on purpose while the app stays mounted under the update screen.
let backendDownForUpdate = false;

export function isBackendDownForDesktopUpdate(): boolean {
  return backendDownForUpdate;
}

/**
 * Tracks the update screen and returns the effect cleanup. Leaving it holds the flag until `resync`
 * settles: Skip & Restart and the shell-failure recovery spawn a fresh backend without waiting for it
 * to answer.
 */
export function followDesktopUpdateScreen(
  isUpdating: boolean,
  wasUpdating: boolean,
  resync: () => Promise<void>,
): () => void {
  const leaving = wasUpdating && !isUpdating;
  backendDownForUpdate = isUpdating || leaving;
  let active = true;
  if (leaving) {
    void resync().finally(() => {
      if (active) backendDownForUpdate = false;
    });
  }
  return () => {
    active = false;
    backendDownForUpdate = false;
  };
}

/**
 * Whether a failed read never reached a backend the update stopped. `downWhenIssued` is latched when
 * the request is issued: its rejection can land ~20s later, after the update screen has gone.
 */
export function isSilencedDesktopUpdateFailure(
  error: unknown,
  downWhenIssued: boolean,
): boolean {
  if (!downWhenIssued && !backendDownForUpdate) return false;
  return (
    (error as { unslothTransportFailure?: boolean } | null)
      ?.unslothTransportFailure === true
  );
}
