// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The desktop update stops the backend on purpose while the app stays mounted under the update screen.
let backendDownForUpdate = false;

export function setBackendDownForDesktopUpdate(down: boolean): void {
  backendDownForUpdate = down;
}

export function isBackendDownForDesktopUpdate(): boolean {
  return backendDownForUpdate;
}

/**
 * Whether a failed background read should stay quiet: the request never reached the backend, and
 * the update stopped it. Latch the flag when the request is ISSUED and pass it here, because the
 * rejection lands up to ~20s later (the Tauri GET ladder is 10.5s and check_backend_present adds a
 * 10s budget) and Skip & Restart can drop the update screen inside that window.
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
