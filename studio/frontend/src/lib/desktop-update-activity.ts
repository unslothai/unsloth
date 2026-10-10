// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The update stops the backend while the app stays mounted under the update screen.
let backendDownForUpdate = false;

export function isBackendDownForDesktopUpdate(): boolean {
  return backendDownForUpdate;
}

/** Holds the flag until `resync` settles, since recovery spawns a backend without waiting. */
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

/** `downWhenIssued` is latched at issue: the rejection can land ~20s later. */
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
