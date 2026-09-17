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
