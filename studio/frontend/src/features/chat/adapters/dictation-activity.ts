// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tracks whether an in-app composer dictation is running. The global voice
// typing chord uses this to drive the built-in dictation (respecting toggle,
// hybrid and hold modes) instead of opening the system overlay while the
// Unsloth window is focused.

const activeOwners = new Set<symbol>();

/** Each composer owns a token so unmounting one never clears another's state. */
export function setInAppDictationActive(owner: symbol, active: boolean): void {
  if (active) activeOwners.add(owner);
  else activeOwners.delete(owner);
}

export function isInAppDictationActive(): boolean {
  return activeOwners.size > 0;
}
