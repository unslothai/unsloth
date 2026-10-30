// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
} from "../../../lib/desktop-update-activity.ts";
import { PinnedChatsBlockedError } from "./pinned-chats-sync.ts";

// Judged when the request fails: the queue reports after a recovery read, by when the flag may be gone.
const failedUnderUpdate = new WeakSet<object>();

function unreachable(error: unknown, down: boolean): boolean {
  return (
    isSilencedDesktopUpdateFailure(error, down) ||
    // The pins API aborts at 10s.
    (down &&
      error instanceof Error &&
      (error.name === "AbortError" || error.name === "TimeoutError"))
  );
}

/** Called by the pins API for every failed request, with the flag as it was at issue. */
export function notePinRequestFailure(error: unknown, downWhenIssued: boolean): void {
  const down = downWhenIssued || isBackendDownForDesktopUpdate();
  if (typeof error === "object" && error !== null && unreachable(error, down)) {
    failedUnderUpdate.add(error);
  }
}

/** A pin sync failure caused by the backend the desktop update stopped: no toast for it. */
export function isPinFailureUnderDesktopUpdate(error: unknown): boolean {
  if (error instanceof PinnedChatsBlockedError) {
    return error.cause === undefined
      ? isBackendDownForDesktopUpdate()
      : isPinFailureUnderDesktopUpdate(error.cause);
  }
  if (typeof error === "object" && error !== null && failedUnderUpdate.has(error)) {
    return true;
  }
  return unreachable(error, isBackendDownForDesktopUpdate());
}
