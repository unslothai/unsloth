// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
} from "../../../lib/desktop-update-activity.ts";
import { PinnedChatsBlockedError } from "./pinned-chats-sync.ts";
import {
  ChatSettingsRequestError,
  isAbortOrTimeout,
  isTerminalSettingsRejection,
} from "./settings-retry.ts";

// Judged when the request fails: the queue reports after a recovery read, by when the flag may be gone.
const failedUnderUpdate = new WeakSet<object>();

function unreachable(error: unknown, down: boolean): boolean {
  return (
    isSilencedDesktopUpdateFailure(error, down) ||
    // The pins API aborts at 10s.
    (down && isAbortOrTimeout(error))
  );
}

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

/** Transient failures the journal retries on the next refresh: no toast for them either. */
export function isExpectedPinSyncInterruption(error: unknown): boolean {
  if (error instanceof PinnedChatsBlockedError) {
    return error.cause !== undefined && isExpectedPinSyncInterruption(error.cause);
  }
  if (isAbortOrTimeout(error)) return true;
  if ((error as { unslothTransportFailure?: unknown } | null)?.unslothTransportFailure === true) {
    return true;
  }
  return error instanceof ChatSettingsRequestError && !isTerminalSettingsRejection(error);
}

/** Only a pin change the sync discarded is news; the journal retries every other failure. */
export function shouldReportPinSyncFailure(error: unknown, editLost: boolean): boolean {
  return (
    editLost && !isPinFailureUnderDesktopUpdate(error) && !isExpectedPinSyncInterruption(error)
  );
}
