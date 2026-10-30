// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isBackendDownForDesktopUpdate,
  isSilencedDesktopUpdateFailure,
} from "../../../lib/desktop-update-activity.ts";
import { PinnedChatsBlockedError } from "./pinned-chats-sync.ts";

/** A pin sync failure caused by the backend the desktop update stopped: no toast under its screen. */
export function isPinFailureUnderDesktopUpdate(error: unknown): boolean {
  const down = isBackendDownForDesktopUpdate();
  return (
    isSilencedDesktopUpdateFailure(error, down) ||
    (down &&
      // The pins API aborts at 10s; a blocked queue entry waits on the request that failed.
      (error instanceof PinnedChatsBlockedError ||
        (error instanceof Error &&
          (error.name === "AbortError" || error.name === "TimeoutError"))))
  );
}
