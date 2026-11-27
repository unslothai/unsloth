// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Filled by each mounted thread's TurnNavigator, which owns the detach-aware jump. */
export const jumpListeners = new Set<(messageId: string) => void>();

/** Jumps the visible thread to a message, for callers outside the thread (the Files menu). */
export function requestJumpToMessage(messageId: string): void {
  for (const listener of jumpListeners) {
    listener(messageId);
  }
}
