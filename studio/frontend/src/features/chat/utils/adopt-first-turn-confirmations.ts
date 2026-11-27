// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Re-keys calls filed under threadId "" to adopted run's id; null when there are none. */
export function adoptFirstTurnConfirmations<T extends { threadId: string }>(
  confirmations: Readonly<Record<string, T>>,
  threadId: string,
): Record<string, T> | null {
  let next: Record<string, T> | null = null;
  for (const [id, entry] of Object.entries(confirmations)) {
    if (entry.threadId !== "") continue;
    next ??= { ...confirmations };
    next[id] = { ...entry, threadId };
  }
  return next;
}
