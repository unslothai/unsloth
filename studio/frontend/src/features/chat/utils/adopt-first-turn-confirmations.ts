// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A first turn that parks before its chat has an id files the call under threadId "". Hands those
 *  calls the id the run is adopted under, or null when there are none. */
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
