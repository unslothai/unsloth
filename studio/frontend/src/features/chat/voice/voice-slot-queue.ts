// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Serialize slot mutations so teardown cannot finish before an earlier load. */
export function createVoiceSlotQueue() {
  let pending: Promise<unknown> = Promise.resolve();
  return function enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const next = pending.then(operation, operation);
    pending = next.catch(() => {});
    return next;
  };
}

// Survives Chat remounts: an old page's unload must precede the new page's load.
export const queueVoiceSlotMutation = createVoiceSlotQueue();
