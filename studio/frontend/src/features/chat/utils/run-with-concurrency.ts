// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Run `task` over `items`, at most `limit` at a time, in order. */
export async function runWithConcurrency<T>(
  items: readonly T[],
  limit: number,
  task: (item: T) => Promise<void>,
): Promise<void> {
  const lanes = Math.max(1, Math.min(limit, items.length));
  let next = 0;
  await Promise.all(
    Array.from({ length: lanes }, async () => {
      while (next < items.length) {
        const item = items[next];
        next += 1;
        await task(item as T);
      }
    }),
  );
}
