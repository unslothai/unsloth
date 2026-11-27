// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Re-reads until no unload is pending: an answer taken mid-teardown may be stale, and a failed unload leaves it live. */
export async function readAfterUnload<T>(
  pending: { current: Promise<unknown> | null },
  read: () => Promise<T>,
): Promise<T> {
  for (;;) {
    const value = await read();
    const unload = pending.current;
    if (!unload) return value;
    await unload.catch(() => undefined);
  }
}
