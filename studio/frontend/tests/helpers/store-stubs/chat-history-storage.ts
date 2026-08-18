// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The thread row, in memory. The store PATCHes a chat's snapshot through
// updateStoredChatThread, so recording the calls is how a test sees what a chat
// was actually saved as, and `rows` is what a reopen would read back.

export interface RecordedThreadWrite {
  threadId: string;
  /** A replacement write: the whole snapshot. */
  settings?: Record<string, unknown>;
  /** A merge write: only the fields the user touched. */
  settingsPatch?: Record<string, unknown>;
  settingsSeq?: number;
  settingsWriter?: string;
}

export const threadRows = {
  writes: [] as RecordedThreadWrite[],
  /** The row as the last write left it, per thread. */
  rows: new Map<string, Record<string, unknown>>(),
  /** Reject the next write, to exercise the failure arm. */
  failNext: false,
  reset(): void {
    threadRows.writes.length = 0;
    threadRows.rows.clear();
    threadRows.failNext = false;
  },
  /** Every write recorded for one chat, oldest first. */
  writesFor(threadId: string): RecordedThreadWrite[] {
    return threadRows.writes.filter((write) => write.threadId === threadId);
  },
};

export async function updateStoredChatThread(
  threadId: string,
  update: {
    settings?: Record<string, unknown>;
    settingsPatch?: Record<string, unknown>;
    settingsSeq?: number;
    settingsWriter?: string;
  },
  _options?: { signal?: AbortSignal },
): Promise<void> {
  if (threadRows.failNext) {
    threadRows.failNext = false;
    throw new Error("stubbed thread write failure");
  }
  threadRows.writes.push({ threadId, ...update });
  const row = threadRows.rows.get(threadId) ?? {};
  // A replacement write replaces settings_json; a patch merges into it. Same as
  // PATCH /api/chat/threads/{id} does server side.
  threadRows.rows.set(
    threadId,
    update.settings !== undefined
      ? { ...update.settings }
      : { ...row, ...update.settingsPatch },
  );
}

export async function ensureStoredChatThread(): Promise<void> {}

export async function getStoredChatThread(
  threadId: string,
): Promise<{ settings: Record<string, unknown> } | null> {
  const row = threadRows.rows.get(threadId);
  return row ? { settings: row } : null;
}
