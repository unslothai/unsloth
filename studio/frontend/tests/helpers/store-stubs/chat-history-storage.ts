// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface RecordedThreadWrite {
  threadId: string;
  settings?: Record<string, unknown>;
  settingsPatch?: Record<string, unknown>;
  settingsSeq?: number;
  settingsWriter?: string;
}

export const threadRows = {
  writes: [] as RecordedThreadWrite[],
  rows: new Map<string, Record<string, unknown>>(),
  failNext: false,
  reset(): void {
    threadRows.writes.length = 0;
    threadRows.rows.clear();
    threadRows.failNext = false;
  },
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
  // Replacement replaces settings_json, patch merges into it, as the server's PATCH does.
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
