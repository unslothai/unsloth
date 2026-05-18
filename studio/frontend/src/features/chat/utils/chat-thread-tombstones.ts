// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const deletedThreadIds = new Set<string>();
const TOMBSTONES_KEY = "unsloth_chat_deleted_thread_ids";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function loadDeletedThreadIds(): string[] {
  if (!canUseStorage()) return [];
  try {
    const parsed = JSON.parse(localStorage.getItem(TOMBSTONES_KEY) ?? "[]");
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

function persistDeletedThreadIds(): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(TOMBSTONES_KEY, JSON.stringify([...deletedThreadIds]));
  } catch {
    // ignore
  }
}

for (const threadId of loadDeletedThreadIds()) {
  deletedThreadIds.add(threadId);
}

export function markChatThreadDeleted(threadId: string): void {
  deletedThreadIds.add(threadId);
  persistDeletedThreadIds();
}

export function markChatThreadsDeleted(threadIds: Iterable<string>): void {
  for (const threadId of threadIds) {
    deletedThreadIds.add(threadId);
  }
  persistDeletedThreadIds();
}

export function isChatThreadDeleted(threadId: string): boolean {
  return deletedThreadIds.has(threadId);
}
