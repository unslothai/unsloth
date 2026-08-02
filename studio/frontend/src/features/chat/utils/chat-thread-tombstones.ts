// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Tombstones mask deleted threads in the Dexie read fallback. Each carries a
 * `deletedAt` so old entries can be GC'd, keeping localStorage bounded. Reads
 * accept both the legacy plain-string format and the {id, deletedAt} tuple.
 */

interface Tombstone {
  id: string;
  deletedAt: number;
}

const TOMBSTONES_KEY = "unsloth_chat_deleted_thread_ids";
const TOMBSTONE_MAX_AGE_MS = 90 * 24 * 60 * 60 * 1000; // 90 days
const TOMBSTONE_MAX_COUNT = 5000;

const deletedThreads = new Map<string, Tombstone>();

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function nowMs(): number {
  return Date.now();
}

function isTombstone(value: unknown): value is Tombstone {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Tombstone).id === "string" &&
    typeof (value as Tombstone).deletedAt === "number"
  );
}

function loadTombstones(): Tombstone[] {
  if (!canUseStorage()) return [];
  try {
    const raw = JSON.parse(localStorage.getItem(TOMBSTONES_KEY) ?? "[]");
    if (!Array.isArray(raw)) return [];
    const now = nowMs();
    const out: Tombstone[] = [];
    for (const item of raw) {
      if (typeof item === "string") {
        // Legacy plain-string format from pre-B6 installs.
        out.push({ id: item, deletedAt: now });
      } else if (isTombstone(item)) {
        out.push(item);
      }
    }
    return out;
  } catch {
    return [];
  }
}

function gc(): void {
  const cutoff = nowMs() - TOMBSTONE_MAX_AGE_MS;
  for (const [id, t] of deletedThreads) {
    if (t.deletedAt < cutoff) deletedThreads.delete(id);
  }
  // Cap size: drop oldest if we exceed the limit (e.g. a bulk thread clear).
  if (deletedThreads.size > TOMBSTONE_MAX_COUNT) {
    const sorted = Array.from(deletedThreads.entries()).sort(
      (a, b) => a[1].deletedAt - b[1].deletedAt,
    );
    const drop = sorted.slice(0, deletedThreads.size - TOMBSTONE_MAX_COUNT);
    for (const [id] of drop) deletedThreads.delete(id);
  }
}

function persist(): void {
  if (!canUseStorage()) return;
  try {
    const arr = Array.from(deletedThreads.values());
    localStorage.setItem(TOMBSTONES_KEY, JSON.stringify(arr));
  } catch {
    // ignore quota / serialization failures
  }
}

for (const t of loadTombstones()) {
  deletedThreads.set(t.id, t);
}
gc();

export function markChatThreadDeleted(threadId: string): void {
  deletedThreads.set(threadId, { id: threadId, deletedAt: nowMs() });
  gc();
  persist();
}

export function markChatThreadsDeleted(threadIds: Iterable<string>): void {
  const now = nowMs();
  for (const id of threadIds) {
    deletedThreads.set(id, { id, deletedAt: now });
  }
  gc();
  persist();
}

export function isChatThreadDeleted(threadId: string): boolean {
  return deletedThreads.has(threadId);
}

/** Rollback support: drop tombstones when a backend delete fails. */
export function removeChatThreadTombstones(threadIds: Iterable<string>): void {
  let changed = false;
  for (const id of threadIds) {
    if (deletedThreads.delete(id)) changed = true;
  }
  if (changed) persist();
}

export function __resetChatThreadTombstonesForTests(): void {
  deletedThreads.clear();
}
