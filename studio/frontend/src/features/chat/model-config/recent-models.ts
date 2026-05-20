// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const STORAGE_KEY = "unsloth_recent_models";
const MAX_ENTRIES = 32;

export interface RecentModelKey {
  id: string;
  ggufVariant?: string | null;
}

interface StoredEntry {
  id: string;
  variant: string;
  ts: number;
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function entryKey(id: string, variant?: string | null): string {
  return `${id}::${variant ?? ""}`;
}

function readAll(): StoredEntry[] {
  if (!canUseStorage()) return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (entry): entry is StoredEntry =>
          entry != null &&
          typeof entry === "object" &&
          typeof (entry as StoredEntry).id === "string" &&
          typeof (entry as StoredEntry).variant === "string" &&
          typeof (entry as StoredEntry).ts === "number",
      )
      .slice(0, MAX_ENTRIES);
  } catch {
    return [];
  }
}

function writeAll(entries: StoredEntry[]): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {
    return;
  }
}

export function touchRecentModel(key: RecentModelKey): void {
  const entries = readAll().filter(
    (entry) => entryKey(entry.id, entry.variant) !== entryKey(key.id, key.ggufVariant),
  );
  entries.unshift({
    id: key.id,
    variant: key.ggufVariant ?? "",
    ts: Date.now(),
  });
  writeAll(entries);
}

export interface RecentRank {
  rank: (id: string, ggufVariant?: string | null) => number;
  hasAny: boolean;
}

export function buildRecentRank(): RecentRank {
  const entries = readAll();
  if (entries.length === 0) {
    return { rank: () => Number.POSITIVE_INFINITY, hasAny: false };
  }
  const idIndex = new Map<string, number>();
  const fullIndex = new Map<string, number>();
  entries.forEach((entry, idx) => {
    const full = entryKey(entry.id, entry.variant);
    fullIndex.set(full, idx);
    if (!idIndex.has(entry.id)) idIndex.set(entry.id, idx);
  });
  return {
    hasAny: true,
    rank: (id, variant) => {
      const full = entryKey(id, variant);
      const exact = fullIndex.get(full);
      if (exact != null) return exact;
      const any = idIndex.get(id);
      if (any != null) return any + 0.5;
      return Number.POSITIVE_INFINITY;
    },
  };
}
