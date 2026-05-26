// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";
import { modelStorageKey, normalizeModelIdentity } from "./model-identity.ts";

const STORAGE_KEY = "unsloth_recent_models";
const MAX_ENTRIES = 32;
const MAX_VARIANTS_PER_MODEL = 4;

export interface RecentModelKey {
  id: string;
  ggufVariant?: string | null;
}

export type RemoveRecentModelTarget =
  | "all"
  | {
      variant: string | null;
    };

interface StoredEntry {
  id: string;
  variant: string;
  ts: number;
}

let recentVersion = 0;
let storageSyncReady = false;
const listeners = new Set<() => void>();

function byStoredRecency(
  a: { entry: StoredEntry; index: number },
  b: { entry: StoredEntry; index: number },
): number {
  const tsCmp = b.entry.ts - a.entry.ts;
  return tsCmp !== 0 ? tsCmp : a.index - b.index;
}

function normalizeEntries(entries: StoredEntry[]): StoredEntry[] {
  const seen = new Set<string>();
  const perModel = new Map<string, number>();
  const normalized: StoredEntry[] = [];
  for (const { entry } of entries
    .map((entry, index) => ({ entry, index }))
    .filter(({ entry }) => Number.isFinite(entry.ts))
    .sort(byStoredRecency)) {
    const modelIdentity = normalizeModelIdentity(entry.id);
    if (!modelIdentity) {
      continue;
    }
    const key = modelStorageKey(entry.id, entry.variant);
    if (seen.has(key)) {
      continue;
    }
    const modelCount = perModel.get(modelIdentity) ?? 0;
    if (modelCount >= MAX_VARIANTS_PER_MODEL) {
      continue;
    }
    seen.add(key);
    perModel.set(modelIdentity, modelCount + 1);
    normalized.push(entry);
    if (normalized.length >= MAX_ENTRIES) {
      break;
    }
  }
  return normalized;
}

function emitRecentModelsChanged(): void {
  recentVersion += 1;
  for (const listener of [...listeners]) {
    listener();
  }
}

function ensureStorageSync(): void {
  if (storageSyncReady || typeof window === "undefined") {
    return;
  }
  storageSyncReady = true;
  window.addEventListener("storage", (event) => {
    if (event.key === STORAGE_KEY) {
      emitRecentModelsChanged();
    }
  });
}

function subscribeRecentModels(listener: () => void): () => void {
  ensureStorageSync();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getRecentSnapshot(): number {
  return recentVersion;
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function readAll(): StoredEntry[] {
  if (!canUseStorage()) {
    return [];
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return normalizeEntries(
      parsed.filter(
        (entry): entry is StoredEntry =>
          entry != null &&
          typeof entry === "object" &&
          typeof (entry as StoredEntry).id === "string" &&
          typeof (entry as StoredEntry).variant === "string" &&
          typeof (entry as StoredEntry).ts === "number",
      ),
    );
  } catch {
    return [];
  }
}

function writeAll(entries: StoredEntry[]): void {
  if (!canUseStorage()) {
    return;
  }
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(normalizeEntries(entries)),
    );
    emitRecentModelsChanged();
  } catch (err) {
    console.warn("Failed to persist recent models:", err);
  }
}

export function touchRecentModel(key: RecentModelKey): void {
  const id = key.id.trim();
  if (!id) {
    return;
  }
  const storedId = normalizeModelIdentity(id);
  const nextKey = modelStorageKey(storedId, key.ggufVariant);
  const entries = readAll().filter(
    (entry) => modelStorageKey(entry.id, entry.variant) !== nextKey,
  );
  entries.unshift({
    id: storedId,
    variant: key.ggufVariant ?? "",
    ts: Date.now(),
  });
  writeAll(entries);
}

export function removeRecentModel(
  id: string,
  target: RemoveRecentModelTarget,
): void {
  const removeAllForModel = target === "all";
  const targetIdentity = normalizeModelIdentity(id);
  const targetKey = removeAllForModel
    ? null
    : modelStorageKey(id, target.variant);
  const entries = readAll().filter((entry) => {
    if (normalizeModelIdentity(entry.id) !== targetIdentity) {
      return true;
    }
    if (removeAllForModel) {
      return false;
    }
    return modelStorageKey(entry.id, entry.variant) !== targetKey;
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
    const modelIdentity = normalizeModelIdentity(entry.id);
    const full = modelStorageKey(entry.id, entry.variant);
    if (!fullIndex.has(full)) {
      fullIndex.set(full, idx);
    }
    if (!idIndex.has(modelIdentity)) {
      idIndex.set(modelIdentity, idx);
    }
  });
  return {
    hasAny: true,
    rank: (id, variant) => {
      const modelIdentity = normalizeModelIdentity(id);
      const full = modelStorageKey(id, variant);
      const exact = fullIndex.get(full);
      if (exact != null) {
        return exact;
      }
      const any = idIndex.get(modelIdentity);
      if (any != null) {
        return any + 0.5;
      }
      return Number.POSITIVE_INFINITY;
    },
  };
}

export function useRecentRank(): RecentRank {
  useSyncExternalStore(
    subscribeRecentModels,
    getRecentSnapshot,
    getRecentSnapshot,
  );
  return buildRecentRank();
}
