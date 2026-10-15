// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type PageStorageArea = "local" | "session" | "cookie";
export type PageStorage = Record<PageStorageArea, Array<[string, string]>>;

// Per area and site, in characters, so caches like a site's script store aren't kept.
const PAGE_STORAGE_QUOTA = 128 * 1024;
const MAX_TOTAL_CHARS = 4 * 1024 * 1024;

type Store = { items: Map<string, string>; chars: number };

// In memory only, so nothing outlives the session.
const stores = new Map<string, Store>();
let totalChars = 0;

const storeKey = (origin: string, tabId: string, area: PageStorageArea) =>
  area === "session" ? `session ${tabId} ${origin}` : `${area} ${origin}`;

export function pageOrigin(url: string | null): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.origin : null;
  } catch {
    return null;
  }
}

export function readPageStorage(origin: string, tabId: string): PageStorage {
  const entries = (area: PageStorageArea) => Array.from(stores.get(storeKey(origin, tabId, area))?.items ?? []);
  return { local: entries("local"), session: entries("session"), cookie: entries("cookie") };
}

export type PageStorageWrite = {
  origin: string;
  tabId: string;
  area: PageStorageArea;
  key: string | null;
  value: string | null;
  from?: unknown;
  url?: string | null;
};

const writeListeners = new Set<(write: PageStorageWrite) => void>();

/** Called after each write with what the store kept, so other open pages of the site can follow. */
export function onPageStorageWrite(listener: (write: PageStorageWrite) => void): () => void {
  writeListeners.add(listener);
  return () => writeListeners.delete(listener);
}

/** One change a page made: a null value removes the key, a null key clears the area. */
export function writePageStorage(
  origin: string,
  tabId: string,
  area: PageStorageArea,
  key: string | null,
  value: string | null,
  from?: unknown,
  url?: string | null,
): void {
  save(origin, tabId, area, key, value);
  const kept = key === null ? null : (stores.get(storeKey(origin, tabId, area))?.items.get(key) ?? null);
  for (const listener of writeListeners) listener({ origin, tabId, area, key, value: kept, from, url });
}

function evict(): void {
  for (const [other, { chars: otherChars }] of stores) {
    if (totalChars <= MAX_TOTAL_CHARS) break;
    stores.delete(other);
    totalChars -= otherChars;
  }
}

function save(origin: string, tabId: string, area: PageStorageArea, key: string | null, value: string | null): void {
  const id = storeKey(origin, tabId, area);
  const store = stores.get(id) ?? { items: new Map<string, string>(), chars: 0 };
  stores.delete(id);
  totalChars -= store.chars;
  if (key === null) return;
  const old = store.items.get(key);
  store.chars -= old === undefined ? 0 : key.length + old.length;
  if (value !== null && store.chars + key.length + value.length <= PAGE_STORAGE_QUOTA) {
    store.items.set(key, value);
    store.chars += key.length + value.length;
  } else {
    store.items.delete(key);
  }
  if (store.items.size === 0) return;
  stores.set(id, store);
  totalChars += store.chars;
  evict();
}

/** A closed tab's sessionStorage can't be reached again. */
export function dropTabPageStorage(tabId: string): void {
  const prefix = `session ${tabId} `;
  for (const [id, store] of stores) {
    if (!id.startsWith(prefix)) continue;
    stores.delete(id);
    totalChars -= store.chars;
  }
}

/** A duplicated tab starts with a copy of the original's sessionStorage, as in a browser. */
export function copyTabPageStorage(fromTabId: string, toTabId: string): void {
  const prefix = `session ${fromTabId} `;
  for (const [id, store] of Array.from(stores)) {
    if (!id.startsWith(prefix)) continue;
    stores.set(`session ${toTabId} ${id.slice(prefix.length)}`, { items: new Map(store.items), chars: store.chars });
    totalChars += store.chars;
  }
  evict();
}

const clearListeners = new Set<() => void>();

/** Called on clearPageStorage, so open pages can drop their own copies. */
export function onPageStorageClear(listener: () => void): () => void {
  clearListeners.add(listener);
  return () => clearListeners.delete(listener);
}

export function clearPageStorage(): void {
  stores.clear();
  totalChars = 0;
  for (const listener of clearListeners) listener();
}
