// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thumbnails are auth-fetched blobs, so the browser cache cannot hold them. Keep the most recent
// ones as object URLs, within a count and a byte budget; a revisit of the page then paints
// instantly instead of refetching. An entry a card is still showing (or waiting on) is never
// evicted, since its URL would be revoked under it; the budget catches up once it is released.

const MAX_CACHED_URLS = 300;
const MAX_CACHED_BYTES = 128 * 1024 * 1024;

interface Entry {
  url: Promise<string>;
  bytes: number;
  /** Cards holding the URL; only an entry nobody holds may go. */
  users: number;
}

const objectUrls = new Map<string, Entry>();
let cachedBytes = 0;

/** Drop every cached URL, so a sign-out leaves nothing of the last account's files. */
export function clearCachedObjectUrls(): void {
  for (const { url } of objectUrls.values()) {
    void url.then((stale) => URL.revokeObjectURL(stale), () => {});
  }
  objectUrls.clear();
  cachedBytes = 0;
}

function evict(key: string): void {
  const entry = objectUrls.get(key);
  if (!entry) return;
  objectUrls.delete(key);
  cachedBytes -= entry.bytes;
  void entry.url.then((stale) => URL.revokeObjectURL(stale), () => {});
}

/** Oldest first, skipping what is in use, until both budgets hold again. */
function trim(): void {
  for (const [key, entry] of objectUrls) {
    if (objectUrls.size <= MAX_CACHED_URLS && cachedBytes <= MAX_CACHED_BYTES) return;
    if (entry.users === 0) evict(key);
  }
}

/**
 * The object URL cached under `key`, loading it with `load` on a miss. Call `release` once the URL
 * is no longer shown; until then it stays valid.
 */
export function acquireObjectUrl(
  key: string,
  load: () => Promise<Blob>,
): { url: Promise<string>; release: () => void } {
  let entry = objectUrls.get(key);
  if (entry) {
    // Most recently used goes last, so eviction takes what has gone unseen longest.
    objectUrls.delete(key);
    objectUrls.set(key, entry);
  } else {
    const created: Entry = { url: Promise.resolve(""), bytes: 0, users: 0 };
    created.url = load().then((blob) => {
      if (objectUrls.get(key) === created) {
        created.bytes = blob.size;
        cachedBytes += blob.size;
      }
      return URL.createObjectURL(blob);
    });
    created.url.then(trim, () => {
      if (objectUrls.get(key) === created) evict(key);
    });
    objectUrls.set(key, created);
    entry = created;
  }
  const held = entry;
  held.users += 1;
  trim();
  let released = false;
  return {
    url: held.url,
    release: () => {
      if (released) return;
      released = true;
      held.users -= 1;
      if (held.users === 0) trim();
    },
  };
}

/** For tests: how many URLs are cached. */
export function cachedObjectUrlCount(): number {
  return objectUrls.size;
}
