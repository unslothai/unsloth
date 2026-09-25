// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thumbnails are auth-fetched blobs, so the browser cache cannot hold them. The most recent stay as
// object URLs within a count and a byte budget, so a revisit paints at once. An entry a card still
// shows (or waits on) is never evicted, since its URL would be revoked under it.

const MAX_CACHED_URLS = 300;
const MAX_CACHED_BYTES = 128 * 1024 * 1024;

interface Entry {
  url: Promise<string>;
  bytes: number;
  users: number;
}

const objectUrls = new Map<string, Entry>();
let cachedBytes = 0;

function revoke(entry: Entry): void {
  void entry.url.then((url) => URL.revokeObjectURL(url), () => {});
}

export function clearCachedObjectUrls(): void {
  objectUrls.forEach(revoke);
  objectUrls.clear();
  cachedBytes = 0;
}

function evict(key: string): void {
  const entry = objectUrls.get(key);
  if (!entry) return;
  objectUrls.delete(key);
  cachedBytes -= entry.bytes;
  revoke(entry);
}

function trim(): void {
  for (const [key, entry] of objectUrls) {
    if (objectUrls.size <= MAX_CACHED_URLS && cachedBytes <= MAX_CACHED_BYTES) return;
    if (entry.users === 0) evict(key);
  }
}

export function acquireObjectUrl(
  key: string,
  load: () => Promise<Blob>,
): { url: Promise<string>; release: () => void } {
  let entry = objectUrls.get(key);
  if (entry) {
    objectUrls.delete(key);
  } else {
    const created: Entry = { url: Promise.resolve(""), bytes: 0, users: 0 };
    created.url = load().then((blob) => {
      if (objectUrls.get(key) === created) {
        created.bytes = blob.size;
        cachedBytes += blob.size;
      }
      return URL.createObjectURL(blob);
    });
    created.url.then(trim, () => objectUrls.get(key) === created && evict(key));
    entry = created;
  }
  objectUrls.set(key, entry);
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

export function cachedObjectUrlCount(): number {
  return objectUrls.size;
}
