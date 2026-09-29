// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Why the archived media view must release a thumbnail when its row goes away.
//
// Eviction there protects the ids currently on screen, so a row that is restored or deleted has to
// be dropped from that protected set AND from the cache. Its element unmounts without the observer
// reporting it, so a stale id would stay "visible" forever and shield its blob from every later
// prune, walking the cache past its budget one restore at a time.

import assert from "node:assert/strict";
import test from "node:test";

import { BlobUrlCache } from "../src/lib/blob-url-cache.ts";

// The cache revokes on eviction; node has no object-URL registry, so record the calls instead.
const revoked: string[] = [];
(globalThis as { URL: { revokeObjectURL: (url: string) => void } }).URL = {
  ...URL,
  revokeObjectURL: (url: string) => {
    revoked.push(url);
  },
} as never;

test("a protected id survives eviction, so a stale one would pin its bytes forever", () => {
  const cache = new BlobUrlCache(100);
  cache.set("stale", "blob:stale", 80);
  cache.set("other", "blob:other", 80);

  // "stale" is still in the protected set, so prune frees the other entry instead and the cache
  // stays over budget. That is correct while the row is on screen, and a leak once it is gone.
  assert.deepEqual(cache.prune(new Set(["stale"])), ["other"]);
  assert.equal(cache.has("stale"), true);
  assert.ok(revoked.includes("blob:other"));
});

test("deleting the dropped row's entry releases its blob and restores the budget", () => {
  const cache = new BlobUrlCache(100);
  cache.set("dropped", "blob:dropped", 80);
  cache.set("kept", "blob:kept", 10);

  // What dropRow now does: release the entry outright rather than leaving it protected.
  assert.equal(cache.delete("dropped"), true);
  assert.ok(revoked.includes("blob:dropped"));
  // With those bytes back, a later prune has no reason to evict what is still on screen.
  assert.deepEqual(cache.prune(new Set(["kept"])), []);
  assert.equal(cache.has("kept"), true);
});

test("deleting an id that was never cached is a no-op", () => {
  // dropRow runs for videos too, where the thumbnail is a signed link and never enters the cache.
  const cache = new BlobUrlCache(100);
  assert.equal(cache.delete("never-cached"), false);
});
