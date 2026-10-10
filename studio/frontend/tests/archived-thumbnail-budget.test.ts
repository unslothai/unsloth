// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A removed row's element unmounts without the observer reporting it, so its id must be dropped
// from the protected set and the cache, or it shields its blob from pruning forever.

import assert from "node:assert/strict";
import test from "node:test";

import { BlobUrlCache } from "../src/lib/blob-url-cache.ts";

// Node has no object-URL registry, so record revoke calls instead.
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

  assert.deepEqual(cache.prune(new Set(["stale"])), ["other"]);
  assert.equal(cache.has("stale"), true);
  assert.ok(revoked.includes("blob:other"));
});

test("deleting the dropped row's entry releases its blob and restores the budget", () => {
  const cache = new BlobUrlCache(100);
  cache.set("dropped", "blob:dropped", 80);
  cache.set("kept", "blob:kept", 10);

  assert.equal(cache.delete("dropped"), true);
  assert.ok(revoked.includes("blob:dropped"));
  assert.deepEqual(cache.prune(new Set(["kept"])), []);
  assert.equal(cache.has("kept"), true);
});

test("deleting an id that was never cached is a no-op", () => {
  const cache = new BlobUrlCache(100);
  assert.equal(cache.delete("never-cached"), false);
});
