// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Library card thumbnails share a bounded cache of object URLs. A card still loading or showing
// its URL must never have it evicted and revoked under it: the card would fall back to its type
// icon for good.

import assert from "node:assert/strict";
import test from "node:test";

import {
  acquireObjectUrl,
  cachedObjectUrlCount,
  clearCachedObjectUrls,
} from "../src/features/library/object-url-cache.ts";

const revoked = new Set<string>();
const realRevoke = URL.revokeObjectURL.bind(URL);
URL.revokeObjectURL = (url: string) => {
  revoked.add(url);
  realRevoke(url);
};

const blob = () => Promise.resolve(new Blob(["x"]));

test("entries in use survive the count cap, and go once released", async () => {
  clearCachedObjectUrls();
  // Held while still loading, as a card waiting on its thumbnail is.
  let finish!: (value: Blob) => void;
  const pending = acquireObjectUrl("pending", () => new Promise((resolve) => (finish = resolve)));
  for (let i = 0; i < 310; i++) {
    const held = acquireObjectUrl(`item-${i}`, blob);
    await held.url;
    held.release();
  }
  finish(new Blob(["late"]));
  const url = await pending.url;
  assert.equal(revoked.has(url), false);
  assert.ok(cachedObjectUrlCount() <= 300);
  pending.release();
  // Released and now the oldest, so the next miss past the cap takes it.
  const next = acquireObjectUrl("one-more", blob);
  await next.url;
  await Promise.resolve();
  assert.equal(revoked.has(url), true);
  next.release();
});

test("a hit is the same URL, and stays valid while anyone holds it", async () => {
  clearCachedObjectUrls();
  const first = acquireObjectUrl("same", blob);
  const second = acquireObjectUrl("same", () => Promise.reject(new Error("not refetched")));
  const url = await first.url;
  assert.equal(await second.url, url);
  first.release();
  for (let i = 0; i < 305; i++) {
    const held = acquireObjectUrl(`filler-${i}`, blob);
    await held.url;
    held.release();
  }
  assert.equal(revoked.has(url), false);
  second.release();
});
