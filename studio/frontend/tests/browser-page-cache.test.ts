// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import type { BrowserPage } from "../src/features/browser/api.ts";
import { PageCache } from "../src/features/browser/page-cache.ts";

const MB = 1024 * 1024;
const html = (bytes: number): BrowserPage => ({
  kind: "html",
  url: "https://example.com/",
  base: "https://example.com/",
  refresh: null,
  html: "x".repeat(bytes / 2),
});

const make = (pages = 12) => new PageCache<object>(pages, 32 * MB, 8 * MB);

test("HTML pages count toward the byte budget, and one over the per-page limit is not kept", () => {
  const cache = make();
  const keys = Array.from({ length: 12 }, () => ({}));
  for (const key of keys) cache.set(key, html(6 * MB));
  // Twelve 6 MB pages would be 72 MB; only the newest five fit in 32 MB.
  assert.equal(cache.size, 5);
  assert.ok(cache.bytes <= 32 * MB);
  assert.equal(cache.get(keys[0]), undefined);
  assert.ok(cache.get(keys[11]));
  cache.set({}, html(20 * MB));
  assert.equal(cache.size, 5);
});

test("replacements and deletes keep the total right", () => {
  const cache = make();
  const key = {};
  cache.set(key, html(2 * MB));
  cache.set(key, html(4 * MB));
  assert.equal(cache.bytes, 4 * MB);
  cache.delete(key);
  assert.equal(cache.bytes, 0);
});

test("the least recently used page goes first", () => {
  const cache = make(2);
  const [a, b, c] = [{}, {}, {}];
  cache.set(a, html(2));
  cache.set(b, html(2));
  cache.get(a);
  cache.set(c, html(2));
  assert.ok(cache.get(a));
  assert.equal(cache.get(b), undefined);
});
