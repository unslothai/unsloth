// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A Library link into a gallery pages back until its item loads. A failing gallery must end the
// search after a few tries, not one request per remaining page.

import assert from "node:assert/strict";
import test from "node:test";

import { loadGalleryUntil } from "../src/lib/gallery-deep-link.ts";

test("a gallery that keeps failing ends the lookup after a few tries", async () => {
  let requests = 0;
  const found = await loadGalleryUntil({
    has: () => false,
    count: () => 50,
    hasMore: () => true,
    refresh: async () => {},
    loadMore: async () => {
      requests += 1;
    },
    busy: () => false,
    cancelled: () => false,
  });
  assert.equal(found, false);
  assert.equal(requests, 3);
});

test("pages keep loading until the item arrives", async () => {
  let loaded = 0;
  const found = await loadGalleryUntil({
    has: () => loaded >= 150,
    count: () => loaded,
    hasMore: () => true,
    refresh: async () => {
      loaded = 50;
    },
    loadMore: async () => {
      loaded += 50;
    },
    busy: () => false,
    cancelled: () => false,
  });
  assert.equal(found, true);
  assert.equal(loaded, 150);
});
