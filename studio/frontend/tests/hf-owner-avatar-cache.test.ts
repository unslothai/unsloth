// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { HfOwnerAvatarCache } from "../src/features/hub/lib/hf-owner-avatar-cache.ts";

test("cached owner avatar updates reach non-fetching subscribers", () => {
  const cache = new HfOwnerAvatarCache(8);
  let shownUrl = cache.getUrl("unsloth");
  let notifications = 0;
  const unsubscribe = cache.subscribe("unsloth", () => {
    notifications += 1;
    shownUrl = cache.getUrl("unsloth");
  });

  cache.set("unsloth", {
    kind: "url",
    url: "https://cdn.example.test/unsloth.png",
    expiresAt: Date.now() + 60_000,
  });

  assert.equal(shownUrl, "https://cdn.example.test/unsloth.png");
  assert.equal(notifications, 1);


  cache.set("unsloth", {
    kind: "miss-transient",
    until: Date.now() + 60_000,
    failures: 1,
    staleUrl: "https://cdn.example.test/unsloth.png",
  });
  assert.equal(shownUrl, "https://cdn.example.test/unsloth.png");
  assert.equal(notifications, 2);

  cache.set("another-owner", {
    kind: "url",
    url: "https://cdn.example.test/another.png",
    expiresAt: Date.now() + 60_000,
  });
  assert.equal(notifications, 2);

  unsubscribe();
  cache.set("unsloth", { kind: "miss-permanent" });
  assert.equal(notifications, 2);
});

test("LRU eviction notifies subscribers and clears their snapshot", () => {
  const cache = new HfOwnerAvatarCache(2);
  let shownUrl: string | null = null;
  let notifications = 0;
  cache.subscribe("first-owner", () => {
    notifications += 1;
    shownUrl = cache.getUrl("first-owner");
  });

  cache.set("first-owner", {
    kind: "url",
    url: "https://cdn.example.test/first.png",
    expiresAt: Date.now() + 60_000,
  });
  cache.set("second-owner", { kind: "miss-permanent" });
  cache.set("third-owner", { kind: "miss-permanent" });

  assert.equal(shownUrl, null);
  assert.equal(notifications, 2);
});