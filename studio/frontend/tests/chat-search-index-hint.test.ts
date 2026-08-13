// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The wiring, as opposed to the two leaf helpers: chatSearchIndexHasRows has to answer for a
// history whose index has not been built yet, which is the first open of every page load.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./chat-search-index-resolver.mjs", import.meta.url);
const { store } = installLocalStorageFake();

const { chatSearchIndexHasRows, writeCachedIndex } = await import(
  "../src/features/chat/hooks/use-chat-search-index.ts"
);
const { setAuthSessionEpochForTest } = await import(
  "./helpers/store-stubs/chat-search-auth.ts"
);

const row = {
  type: "single" as const,
  id: "t1",
  title: "Acme roadmap",
  snippet: "",
  updatedAt: 1,
};

test("an unbuilt index falls back to the last completed build's hint", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex(null);
  // Nothing ever built: no hint, so compact, which is the brand-new-profile case.
  assert.equal(chatSearchIndexHasRows(), false);

  // A completed build with rows, then the page reloads: the module cache is gone.
  writeCachedIndex([row]);
  assert.equal(chatSearchIndexHasRows(), true);
  writeCachedIndex(null);
  assert.equal(
    chatSearchIndexHasRows(),
    true,
    "150 stored chats must not read as an empty history on the next page load",
  );
});

test("an invalidated cache keeps the hint, an empty completed build clears it", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);

  // Invalidation says stale, not empty.
  writeCachedIndex(null);
  assert.equal(chatSearchIndexHasRows(), true);

  // A completed build that found nothing is the one thing that means "no chats".
  writeCachedIndex([]);
  assert.equal(chatSearchIndexHasRows(), false);
  writeCachedIndex(null);
  assert.equal(chatSearchIndexHasRows(), false);
});

test("a session change inside one page load drops the previous account's hint", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);
  assert.equal(chatSearchIndexHasRows(), true);

  setAuthSessionEpochForTest(1);
  assert.equal(
    chatSearchIndexHasRows(),
    false,
    "the next account must not be sized by the previous one's history",
  );
});
