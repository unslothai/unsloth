// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dialog sizes its list before its opening paint, so the answer must be synchronous. On
// the first open of a page load the in-memory cache is empty whatever the history holds, so
// the persisted hint is what tells an empty history from an unread one.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const { isCompactChatSearchList } = await import(
  "../src/features/chat/utils/chat-search-list-height.ts"
);
const {
  chatSearchHadRows,
  forgetChatSearchHasRows,
  rememberChatSearchHasRows,
} = await import("../src/features/chat/utils/chat-search-history-hint.ts");

test("a history known to have no rows opens compact", () => {
  assert.equal(isCompactChatSearchList(true, false), true);
});

test("a history known to have rows opens at the fixed height", () => {
  assert.equal(isCompactChatSearchList(true, true), false);
});

// The point of the fixed height: once the list has rows, narrowing it must not resize.
test("a populated open keeps the fixed height while a query narrows it", () => {
  assert.equal(isCompactChatSearchList(false, true), false);
  assert.equal(isCompactChatSearchList(false, false), false);
});

// Backstop for a browser that has never built the index: the compact height is given up when
// the first build lands and held for the rest of that open.
test("rows arriving mid-open take the fixed height for the rest of that open", () => {
  const afterBuild = isCompactChatSearchList(true, true);
  assert.equal(afterBuild, false);
  assert.equal(isCompactChatSearchList(afterBuild, false), false);
});

test("a completed build with rows is remembered, so the next page load opens fixed", () => {
  store.clear();
  // Nothing recorded yet reads as unknown, which is not the same as a history known to
  // be empty: only the latter may be sized compact.
  assert.equal(chatSearchHadRows(), null);
  rememberChatSearchHasRows(true);
  assert.equal(chatSearchHadRows(), true);
  // The regression: with 150 stored chats and a cache that has not been built yet, the
  // opening render must NOT pick compact and then grow to 420px when the rows land.
  assert.equal(isCompactChatSearchList(true, chatSearchHadRows()), false);
});

test("a completed build with no rows keeps the first open compact", () => {
  store.clear();
  rememberChatSearchHasRows(true);
  rememberChatSearchHasRows(false);
  assert.equal(chatSearchHadRows(), false);
  assert.equal(isCompactChatSearchList(true, chatSearchHadRows()), true);
});

test("an empty answer stops being trusted once it is old", () => {
  store.clear();
  rememberChatSearchHasRows(false);
  assert.equal(chatSearchHadRows(), false);

  // Chats made on another device or through the API never reach this tab, so this answer
  // cannot stand indefinitely: past its window it goes back to unknown rather than sizing a
  // populated dialog compact and growing it mid-open.
  const raw = store.get("unsloth_chat_search_has_rows") ?? "";
  const stale = Date.now() - (12 * 60 * 60 * 1000 + 1000);
  store.set("unsloth_chat_search_has_rows", `0.${stale}`);
  assert.notEqual(raw, "");
  assert.equal(chatSearchHadRows(), null);
  assert.equal(isCompactChatSearchList(true, chatSearchHadRows()), false);

  // A rows answer has no such risk: sizing an emptied history at the fixed height costs
  // nothing, so it is not aged out.
  rememberChatSearchHasRows(true);
  assert.equal(chatSearchHadRows(), true);
});

test("an empty answer written before the stamp existed reads as unknown", () => {
  store.clear();
  // What an upgrade finds in storage.
  store.set("unsloth_chat_search_has_rows", "0");
  assert.equal(chatSearchHadRows(), null);
  store.set("unsloth_chat_search_has_rows", "0.not-a-time");
  assert.equal(chatSearchHadRows(), null);
  // A clock moved backwards leaves an answer that cannot be aged, so it says as little.
  store.set("unsloth_chat_search_has_rows", `0.${Date.now() + 60_000}`);
  assert.equal(chatSearchHadRows(), null);
});

test("a session change drops the hint, so the next account never inherits it", () => {
  store.clear();
  rememberChatSearchHasRows(true);
  forgetChatSearchHasRows();
  assert.equal(chatSearchHadRows(), null);
  // Unknown, so the next account reserves the fixed height rather than opening compact
  // on the strength of the previous account's history.
  assert.equal(isCompactChatSearchList(true, chatSearchHadRows()), false);
});
