// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dialog sizes before first paint, so the persisted hint tells empty history from unread.

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

test("a populated open keeps the fixed height while a query narrows it", () => {
  assert.equal(isCompactChatSearchList(false, true), false);
  assert.equal(isCompactChatSearchList(false, false), false);
});

test("rows arriving mid-open take the fixed height for the rest of that open", () => {
  const afterBuild = isCompactChatSearchList(true, true);
  assert.equal(afterBuild, false);
  assert.equal(isCompactChatSearchList(afterBuild, false), false);
});

test("a completed build with rows is remembered, so the next page load opens fixed", () => {
  store.clear();
  assert.equal(chatSearchHadRows(), null);
  rememberChatSearchHasRows(true);
  assert.equal(chatSearchHadRows(), true);
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

  // Chats from other devices never reach this tab, so the empty answer ages out.
  const raw = store.get("unsloth_chat_search_has_rows") ?? "";
  const stale = Date.now() - (12 * 60 * 60 * 1000 + 1000);
  store.set("unsloth_chat_search_has_rows", `0.${stale}`);
  assert.notEqual(raw, "");
  assert.equal(chatSearchHadRows(), null);
  assert.equal(isCompactChatSearchList(true, chatSearchHadRows()), false);

  rememberChatSearchHasRows(true);
  assert.equal(chatSearchHadRows(), true);
});

test("an empty answer written before the stamp existed reads as unknown", () => {
  store.clear();
  store.set("unsloth_chat_search_has_rows", "0");
  assert.equal(chatSearchHadRows(), null);
  store.set("unsloth_chat_search_has_rows", "0.not-a-time");
  assert.equal(chatSearchHadRows(), null);
  store.set("unsloth_chat_search_has_rows", `0.${Date.now() + 60_000}`);
  assert.equal(chatSearchHadRows(), null);
});

test("a session change drops the hint, so the next account never inherits it", () => {
  store.clear();
  rememberChatSearchHasRows(true);
  forgetChatSearchHasRows();
  assert.equal(chatSearchHadRows(), null);
  assert.equal(isCompactChatSearchList(true, chatSearchHadRows()), false);
});
