// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dialog sizes its list before its opening paint, so the answer has to be available
// synchronously. The index is only built while the dialog is open, so on the first open of a
// page load the in-memory cache is empty whatever the history holds; the persisted hint left
// by the last completed build is what tells an empty history from an unread one.

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

// The point of the fixed height: once the list has rows, a query that narrows it to a few
// rows, or to none at all, must not resize the dialog.
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
  assert.equal(chatSearchHadRows(), false);
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

test("a session change drops the hint, so the next account never inherits it", () => {
  store.clear();
  rememberChatSearchHasRows(true);
  forgetChatSearchHasRows();
  assert.equal(chatSearchHadRows(), false);
});
