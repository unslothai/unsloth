// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dialog sizes its list from what the index cache already knows. The index is only
// built while the dialog is open, so the first open of a page load reads a count of zero
// whatever the history holds, and must not reserve a height it may never fill.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isCompactChatSearchList } = await import(
  "../src/features/chat/utils/chat-search-list-height.ts"
);

test("an index that has not been built yet opens compact", () => {
  assert.equal(isCompactChatSearchList(true, 0), true);
});

test("an index known to have rows opens at the fixed height", () => {
  assert.equal(isCompactChatSearchList(true, 150), false);
});

// The point of the fixed height: once the list has rows, a query that narrows it to a few
// rows, or to none at all, must not resize the dialog.
test("a populated open keeps the fixed height while a query narrows it", () => {
  assert.equal(isCompactChatSearchList(false, 4), false);
  assert.equal(isCompactChatSearchList(false, 0), false);
});

// The first build lands during the open, so the compact height is given up then and held.
test("rows arriving mid-open take the fixed height for the rest of that open", () => {
  const afterBuild = isCompactChatSearchList(true, 150);
  assert.equal(afterBuild, false);
  assert.equal(isCompactChatSearchList(afterBuild, 0), false);
});
