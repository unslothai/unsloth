// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { summarizeClearAllChats } from "../src/features/chat/utils/clear-all-chats-summary.ts";
import { readSrc } from "./helpers/kit.ts";

const result = (over: Record<string, unknown> = {}) => ({
  backend: "cleared" as const,
  legacy: "cleared" as const,
  deletedThreadIds: [] as string[],
  failedThreadIds: [] as string[],
  sandboxesKept: [] as string[],
  ...over,
});

test("a clean clear reports success with the right plural", () => {
  assert.deepEqual(summarizeClearAllChats(result(), 0).title, { key: "settings.chat.clearedAllChats" });
  assert.deepEqual(summarizeClearAllChats(result({ deletedThreadIds: ["a"] }), 0).title, {
    key: "settings.chat.clearedOneChat",
  });
  const many = summarizeClearAllChats(result({ deletedThreadIds: ["a", "b"] }), 0);
  assert.equal(many.ok, true);
  assert.deepEqual(many.title.values, { count: 2 });
});

test("a failed backend store is a warning, never 'no chats'", () => {
  const summary = summarizeClearAllChats(result({ backend: "failed" }), 3);
  assert.equal(summary.ok, false);
  assert.deepEqual(summary.description, { key: "settings.chat.storageClearFailed", values: { count: 3 } });
  assert.equal(
    summarizeClearAllChats(result({ legacy: "failed" }), 1).description?.key,
    "settings.chat.storageClearFailedOne",
  );
});

test("named failures report cleared and remaining counts", () => {
  const summary = summarizeClearAllChats(result({ deletedThreadIds: ["a", "b"], failedThreadIds: ["c", "d"] }), 2);
  assert.equal(summary.ok, false);
  assert.deepEqual(summary.description, {
    key: "settings.chat.chatsClearedRemain",
    values: { clearedCount: 2, remainingCount: 2 },
  });
  assert.equal(
    summarizeClearAllChats(result({ deletedThreadIds: ["a"], failedThreadIds: ["c"] }), 1).description?.key,
    "settings.chat.oneChatClearedRemainOne",
  );
});

test("the Recents menu and Settings share one clear-and-report path", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  assert.match(sidebar, /onSelect=\{\(\) => openDeleteDialog\(\{ kind: "clearAll" \}\)\}/);
  const branch = sidebar.slice(sidebar.indexOf('if (target.kind === "clearAll")'));
  // The open chat (or Compare pair id) is left before the clear can yield.
  assert.ok(branch.indexOf("navigate(") < branch.indexOf("await clearAllChatsWithToast({ deleteFiles: shouldDeleteChatFiles })"));
  assert.ok(branch.indexOf("navigate(") > 0);
  const dataTab = readSrc("features/settings/tabs/data-tab.tsx");
  assert.match(dataTab, /clearAllChatsWithToast\(/);
  const wrapper = readSrc("features/chat/utils/clear-all-chats-toast.ts");
  assert.match(wrapper, /offerToDeleteKeptSandboxes\(result\.sandboxesKept\)/);
});
