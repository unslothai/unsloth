// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { useBookmarkedTurnsStore } from "../src/features/chat/stores/bookmarked-turns-store.ts";

const store = () => useBookmarkedTurnsStore.getState();

test("deleted chats leave no bookmark records behind", () => {
  useBookmarkedTurnsStore.setState({ bookmarkedByThread: {} });
  store().toggleBookmarkedTurn("t1", "u1");
  store().toggleBookmarkedTurn("t1", "u2");
  store().toggleBookmarkedTurn("t2", "u9");
  store().toggleBookmarkedTurn("t3", "u5");
  store().forgetThreads(["t1", "t3", "never-bookmarked"]);
  assert.deepEqual(store().bookmarkedByThread, { t2: ["u9"] });
});

test("forgetting chats with no bookmarks keeps the state object", () => {
  useBookmarkedTurnsStore.setState({ bookmarkedByThread: { t2: ["u9"] } });
  const before = store().bookmarkedByThread;
  store().forgetThreads(["t1"]);
  assert.equal(store().bookmarkedByThread, before);
});

test("every chat delete route forgets the deleted chats' bookmarks", async () => {
  // project delete and clear-all remove chats without going through deleteChatItems
  const { readSrcAsync } = await import("./helpers/kit.ts");
  for (const file of [
    "features/chat/hooks/use-chat-sidebar-items.ts",
    "features/chat/hooks/use-chat-projects.ts",
    "features/chat/utils/clear-all-chats.ts",
  ]) {
    assert.match(
      await readSrcAsync(file),
      /forgetThreads\(/,
      `${file} leaves bookmarks behind`,
    );
  }
});

test("clear all chats forgets the cleared chats' bookmarks", async () => {
  const clearAll = await readFile(
    new URL("../src/features/chat/utils/clear-all-chats.ts", import.meta.url),
    "utf8",
  );
  assert.match(clearAll, /forgetThreads\(result\.deletedThreadIds\)/);
});

test("forgetting a deleted prompt drops only that bookmark", () => {
  store().toggleBookmarkedTurn("chat-a", "m1");
  store().toggleBookmarkedTurn("chat-a", "m2");
  store().forgetTurns("chat-a", ["m1"]);
  assert.deepEqual(store().bookmarkedByThread["chat-a"], ["m2"]);
  store().forgetTurns("chat-a", ["m2"]);
  assert.equal("chat-a" in store().bookmarkedByThread, false);
  const before = store().bookmarkedByThread;
  store().forgetTurns("chat-a", ["m3"]);
  assert.equal(store().bookmarkedByThread, before);
});

test("deleting a message forgets its bookmark, and the rail resyncs after keyboard focus leaves", async () => {
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(thread, /forgetTurns\(remoteId, \[messageId\]\)/);
  const nav = await readFile(
    new URL(
      "../src/components/assistant-ui/turn-navigation.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(nav, /addEventListener\("focusout", onFocusOut\)/);
  assert.match(nav, /addEventListener\("scroll", fadeEnds/);
});

test("an open turn card reads its reply live while it streams", async () => {
  const nav = await readFile(
    new URL(
      "../src/components/assistant-ui/turn-navigation.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(nav, /const liveReply = useAuiState/);
  assert.match(nav, /\(liveReply \?\? preview\.reply\)/);
  // the live scan stops at the latest prompt, so an older card costs one turn per token
  assert.match(
    nav,
    /role === "user"\) \{\s*return messages\[index\]\.id === previewOpenerId/,
  );
});

test("the turn card stays inside a phone-width viewport", async () => {
  const nav = await readFile(
    new URL(
      "../src/components/assistant-ui/turn-navigation.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // the card is positioned from the right, so the cap is what keeps its left edge on screen
  assert.match(
    nav,
    /right: `calc\(\$\{preview\.right\}px \+ 1\.5rem\)`,[\s\S]{0,160}maxWidth: `calc\(100vw - \$\{preview\.right\}px - 2rem\)`/,
  );
});

test("the turn card follows its focused marker when the rail scrolls", async () => {
  const nav = await readFile(
    new URL(
      "../src/components/assistant-ui/turn-navigation.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    nav,
    /const onRailScroll = useCallback\([\s\S]{0,700}current\.openerId === focused\.dataset\.turnId[\s\S]{0,120}top: box\.top \+ box\.height \/ 2/,
  );
});

test("the turn card names its turn and stays out of select-all copies", async () => {
  const nav = await readFile(
    new URL(
      "../src/components/assistant-ui/turn-navigation.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // the card is portaled to body, so the rail anchor's select-none does not reach it
  assert.match(nav, /"aui-turn-preview [^"]*\bselect-none\b/);
  assert.match(
    nav,
    /<span className="shrink-0 text-muted-foreground\/80 text-ui-11">\s*\{preview \? t\("turns\.label", \{ number: preview\.turn \}\)/,
  );
});
