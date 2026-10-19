// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

// Each chat keeps its own browser tabs: switching chats swaps them, and a run in a chat off screen
// opens its tab there without moving the panel on screen.
register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { chatKeyOfThread, chatTabs, currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");

const urls = () =>
  chatTabs(useBrowserStore.getState()).map((tab) => {
    const entry = currentEntry(tab);
    return entry.kind === "web" ? entry.url : entry.kind;
  });

test("switching chats keeps each chat's tabs, panel and active tab; a new chat gets a new tab", () => {
  const store = useBrowserStore.getState();
  const a = chatKeyOfThread("a");
  const b = chatKeyOfThread("b");
  store.showChat(a);
  store.openUrl("https://one.example/");
  store.openUrl("https://two.example/");
  const two = useBrowserStore.getState().activeTabId;
  store.activateTab(useBrowserStore.getState().tabs[0]?.id ?? "");
  const one = useBrowserStore.getState().activeTabId;
  assert.notEqual(one, two);

  // The panel was open: the next chat opens on a new tab of its own, not on chat a's pages.
  store.showChat(b);
  assert.equal(useBrowserStore.getState().open, true);
  assert.deepEqual(urls(), ["newtab"]);
  store.openUrl("https://three.example/", { newTab: false });
  assert.deepEqual(urls(), ["https://three.example/"]);
  store.closePanel();

  // Back in chat a: its two pages, the one it had in front, and its panel open.
  store.showChat(a);
  assert.deepEqual(urls(), ["https://one.example/", "https://two.example/"]);
  assert.equal(useBrowserStore.getState().activeTabId, one);
  assert.equal(useBrowserStore.getState().open, true);
  // Chat b closed its panel, and comes back closed with its page.
  store.showChat(b);
  assert.equal(useBrowserStore.getState().open, false);
  assert.deepEqual(urls(), ["https://three.example/"]);
  // A chat's last tab closing closes only its panel; the other chat's tabs stay.
  store.openPanel();
  store.closeTab(chatTabs(useBrowserStore.getState())[0]?.id ?? "");
  assert.equal(useBrowserStore.getState().open, false);
  store.showChat(a);
  assert.deepEqual(urls(), ["https://one.example/", "https://two.example/"]);
});

test("a tab opened for a chat off screen waits there; the chat on screen is untouched", () => {
  const store = useBrowserStore.getState();
  const shown = chatKeyOfThread("shown");
  const away = chatKeyOfThread("away");
  store.closePanel();
  store.showChat(shown);
  store.openUrl("https://mine.example/");
  const mine = useBrowserStore.getState().activeTabId;
  const sequence = useBrowserStore.getState().openSequence;

  const agentTab = store.openTabIn(away, "https://agent.example/");
  assert.ok(agentTab);
  assert.equal(useBrowserStore.getState().activeTabId, mine);
  assert.equal(useBrowserStore.getState().openSequence, sequence, "the panel on screen is not re-opened");
  assert.deepEqual(urls(), ["https://mine.example/"]);
  assert.equal(useBrowserStore.getState().tabs.find((tab) => tab.id === agentTab)?.chat, away);

  // The user's own tab in front of that chat when they come back is the agent's.
  store.revealIn(away, agentTab ?? "");
  store.showChat(away);
  assert.equal(useBrowserStore.getState().open, true);
  assert.equal(useBrowserStore.getState().activeTabId, agentTab);
  assert.deepEqual(urls(), ["https://agent.example/"]);

  // Closing a tab of a chat off screen moves that chat's front tab, not this one's.
  store.openUrl("https://second.example/");
  const second = useBrowserStore.getState().activeTabId;
  store.showChat(shown);
  store.closeTab(second ?? "");
  store.showChat(away);
  assert.equal(useBrowserStore.getState().activeTabId, agentTab);
});

test("a new chat's tabs follow it to its id on the first save, and moving a tab stays within the chat", () => {
  const store = useBrowserStore.getState();
  store.closePanel();
  store.showChat("single:nonce-1");
  store.openUrl("https://draft.example/");
  store.openUrl("https://draft2.example/");
  store.adoptChat("single:nonce-1", chatKeyOfThread("saved"));
  assert.equal(useBrowserStore.getState().chat, chatKeyOfThread("saved"));
  assert.deepEqual(urls(), ["https://draft.example/", "https://draft2.example/"]);
  // Other chats' tabs sit between in the full list; the strip order is what moves.
  const [first, second] = chatTabs(useBrowserStore.getState());
  store.moveTab(second?.id ?? "", 0);
  assert.deepEqual(
    chatTabs(useBrowserStore.getState()).map((tab) => tab.id),
    [second?.id, first?.id],
  );
  // Opening the same address again focuses this chat's tab, never another chat's.
  store.closePanel();
  store.showChat(chatKeyOfThread("other"));
  store.openUrl("https://draft.example/");
  assert.equal(chatTabs(useBrowserStore.getState()).length, 1);
});

test("a deleted chat's tabs and panel state go with it; the chat on screen keeps its own", () => {
  const store = useBrowserStore.getState();
  store.closePanel();
  store.showChat(chatKeyOfThread("keep"));
  store.openUrl("https://kept.example/");
  const gone = chatKeyOfThread("gone");
  const popup = store.openTabIn(gone, "https://gone.example/", { background: true });
  assert.ok(popup);
  assert.ok(gone in useBrowserStore.getState().chats);
  // Chat deletion reaches the store through the chat feature's listener.
  (globalThis as { __chatDeleted?: (ids: string[], pairIds: string[]) => void }).__chatDeleted?.(["gone"], []);
  assert.equal(useBrowserStore.getState().tabs.some((tab) => tab.chat === gone), false);
  assert.equal(gone in useBrowserStore.getState().chats, false);
  assert.deepEqual(urls(), ["https://kept.example/"]);
});

test("deleting the chat on screen drops its tabs; deleting a comparison drops its pair's browser", () => {
  const store = useBrowserStore.getState();
  const deleted = (globalThis as { __chatDeleted?: (threadIds: string[], pairIds: string[]) => void }).__chatDeleted;
  store.closePanel();
  store.showChat(chatKeyOfThread("current"));
  store.openUrl("https://current.example/");
  deleted?.(["current"], []);
  assert.equal(useBrowserStore.getState().tabs.some((tab) => tab.chat === chatKeyOfThread("current")), false);
  assert.equal(useBrowserStore.getState().open, false);
  // The next chat shown does not bring the deleted one's state back.
  store.showChat(chatKeyOfThread("next"));
  assert.equal(chatKeyOfThread("current") in useBrowserStore.getState().chats, false);

  store.openTabIn("compare:pair9", "https://pair.example/", { background: true });
  deleted?.(["left", "right"], ["pair9"]);
  assert.equal(useBrowserStore.getState().tabs.some((tab) => tab.chat === "compare:pair9"), false);
  assert.equal("compare:pair9" in useBrowserStore.getState().chats, false);
});
