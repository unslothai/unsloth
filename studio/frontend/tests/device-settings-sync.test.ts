// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/tauri-webview-resolver.mjs", import.meta.url);
register("./helpers/settings-api-resolver.mjs", import.meta.url);
const { store, fireWindowEvent } = installLocalStorageFake();
const attrs = new Map<string, string>();
Object.assign(globalThis.document, {
  documentElement: {
    style: {},
    getAttribute: (name: string) => attrs.get(name) ?? null,
    setAttribute: (name: string, value: string) => attrs.set(name, value),
    removeAttribute: (name: string) => attrs.delete(name),
  },
});

const empty = {
  saved: false,
  updatedAt: null,
  interfaceScale: null,
  browser: null,
  chat: null,
  plusMenu: null,
  timeFormat: null,
  spellcheck: null,
};
let account: Record<string, unknown> = empty;
const puts: Record<string, unknown>[] = [];
globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
  if (init?.method === "PUT") puts.push(JSON.parse(String(init.body)));
  return new Response(JSON.stringify(account), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const { startClientPreferencesSync } = await import(
  "../src/features/settings/hooks/use-device-settings-sync.ts"
);
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);
const { PLUS_MENU_ORDER, usePlusMenuPrefsStore } = await import(
  "../src/features/chat/stores/plus-menu-prefs-store.ts"
);
const { getTimeFormat, setTimeFormat, TIME_FORMAT_STORAGE_KEY, watchTimeFormat } =
  await import("../src/lib/time-format.ts");
const { getSpellCheck, setSpellCheck, SPELLCHECK_STORAGE_KEY, watchSpellCheck } =
  await import("../src/lib/spellcheck.ts");
const { BROWSER_ACCOUNT_KEY } = await import("../src/lib/account-transition.ts");
const { markPending, readPending } = await import(
  "../src/features/settings/lib/client-prefs-pending.ts"
);

const settle = () => new Promise((resolve) => setTimeout(resolve, 1600));

function freshBrowser() {
  store.clear();
  puts.length = 0;
  useChatPreferencesStore.setState(useChatPreferencesStore.getInitialState());
  usePlusMenuPrefsStore.setState(usePlusMenuPrefsStore.getInitialState());
  setTimeFormat("system");
  store.clear();
}

test("a fresh browser takes the account's preferences and sends no defaults back", async () => {
  freshBrowser();
  account = {
    ...empty,
    saved: true,
    chat: { sendShortcut: "mod-enter", showReplyTimes: true, toolVisibility: "expanded" },
    plusMenu: {
      pins: { mcp: true },
      order: [...PLUS_MENU_ORDER].reverse(),
      pinnedPromptIds: ["p1"],
    },
    timeFormat: "24h",
    spellcheck: false,
  };
  const stop = startClientPreferencesSync();
  await settle();
  stop();
  const chat = useChatPreferencesStore.getState();
  assert.equal(chat.sendShortcut, "mod-enter");
  assert.equal(chat.showReplyTimes, true);
  assert.equal(chat.toolVisibility, "expanded");
  assert.equal(chat.confirmDeleteChats, true);
  const menu = usePlusMenuPrefsStore.getState();
  assert.equal(menu.pins.mcp, true);
  assert.equal(menu.pins.deepResearch, true);
  assert.deepEqual(menu.order, [...PLUS_MENU_ORDER].reverse());
  assert.deepEqual(menu.pinnedPromptIds, ["p1"]);
  assert.equal(getTimeFormat(), "24h");
  assert.equal(getSpellCheck(), false);
  assert.deepEqual(puts, []);
});

test("an edit sends only its own section, without this browser's local-only fields", async () => {
  freshBrowser();
  account = { ...empty, saved: true, timeFormat: "24h" };
  const stop = startClientPreferencesSync();
  await settle();
  useChatPreferencesStore.getState().setShowReplyTimes(true);
  useChatPreferencesStore.getState().setNotifyWhenReplyFinishes(true);
  await settle();
  stop();
  assert.equal(puts.length, 1);
  assert.deepEqual(Object.keys(puts[0]), ["chat"]);
  const chat = puts[0].chat as Record<string, unknown>;
  assert.equal(chat.showReplyTimes, true);
  assert.equal("notifyWhenReplyFinishes" in chat, false);
  assert.equal("filesPanelCollapsed" in chat, false);
});

test("choices made here before the account had any are uploaded once", async () => {
  freshBrowser();
  setTimeFormat("12h");
  account = empty;
  const stop = startClientPreferencesSync();
  await settle();
  stop();
  assert.deepEqual(puts, [{ timeFormat: "12h" }]);
});

test("an edit not yet sent wins over the account's copy and is sent after load", async () => {
  freshBrowser();
  useChatPreferencesStore.getState().setShowReplyTimes(true);
  markPending(["chat"], true);
  account = { ...empty, saved: true, chat: { showReplyTimes: false }, timeFormat: "24h" };
  const stop = startClientPreferencesSync();
  await settle();
  stop();
  assert.equal(useChatPreferencesStore.getState().showReplyTimes, true);
  assert.equal(getTimeFormat(), "24h");
  assert.equal(puts.length, 1);
  assert.deepEqual(Object.keys(puts[0]), ["chat"]);
  assert.equal((puts[0].chat as Record<string, unknown>).showReplyTimes, true);
  assert.equal(readPending().chat, false);
});

test("another tab switching accounts does not mark its cleanup as edits", async () => {
  freshBrowser();
  setTimeFormat("24h");
  setSpellCheck(false);
  account = { ...empty, saved: true, timeFormat: "24h", spellcheck: false };
  watchTimeFormat(window);
  watchSpellCheck(window);
  const stop = startClientPreferencesSync();
  await settle();
  store.delete(TIME_FORMAT_STORAGE_KEY);
  store.delete(SPELLCHECK_STORAGE_KEY);
  store.set(BROWSER_ACCOUNT_KEY, "account:u2:alice");
  fireWindowEvent("storage", { key: TIME_FORMAT_STORAGE_KEY, newValue: null });
  fireWindowEvent("storage", { key: SPELLCHECK_STORAGE_KEY, newValue: null });
  await settle();
  stop();
  assert.equal(readPending().timeFormat, false);
  assert.equal(readPending().spellcheck, false);
  assert.deepEqual(puts, []);
});

test("a same-account marker upgrade keeps syncing edits", async () => {
  freshBrowser();
  store.set(BROWSER_ACCOUNT_KEY, "alice");
  account = { ...empty, saved: true, timeFormat: "12h" };
  const stop = startClientPreferencesSync();
  await settle();
  store.set(BROWSER_ACCOUNT_KEY, "account:a1:alice");
  setTimeFormat("24h");
  await settle();
  stop();
  assert.deepEqual(puts, [{ timeFormat: "24h" }]);
});

test("a field the account record lacks keeps this browser's value", async () => {
  freshBrowser();
  useChatPreferencesStore.setState({ generationStatsUpdates: "realtime" });
  account = { ...empty, saved: true, chat: { sendShortcut: "mod-enter" } };
  const stop = startClientPreferencesSync();
  await settle();
  stop();
  const chat = useChatPreferencesStore.getState();
  assert.equal(chat.sendShortcut, "mod-enter");
  assert.equal(chat.generationStatsUpdates, "realtime");
});

test("a load that lands after this tab switched accounts applies nothing", async () => {
  freshBrowser();
  store.set(BROWSER_ACCOUNT_KEY, "account:u1:alice");
  account = { ...empty, saved: true, chat: { sendShortcut: "mod-enter" }, timeFormat: "24h" };
  const realFetch = globalThis.fetch;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => (release = resolve));
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    await gate;
    return realFetch(input, init);
  }) as typeof fetch;
  const stop = startClientPreferencesSync();
  store.set(BROWSER_ACCOUNT_KEY, "account:u2:bob");
  release();
  await settle();
  stop();
  globalThis.fetch = realFetch;
  assert.equal(useChatPreferencesStore.getState().sendShortcut, "enter");
  assert.equal(getTimeFormat(), "system");
  assert.deepEqual(puts, []);
});

test("an edit still queued when this tab switches accounts is not sent as the next one", async () => {
  freshBrowser();
  store.set(BROWSER_ACCOUNT_KEY, "account:u1:alice");
  account = { ...empty, saved: true, timeFormat: "12h" };
  const stop = startClientPreferencesSync();
  await settle();
  setTimeFormat("24h");
  store.set(BROWSER_ACCOUNT_KEY, "account:u2:bob");
  await settle();
  stop();
  assert.deepEqual(puts, []);
});
