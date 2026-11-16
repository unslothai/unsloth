// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const local = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
const session = new Map<string, string>();
const writes: string[] = [];
const storageOf = (data: Map<string, string>, log?: string[]) => ({
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => {
    log?.push(key);
    data.set(key, value);
  },
  removeItem: (key: string) => void data.delete(key),
});
const localStorage = storageOf(local);
Object.defineProperty(globalThis, "sessionStorage", { value: storageOf(session, writes), configurable: true, writable: true });
Object.assign(globalThis, { localStorage, window: { localStorage, location: { protocol: "http:" }, addEventListener: () => undefined } });

register("./helpers/browser-store-resolver.mjs", import.meta.url);
type StoreModule = typeof import("../src/features/browser/store.ts");
const { chatKeyOfThread, chatTabs, currentEntry, useBrowserStore }: StoreModule = await import("../src/features/browser/store.ts");
const { TAB_SESSION_KEY, loadTabSession, parseTabSession, serializeTabSession } = await import("../src/features/browser/tab-session.ts");
const { accountDatabaseName } = await import("../src/lib/account-transition.ts");

let reloads = 0;
// A reload whose shown chat is a single: chat asks storage whether that chat was saved (tab-session.ts
// chatIsUnsaved); the stub answers through __storedThreads.has, so the two counts say when every answer is in.
let lookupsStarted = 0;
let lookupsAnswered = 0;
// The store as a reload of the page builds it: a new module instance reading the same sessionStorage.
const reloaded = async (): Promise<StoreModule> => {
  if (loadTabSession().shown?.startsWith("single:")) lookupsStarted++;
  return import(`../src/features/browser/store.ts?reload=${++reloads}`);
};
const shownUrls = (module: StoreModule) =>
  chatTabs(module.useBrowserStore.getState()).map((tab) => {
    const entry = currentEntry(tab);
    return entry.kind === "web" ? entry.url : entry.kind;
  });

const A = chatKeyOfThread("a");
const B = chatKeyOfThread("b");
class StoredThreads extends Set<string> {
  has(id: string): boolean {
    lookupsAnswered++;
    return super.has(id);
  }
}
Object.assign(globalThis, { __storedThreads: new StoredThreads(["a", "b"]) });
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
// The lookup starts with a dynamic import through the test's module hooks, which can take longer than one
// timer tick on a loaded runner, so wait for the answers themselves rather than a fixed delay.
const settled = async () => {
  const deadline = Date.now() + 10_000;
  while (lookupsAnswered < lookupsStarted && Date.now() < deadline) await tick();
  assert.equal(lookupsAnswered, lookupsStarted, "a stored-chat lookup never answered");
  await tick();
};

test("a reload restores the chat's web and new tabs with its panel open, and keeps them for it while another chat shows", async () => {
  const store = useBrowserStore.getState();
  store.showChat(A);
  store.openUrl("https://one.example/");
  const one = useBrowserStore.getState().activeTabId ?? "";
  store.navigate(one, { url: "https://one.example/next" }, { replace: false });
  store.updateTab(one, { title: "One", favicon: "https://one.example/icon.png", loading: true });
  store.renameTab(one, "Mine");
  store.setZoom(one, 1.5);
  store.setMuted(one, true);
  store.goBack(one);
  store.newTab();
  store.activateTab(one);
  store.setFloating(true);

  const first = await reloaded();
  assert.equal(first.useBrowserStore.getState().open, false);
  first.useBrowserStore.getState().showChat(A);
  const state = first.useBrowserStore.getState();
  assert.equal(state.open, true);
  assert.equal(state.floating, true);
  assert.equal(state.activeTabId, one);
  assert.deepEqual(shownUrls(first), ["https://one.example/", "newtab"]);
  const tab = state.tabs.find((candidate) => candidate.id === one);
  assert.deepEqual(tab?.history, [{ kind: "web", url: "https://one.example/" }, { kind: "web", url: "https://one.example/next" }]);
  assert.deepEqual(
    [tab?.index, tab?.customTitle, tab?.zoom, tab?.muted, tab?.loading, tab?.openKey],
    [0, "Mine", 1.5, true, false, null],
  );

  // Loaded on another chat: its panel stays shut, and chat a's comes back with it.
  const second = await reloaded();
  second.useBrowserStore.getState().showChat(B);
  assert.equal(second.useBrowserStore.getState().open, false);
  assert.deepEqual(shownUrls(second), []);
  second.useBrowserStore.getState().showChat(A);
  assert.equal(second.useBrowserStore.getState().open, true);
  assert.equal(second.useBrowserStore.getState().activeTabId, one);
  assert.deepEqual(shownUrls(second), ["https://one.example/", "newtab"]);
});

test("changes to what is not kept do not rewrite the saved tabs", () => {
  const store = useBrowserStore.getState();
  const id = useBrowserStore.getState().activeTabId ?? "";
  store.updateTab(id, { title: "Kept" });
  writes.length = 0;
  store.updateTab(id, { loading: true, nativeHistory: { back: true, forward: false } });
  store.updateTab(id, { loading: false });
  store.focusAddress();
  store.setDevice("mobile");
  store.reload(id);
  assert.deepEqual(writes, []);
  store.updateTab(id, { title: "Changed" });
  assert.deepEqual(writes, [TAB_SESSION_KEY]);
});

test("file, posted and temporary pages are dropped, leaving the tab on the kept page before them", () => {
  const web = (url: string, extra = {}) => ({ kind: "web" as const, url, ...extra });
  const base = { title: "", favicon: null, documentType: null, displayUrl: null, loading: false, reloadKey: 0, openKey: null, zoom: 1, nativeHistory: null, nativeError: null, customTitle: null, muted: false, pinnedId: null, chat: A };
  const file = { kind: "file" as const, fileId: "file-1", name: "a.txt", contentType: "text/plain" };
  const history = [web("https://a.example/"), file, web("https://b.example/"), web("https://form.example/", { method: "POST" as const, body: "x=1" }), web("https://c.example/", { temporary: true as const }), web("https://d.example/")];
  const tabs = [
    { ...base, id: "at-post", history, index: 3 },
    { ...base, id: "at-last", history, index: 5 },
    { ...base, id: "at-file", history: [file, web("https://a.example/")], index: 0 },
    { ...base, id: "only-file", history: [file], index: 0 },
    { ...base, id: "big-icon", history: [{ kind: "newtab" as const }], index: 0, favicon: `data:image/png;base64,${"A".repeat(5000)}` },
  ];
  const raw = serializeTabSession({ tabs, chat: A, chats: {}, open: true, activeTabId: "only-file", fullView: true, floating: false });
  assert.doesNotMatch(raw ?? "", /file-1|form\.example|c\.example|base64/);
  const restored = parseTabSession(raw);
  const kept = ["https://a.example/", "https://b.example/", "https://d.example/"].map((url) => ({ kind: "web", url }));
  assert.deepEqual(restored.tabs.map((tab) => [tab.id, tab.index]), [["at-post", 1], ["at-last", 2], ["at-file", 0], ["big-icon", 0]]);
  assert.deepEqual(restored.tabs[0]?.history, kept);
  assert.equal(restored.tabs[3]?.favicon, null);
  assert.deepEqual(restored.chats[A], { open: true, activeTabId: "big-icon", fullView: true, floating: false });
});

test("corrupt or unexpected saved tabs are ignored", async () => {
  const good = { id: "t1", chat: A, history: [{ kind: "web", url: "https://ok.example/" }], index: 7, zoom: 99 };
  for (const raw of ["{not json", "null", "[]", '{"version":2,"tabs":[]}', '{"version":1,"tabs":{}}', JSON.stringify({ version: 1, tabs: [good], chats: null }).slice(0, -5)]) {
    assert.deepEqual(parseTabSession(raw), { tabs: [], chats: {}, shown: null });
  }
  const bad = [
    null,
    "tab",
    { ...good, id: 5 },
    { ...good, id: "t2", history: [{ kind: "web", url: "javascript:alert(1)" }, { kind: "web", url: "file:///etc/passwd" }, { kind: "web", url: "https://" }, { kind: "internal", page: "nope" }, { kind: "other" }] },
    { ...good, id: "t3", history: "x" },
    { ...good },
  ];
  const restored = parseTabSession(JSON.stringify({ version: 1, tabs: [good, ...bad], chats: { [A]: { open: "yes", activeTabId: "gone", fullView: true }, other: { open: true } } }));
  assert.deepEqual(restored.tabs.map((tab) => [tab.id, tab.index, tab.zoom]), [["t1", 0, 5]]);
  assert.deepEqual(restored.chats, { [A]: { open: false, activeTabId: "t1", fullView: false, floating: false } });

  session.set(TAB_SESSION_KEY, "{not json");
  const module = await reloaded();
  module.useBrowserStore.getState().showChat(A);
  assert.deepEqual([module.useBrowserStore.getState().tabs, module.useBrowserStore.getState().open], [[], false]);
  Object.defineProperty(globalThis, "sessionStorage", { get: () => { throw new Error("denied"); }, configurable: true });
  assert.deepEqual(loadTabSession(), { tabs: [], chats: {}, shown: null });
  assert.doesNotThrow(() => useBrowserStore.getState().newTab());
  Object.defineProperty(globalThis, "sessionStorage", { value: storageOf(session, writes), configurable: true, writable: true });
});

test("the tabs are saved under the account that opened them, so another account restores none", () => {
  useBrowserStore.getState().openUrl("https://private.example/", { newTab: true });
  assert.equal(TAB_SESSION_KEY, "unsloth_browser_tabs:b1");
  assert.match(session.get(TAB_SESSION_KEY) ?? "", /private\.example/);
  assert.equal(loadTabSession().tabs.at(-1)?.openKey, "url:https://private.example/");
  for (const marker of ["account:e2:eve", "unsloth"]) {
    const key = accountDatabaseName("unsloth_browser_tabs", { getItem: () => marker });
    assert.notEqual(key, TAB_SESSION_KEY);
    assert.deepEqual(loadTabSession(key), { tabs: [], chats: {}, shown: null });
  }
});

test("an unsaved chat's tabs follow it to the key a reload gives it, by either way the chat page names it", async () => {
  session.clear();
  const draft = async (key: string) => {
    const module = await reloaded();
    module.useBrowserStore.getState().showChat(key);
    module.useBrowserStore.getState().openUrl("https://draft.example/");
    return module;
  };
  await draft(chatKeyOfThread("draft-1"));
  const first = await reloaded();
  await settled();
  first.useBrowserStore.getState().showChat(chatKeyOfThread("draft-2"));
  assert.deepEqual([first.useBrowserStore.getState().open, shownUrls(first)], [true, ["https://draft.example/"]]);
  // Unnamed first, then adopted once the thread id is known; the lookup may land after either.
  for (const late of [false, true]) {
    const next = await reloaded();
    if (!late) await settled();
    next.useBrowserStore.getState().showChat("single:new");
    assert.deepEqual(shownUrls(next), []);
    next.useBrowserStore.getState().adoptChat("single:new", chatKeyOfThread(`draft-${late}`));
    if (late) await settled();
    assert.deepEqual([next.useBrowserStore.getState().open, shownUrls(next)], [true, ["https://draft.example/"]]);
  }
  (globalThis as { __storedThreads?: Set<string> }).__storedThreads?.add("draft-true");
  const other = await reloaded();
  await settled();
  other.useBrowserStore.getState().showChat(chatKeyOfThread("elsewhere"));
  assert.deepEqual([other.useBrowserStore.getState().open, shownUrls(other)], [false, []]);
});

test("an unsaved project chat's tabs come back in that project's new chat, not another project's or a plain chat", async () => {
  for (const [landing, restored] of [["project:p1:nonce-2", true], ["project:p2:nonce-3", false], [chatKeyOfThread("plain"), false]] as const) {
    session.clear();
    const before = await reloaded();
    before.useBrowserStore.getState().showChat("project:p1:nonce-1");
    before.useBrowserStore.getState().openUrl("https://project.example/");
    const after = await reloaded();
    await settled();
    after.useBrowserStore.getState().showChat(landing);
    assert.deepEqual([after.useBrowserStore.getState().open, shownUrls(after)], restored ? [true, ["https://project.example/"]] : [false, []], landing);
  }
});
