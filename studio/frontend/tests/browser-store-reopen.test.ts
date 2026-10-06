// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { browserFile, cachePage, cachedPage, currentEntry, sentPosts, setNativeWebHistory, useBrowserStore } = await import(
  "../src/features/browser/store.ts"
);

test("reopening a rewritten file shows its new bytes; an unchanged one keeps its viewer", async () => {
  const store = useBrowserStore.getState();
  const open = (text: string) => store.openFile({ blob: new Blob([text]), name: "a.txt", key: "sandbox/a.txt" });
  const shown = () => {
    const tab = useBrowserStore.getState().tabs[0];
    const entry = tab && currentEntry(tab);
    return entry?.kind === "file" ? entry.fileId : null;
  };
  open("version 1");
  const first = shown();
  open("version 1");
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(shown(), first);
  open("version 2");
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.notEqual(shown(), first);
  assert.equal(await browserFile(shown() ?? "")?.text(), "version 2");
  // Two quick reopens: the last one wins.
  open("version 3");
  open("version 4");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(await browserFile(shown() ?? "")?.text(), "version 4");
  assert.equal(useBrowserStore.getState().tabs.length, 1);
  // Closing another tab while a reopen is queued keeps the queued bytes.
  store.openFile({ blob: new Blob(["other"]), name: "b.txt", key: "sandbox/b.txt" });
  open("version 5");
  store.closeTab(useBrowserStore.getState().tabs[1]?.id ?? "");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(await browserFile(shown() ?? "")?.text(), "version 5");
});

test("a large file differing only in its last byte is told apart across compare slices", async () => {
  const store = useBrowserStore.getState();
  const bytes = new Uint8Array(2.5 * 1024 * 1024 + 3).fill(7);
  const open = (data: Uint8Array<ArrayBuffer>) => store.openFile({ blob: new Blob([data]), name: "big.bin", key: "sandbox/big.bin" });
  const shown = () => {
    const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.openKey === "file:sandbox/big.bin");
    const entry = tab && currentEntry(tab);
    return entry?.kind === "file" ? entry.fileId : null;
  };
  open(bytes);
  const first = shown();
  open(bytes.slice());
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.equal(shown(), first);
  const changed = bytes.slice();
  changed[changed.length - 1] = 8;
  open(changed);
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.notEqual(shown(), first);
});

test("a native tab's navigations replace its entry unless asked to keep it", () => {
  setNativeWebHistory(true);
  const store = useBrowserStore.getState();
  store.openUrl("https://a.example/", { newTab: true });
  const tab = () => useBrowserStore.getState().tabs.find((candidate) => candidate.id === id);
  const id = useBrowserStore.getState().activeTabId ?? "";
  store.navigate(id, { url: "https://b.example/" });
  assert.equal(tab()?.history.length, 1);
  store.navigate(id, { url: "https://c.example/" }, { replace: false });
  assert.deepEqual(
    tab()?.history.map((entry) => (entry.kind === "web" ? entry.url : "")),
    ["https://b.example/", "https://c.example/"],
  );
  setNativeWebHistory(false);
});

test("a reload refetches its own page, not the pages behind it", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://a.example/", { newTab: true });
  const id = useBrowserStore.getState().activeTabId ?? "";
  const entry = () => {
    const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === id);
    assert.ok(tab);
    return currentEntry(tab);
  };
  const page = (url: string) => ({ kind: "html" as const, url, base: url, html: "<p>x</p>", refresh: null });
  const first = entry();
  cachePage(first, page("https://a.example/"));
  store.navigate(id, { url: "https://b.example/" });
  const second = entry();
  cachePage(second, page("https://b.example/"));
  store.reload(id);
  assert.equal(cachedPage(second), undefined);
  assert.ok(cachedPage(first));
});

test("closing the last tab leaves full view, so the browser reopens beside the chat", () => {
  const store = useBrowserStore.getState();
  for (const tab of useBrowserStore.getState().tabs) store.closeTab(tab.id);
  store.openUrl("https://a.example/", { newTab: true });
  store.setFullView(true);
  store.closeTab(useBrowserStore.getState().activeTabId ?? "");
  assert.equal(useBrowserStore.getState().open, false);
  assert.equal(useBrowserStore.getState().fullView, false);
});

test("a dragged tab moves to where it is dropped, keeping which tab is open", () => {
  const store = useBrowserStore.getState();
  for (const tab of useBrowserStore.getState().tabs) store.closeTab(tab.id);
  for (const host of ["a", "b", "c"]) store.openUrl(`https://${host}.example/`, { newTab: true });
  const hosts = () =>
    useBrowserStore.getState().tabs.map((tab) => {
      const entry = currentEntry(tab);
      return entry.kind === "web" ? new URL(entry.url).hostname[0] : "?";
    });
  const [first] = useBrowserStore.getState().tabs;
  const active = useBrowserStore.getState().activeTabId;
  store.moveTab(first?.id ?? "", 2);
  assert.deepEqual(hosts(), ["b", "c", "a"]);
  store.moveTab(first?.id ?? "", 99);
  assert.deepEqual(hosts(), ["b", "c", "a"]);
  store.moveTab(first?.id ?? "", 0);
  assert.deepEqual(hosts(), ["a", "b", "c"]);
  assert.equal(useBrowserStore.getState().activeTabId, active);
});

test("a duplicated form result asks before posting again; a duplicated page just loads", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://example.com/order", { method: "POST", body: "item=1" });
  store.openUrl("https://example.com/page", { newTab: true });
  const [posted, page] = useBrowserStore.getState().tabs.slice(-2);
  store.duplicateTab(posted?.id ?? "");
  store.duplicateTab(page?.id ?? "");
  const copyOf = (id: string | undefined) => {
    const tabs = useBrowserStore.getState().tabs;
    return tabs[tabs.findIndex((tab) => tab.id === id) + 1];
  };
  const [postTab, pageTab] = [copyOf(posted?.id), copyOf(page?.id)];
  assert.ok(postTab && pageTab);
  const postCopy = currentEntry(postTab);
  assert.equal(postCopy.kind === "web" && postCopy.method, "POST");
  assert.equal(sentPosts.has(postCopy), true);
  assert.equal(sentPosts.has(currentEntry(pageTab)), false);
});

test("a form result pushed out of the cache still asks before posting again", () => {
  const store = useBrowserStore.getState();
  store.openUrl("https://example.com/pay", { method: "POST", body: "card=1" });
  const posted = useBrowserStore.getState().tabs.at(-1);
  assert.ok(posted);
  const entry = currentEntry(posted);
  sentPosts.add(entry);
  cachePage(entry, { kind: "html", url: "https://example.com/pay", base: "https://example.com/pay", refresh: null, html: "paid" });
  for (let i = 0; i < 12; i++) {
    store.openUrl(`https://example.com/${i}`, { newTab: true });
    const tab = useBrowserStore.getState().tabs.at(-1);
    assert.ok(tab);
    cachePage(currentEntry(tab), { kind: "html", url: `https://example.com/${i}`, base: `https://example.com/${i}`, refresh: null, html: "x" });
  }
  // What the page view checks: no cached copy, already sent, so it asks rather than resending.
  assert.equal(cachedPage(entry), undefined);
  assert.equal(sentPosts.has(entry), true);
});

test("going Back to a keyed file restores its key, so its card finds the tab again", () => {
  const store = useBrowserStore.getState();
  store.openFile({ blob: new Blob(["<p>hi</p>"]), name: "page.html", contentType: "text/html", key: "html:back" });
  const tabId = useBrowserStore.getState().activeTabId ?? "";
  const tab = () => useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
  assert.equal(tab()?.openKey, "file:html:back");
  store.navigate(tabId, { url: "https://example.com/" });
  assert.equal(tab()?.openKey, null);
  store.goBack(tabId);
  assert.equal(tab()?.openKey, "file:html:back");
  store.goForward(tabId);
  assert.equal(tab()?.openKey, null);
  // A copy going Back does not claim the original's key.
  store.goBack(tabId);
  store.duplicateTab(tabId);
  const copy = useBrowserStore.getState().tabs.find((candidate) => candidate.id !== tabId && candidate.history.some((entry) => entry.kind === "file" && entry.name === "page.html"));
  assert.equal(copy?.openKey, null);
  store.navigate(copy?.id ?? "", { url: "https://example.com/" });
  store.goBack(copy?.id ?? "");
  assert.equal(useBrowserStore.getState().tabs.find((candidate) => candidate.id === copy?.id)?.openKey, null);
});
