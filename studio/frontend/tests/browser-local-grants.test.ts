// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

register("./helpers/browser-local-resolver.mjs", import.meta.url);
const { grantLocalOrigin, localOriginGranted } = await import("../src/features/browser/native-support.ts");
const { currentEntry, setNativeWebHistory, useBrowserStore } = await import("../src/features/browser/store.ts");
const globals = globalThis as Record<string, unknown>;
// Native calls wait on a lazy import, so a fixed tick can pass before they're made.
async function until(done: () => boolean): Promise<void> {
  for (let tries = 0; !done() && tries < 200; tries++) await new Promise((resolve) => setTimeout(resolve, 5));
}

test("local grants belong to one tab and close even when the native view was pruned", async () => {
  const calls: { command: string; args: Record<string, unknown> }[] = [];
  globals.__localInvoke = async (command: string, args: Record<string, unknown>) => { calls.push({ command, args }); };
  const store = useBrowserStore.getState();
  const owner = store.openTabIn(store.chat, "http://localhost:7860/")!;
  const publicTab = store.openTabIn(store.chat, "https://example.com/")!;
  await grantLocalOrigin(owner, "http://localhost:7860/");
  assert.equal(localOriginGranted(owner, "http://localhost:7860/app"), true);
  assert.equal(localOriginGranted(publicTab, "http://localhost:7860/app"), false);
  assert.equal(localOriginGranted(owner, "http://localhost:7861/"), false);
  store.closeTab(owner);
  await until(() => calls.at(-1)?.command === "browser_view_close");
  assert.equal(localOriginGranted(owner, "http://localhost:7860/"), false);
  assert.deepEqual(calls.at(-1), { command: "browser_view_close", args: { tabId: owner, forgetLocal: true } });
  store.closeTab(publicTab);
});

test("a tab closed while a grant was pending cannot regain local access", async () => {
  let release!: () => void;
  const calls: string[] = [];
  globals.__localInvoke = async (command: string) => {
    calls.push(command);
    if (command === "browser_view_grant") await new Promise<void>((resolve) => { release = resolve; });
  };
  const store = useBrowserStore.getState();
  const tab = store.openTabIn(store.chat, "http://localhost:7860/")!;
  const granting = grantLocalOrigin(tab, "http://localhost:7860/");
  await until(() => calls.includes("browser_view_grant"));
  store.closeTab(tab);
  release();
  await granting;
  assert.equal(localOriginGranted(tab, "http://localhost:7860/"), false);
  assert.deepEqual(calls, ["browser_view_grant", "browser_view_close"]);
});

test("leaving a granted origin for a page that isn't web forgets the native grant too", async () => {
  const calls: { command: string; args: Record<string, unknown> }[] = [];
  globals.__localInvoke = async (command: string, args: Record<string, unknown>) => { calls.push({ command, args }); };
  const store = useBrowserStore.getState();
  store.newTab();
  const tab = useBrowserStore.getState().activeTabId!;
  store.navigate(tab, { url: "http://localhost:7860/" });
  await grantLocalOrigin(tab, "http://localhost:7860/");
  store.goBack(tab);
  await until(() => calls.at(-1)?.command === "browser_view_close");
  assert.equal(localOriginGranted(tab, "http://localhost:7860/"), false);
  assert.deepEqual(calls.at(-1), { command: "browser_view_close", args: { tabId: tab, forgetLocal: true } });
  store.closeTab(tab);
});

test("a native tab keeps the page it leaves for a local origin, whose view starts with no history", () => {
  setNativeWebHistory(true);
  try {
    const store = useBrowserStore.getState();
    const tab = store.openTabIn(store.chat, "https://example.com/")!;
    store.navigate(tab, { url: "https://example.org/" });
    store.updateTab(tab, { displayUrl: "https://example.org/docs/intro" });
    store.navigate(tab, { url: "http://localhost:7860/" });
    store.navigate(tab, { url: "http://localhost:7860/app" });
    const { history } = useBrowserStore.getState().tabs.find((item) => item.id === tab)!;
    assert.deepEqual(history.map((entry) => (entry.kind === "web" ? entry.url : entry.kind)), [
      "https://example.org/docs/intro",
      "http://localhost:7860/app",
    ]);
    store.goBack(tab);
    const entry = currentEntry(useBrowserStore.getState().tabs.find((item) => item.id === tab)!);
    assert.equal(entry.kind === "web" && entry.url, "https://example.org/docs/intro");
    store.closeTab(tab);
  } finally {
    setNativeWebHistory(false);
  }
});

test("the Local badge follows the tab's granted entry, not a data: or about:blank page it shows", () => {
  const panel = readSrc("features/browser/browser-panel.tsx");
  assert.match(panel, /const local = entry\.kind === "web" && !tab\.nativeError && localOriginGranted\(tab\.id, entry\.url\);/);
});

test("a generated page a native tab showed is not kept as its history across a local boundary", () => {
  setNativeWebHistory(true);
  try {
    const store = useBrowserStore.getState();
    const tab = store.openTabIn(store.chat, "https://example.com/")!;
    store.updateTab(tab, { displayUrl: "about:blank" });
    store.navigate(tab, { url: "http://localhost:7860/" });
    const { history } = useBrowserStore.getState().tabs.find((item) => item.id === tab)!;
    assert.deepEqual(history.map((entry) => (entry.kind === "web" ? entry.url : entry.kind)), [
      "https://example.com/",
      "http://localhost:7860/",
    ]);
    store.closeTab(tab);
  } finally {
    setNativeWebHistory(false);
  }
});
