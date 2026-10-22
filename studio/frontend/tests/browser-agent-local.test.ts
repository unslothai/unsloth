// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");

const globals = globalThis as Record<string, unknown>;
useBrowserStore.subscribe(({ tabs }) => {
  for (const tab of tabs) if (tab.loading) queueMicrotask(() => useBrowserStore.getState().updateTab(tab.id, { loading: false }));
});

type Posted = { result: { ok: boolean; url?: string; error?: string } };

async function navigate(id: string, url: string) {
  const posted: Posted[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    if (op.op === "status") return { installed: true, result: { ok: true, requests: 0, quietMs: 1000 } };
    return { installed: true, result: { ok: true, text: "Local dev app", links: [] } };
  };
  await runBrowserAction(
    { id, sessionId: "s", op: "navigate", url, policy: { allowedDomains: [], blockedDomains: [] } },
    { runKey: "r", userText: `open ${url}` },
  );
  assert.equal(posted.length, 1);
  return posted[0].result;
}

test("the agent opens a local address only once the user has typed it in the address bar", async () => {
  const commands: string[] = [];
  globals.__callNative = async (command: string) => void commands.push(command);
  globals.__native = true;
  const refused = await navigate("a3", "http://localhost:7860/");
  globals.__native = false;
  assert.equal(refused.ok, false);
  assert.match(refused.error ?? "", /^localhost:7860 is on the user's computer.*Ask the user to open it there first\.$/);
  // A search engine's redirect link opens its target, so the target needs the grant too.
  const redirected = await navigate("a3-redirect", "https://www.google.com/url?q=http://localhost:7860/");
  assert.equal(redirected.error, "localhost:7860 is a local address, and the browser only opens public sites.");
  const web = await navigate("a3-web", "http://localhost:7860/");
  assert.equal(web.error, "localhost:7860 is a local address, and the browser only opens public sites.");
  assert.equal(JSON.stringify(useBrowserStore.getState().tabs).includes("localhost"), false);
  assert.deepEqual(commands, []);

  const userTab = useBrowserStore.getState().openTabIn(useBrowserStore.getState().chat, "http://localhost:7860/");
  assert.ok(userTab);
  globals.__granted = new Map([[userTab, "http://localhost:7860"]]);
  const count = useBrowserStore.getState().tabs.length;
  const opened = await navigate("a4", "http://localhost:7860/");
  assert.equal(opened.error, undefined);
  assert.equal(opened.ok, true);
  assert.equal(opened.url, "http://localhost:7860/");
  assert.equal(useBrowserStore.getState().tabs.length, count);
  assert.equal(useBrowserStore.getState().activeTabId, userTab);
  for (const [index, other] of ["http://localhost:7861/", "http://127.0.0.1:7860/", "https://localhost:7860/"].entries()) {
    assert.equal((await navigate(`a5-${index}`, other)).ok, false, other);
  }
  assert.deepEqual(commands, []);

  globals.__released = [];
  // the granted page showing an about:blank document is still the user's local tab
  useBrowserStore.getState().updateTab(userTab, { displayUrl: "about:blank" });
  const publicPage = await navigate("a6", "https://example.com/");
  assert.equal(publicPage.ok, true);
  // leaving the user's tab for its own public one, the run lets the user's go
  assert.deepEqual(globals.__released, [userTab]);
  const local = useBrowserStore.getState().tabs.find((tab) => tab.id === userTab);
  const entry = local && currentEntry(local);
  assert.equal(entry?.kind === "web" ? entry.url : undefined, "http://localhost:7860/");
  assert.notEqual(useBrowserStore.getState().activeTabId, userTab);
  useBrowserStore.getState().closeTab(userTab);
  assert.equal((await navigate("closed-local", "http://localhost:7860/")).ok, false);
  globals.__granted = undefined;
});

test("a run keeps its own granted tab when another tab of the chat has the same local origin", async () => {
  globals.__native = false;
  const store = useBrowserStore.getState();
  const first = store.openTabIn(store.chat, "http://localhost:7870/")!;
  const second = store.openTabIn(store.chat, "http://localhost:7870/")!;
  globals.__granted = new Map([[second, "http://localhost:7870"]]);
  assert.equal((await navigate("b1", "http://localhost:7870/a")).ok, true);
  assert.equal(useBrowserStore.getState().activeTabId, second);
  globals.__granted = new Map([[first, "http://localhost:7870"], [second, "http://localhost:7870"]]);
  assert.equal((await navigate("b2", "http://localhost:7870/b")).ok, true);
  assert.equal(useBrowserStore.getState().activeTabId, second);
  const entry = currentEntry(useBrowserStore.getState().tabs.find((tab) => tab.id === second)!);
  assert.equal(entry.kind === "web" ? entry.url : undefined, "http://localhost:7870/b");
  store.closeTab(first);
  store.closeTab(second);
  globals.__granted = undefined;
});
