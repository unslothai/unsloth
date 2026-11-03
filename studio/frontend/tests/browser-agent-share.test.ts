// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const agent = await import("../src/features/browser/agent/agent-store.ts");
const { shareAgentTab } = await import("../src/features/browser/agent/tab-sharing.ts");
const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };

test("sharing an existing tab keeps its page and isolates chats; revocation cancels a pending action", async () => {
  const store = useBrowserStore.getState();
  const chat = store.chat;
  const tab = store.openTabIn(chat, "https://example.com/form")!;
  const otherTab = store.openTabIn("other-chat", "https://other.example.org/")!;
  const secondTab = store.openTabIn(chat, "https://example.com/second")!;
  store.activateTab(tab);
  const before = useBrowserStore.getState().tabs.find((item) => item.id === tab)!;
  const posts: Array<{ result: { ok: boolean; text?: string; error?: string } }> = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => { posts.push(JSON.parse(init.body)); return new Response(null, { status: 200 }); };
  let resolveDescribe: (value: unknown) => void = () => {};
  let described: () => void = () => {};
  const started = new Promise<void>((resolve) => { described = resolve; });
  const calls: string[] = [];
  globals.__frameAgent = async (_tab: string, op: Record<string, unknown>) => {
    calls.push(String(op.op));
    if (op.op === "describe") { described(); return new Promise((resolve) => { resolveDescribe = resolve; }); }
    return { installed: true, result: { ok: true, text: "unsaved form state", links: [] } };
  };
  assert.equal(shareAgentTab(chat, tab), true);
  assert.equal(agent.useBrowserAgentStore.getState().enabled, true);
  assert.equal(shareAgentTab(chat, otherTab), false);
  assert.equal(shareAgentTab("other-chat", otherTab), false);
  assert.equal(shareAgentTab(chat, tab), true);
  assert.equal(useBrowserStore.getState().tabs.find((item) => item.id === tab), before);
  assert.equal(agent.agentChat("other-chat").tabId, null);
  await runBrowserAction({ id: "share-snapshot", sessionId: "s", op: "snapshot", policy }, { runKey: "one", chat, userText: "example.com" });
  assert.equal(posts[0].result.text, "unsaved form state");
  assert.equal(shareAgentTab(chat, tab), true);
  assert.notEqual(agent.agentChat(chat).stoppedRun, "one");
  const pending = runBrowserAction({ id: "share-click", sessionId: "s", op: "click", ref: "r1", policy }, { runKey: "one", chat, userText: "example.com" });
  await started;
  const queued = runBrowserAction({ id: "share-queued", sessionId: "s", op: "snapshot", policy }, { runKey: "one", chat, userText: "example.com" });
  store.activateTab(secondTab);
  shareAgentTab(chat, secondTab);
  resolveDescribe({ installed: true, result: { ok: true, info: { role: "button", name: "OK", tag: "button" } } });
  await pending;
  await queued;
  assert.equal(agent.agentChat(chat).tabId, secondTab);
  assert.equal(calls.includes("click"), false);
  assert.equal(posts[1].result.ok, false);
  assert.equal(posts[2].result.ok, false);
  agent.endBrowserAgentRun("one");
  await runBrowserAction({ id: "share-new-run", sessionId: "s", op: "snapshot", policy }, { runKey: "two", chat, userText: "example.com" });
  assert.equal(posts[3].result.text, "unsaved form state");
  shareAgentTab(chat, null);
  assert.equal(agent.agentChat(chat).tabId, null);
  agent.endBrowserAgentRun("two");
  await runBrowserAction({ id: "share-after-revoke", sessionId: "s", op: "snapshot", policy }, { runKey: "three", chat, userText: "example.com" });
  assert.equal(posts[4].result.ok, false);
  assert.match(posts[4].result.error ?? "", /No page is open/);
  assert.equal(useBrowserStore.getState().tabs.find((item) => item.id === tab), before);
});

test("revoking during a pending observation suppresses the page contents", async () => {
  agent.endBrowserAgentRun("three");
  const chat = useBrowserStore.getState().chat;
  const tab = useBrowserStore.getState().activeTabId!;
  assert.equal(shareAgentTab(chat, tab), true);
  let release: (value: unknown) => void = () => {};
  let observed: () => void = () => {};
  const started = new Promise<void>((resolve) => { observed = resolve; });
  globals.__frameAgent = async (_tab: string, op: Record<string, unknown>) => {
    if (op.op === "snapshot") { observed(); return new Promise((resolve) => { release = resolve; }); }
    return { installed: true, result: { ok: true } };
  };
  const posts: Array<{ result: { ok: boolean; text?: string } }> = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => { posts.push(JSON.parse(init.body)); return new Response(null, { status: 200 }); };
  const pending = runBrowserAction({ id: "share-observation", sessionId: "s", op: "snapshot", policy }, { runKey: "four", chat, userText: "example.com" });
  await started;
  shareAgentTab(chat, null);
  release({ installed: true, result: { ok: true, text: "private contents" } });
  await pending;
  assert.equal(posts[0].result.ok, false);
  assert.equal(posts[0].result.text, undefined);
});

test("revoking while an uninstalled page agent answers cannot replay a click", async () => {
  agent.endBrowserAgentRun("four");
  const chat = useBrowserStore.getState().chat;
  const tab = useBrowserStore.getState().activeTabId!;
  assert.equal(shareAgentTab(chat, tab), true);
  let release: (value: unknown) => void = () => {};
  let sent: () => void = () => {};
  const started = new Promise<void>((resolve) => { sent = resolve; });
  let clicks = 0;
  globals.__frameAgent = async (_tab: string, op: Record<string, unknown>) => {
    if (op.op === "describe") return { installed: true, result: { ok: true, info: { role: "button", name: "OK", tag: "button" } } };
    if (op.op === "click") { clicks++; sent(); return new Promise((resolve) => { release = resolve; }); }
    return { installed: true, result: { ok: true } };
  };
  const pending = runBrowserAction({ id: "share-replay", sessionId: "s", op: "click", ref: "r1", policy }, { runKey: "five", chat, userText: "example.com" });
  await started;
  shareAgentTab(chat, null);
  release({ installed: false, result: null });
  await pending;
  assert.equal(clicks, 1);
});
