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
  assert.equal(agent.browserAgentEnabled(chat), true);
  assert.equal(agent.browserAgentEnabled("other-chat"), false);
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

test("a queued first action cannot follow a target switch", async () => {
  agent.endBrowserAgentRun("five");
  const store = useBrowserStore.getState();
  const chat = store.chat;
  const first = store.activeTabId!;
  const second = store.openTabIn(chat, "https://example.com/queue")!;
  store.activateTab(first); shareAgentTab(chat, first);
  let release!: () => void; let started!: () => void;
  const posting = new Promise<void>((resolve) => { started = resolve; });
  const calls: string[] = []; const results: Array<{ result: { ok: boolean } }> = [];
  globals.__frameAgent = async (tab: string) => { calls.push(tab); return { installed: true, result: { ok: true, text: "page", links: [] } }; };
  globals.__authFetch = async (_url: string, init: { body: string }) => { results.push(JSON.parse(init.body)); if (results.length === 1) { started(); await new Promise<void>((resolve) => { release = resolve; }); } return new Response(null, { status: 200 }); };
  const old = runBrowserAction({ id: "queue-old", sessionId: "s", op: "snapshot", policy }, { runKey: "queue-old", chat, userText: "example.com" });
  await posting; agent.endBrowserAgentRun("queue-old");
  const queued = runBrowserAction({ id: "queue-new", sessionId: "s", op: "snapshot", policy }, { runKey: "queue-new", chat, userText: "example.com" });
  store.activateTab(second); shareAgentTab(chat, second); release();
  await Promise.all([old, queued]);
  agent.endBrowserAgentRun("queue-new");
  assert.deepEqual(calls, [first]); assert.equal(results[1].result.ok, false);
});

test("revocation does not install after a pending description", async () => {
  const store = useBrowserStore.getState(); const chat = store.chat; shareAgentTab(chat, store.activeTabId!);
  let release!: (value: unknown) => void; let sent!: () => void;
  const started = new Promise<void>((resolve) => { sent = resolve; }); let count = 0;
  globals.__frameAgent = async (_tab: string, op: Record<string, unknown>) => { if (op.op === "describe") { count++; if (count === 1) { sent(); return new Promise((resolve) => { release = resolve; }); } } return { installed: true, result: { ok: true } }; };
  globals.__authFetch = async () => new Response(null, { status: 200 });
  const pending = runBrowserAction({ id: "desc-revoke", sessionId: "s", op: "click", ref: "r1", policy }, { runKey: "desc-run", chat, userText: "example.com" });
  await started; shareAgentTab(chat, null); release({ installed: false, result: null }); await pending;
  agent.endBrowserAgentRun("desc-run"); assert.equal(count, 1);
});

test("run end and reshare cannot revive queued work", async () => {
  const store = useBrowserStore.getState(); const chat = store.chat; const first = store.activeTabId!;
  const second = store.openTabIn(chat, "https://example.com/end")!;
  store.activateTab(first); shareAgentTab(chat, first);
  let release!: (value: unknown) => void; let sent!: () => void;
  const started = new Promise<void>((resolve) => { sent = resolve; }); const calls: string[] = [];
  globals.__frameAgent = async (tab: string) => { calls.push(tab); if (calls.length === 1) { sent(); return new Promise((resolve) => { release = resolve; }); } return { installed: true, result: { ok: true, text: "page", links: [] } }; };
  globals.__authFetch = async () => new Response(null, { status: 200 });
  const pending = runBrowserAction({ id: "end-pending", sessionId: "s", op: "snapshot", policy }, { runKey: "end-run", chat, userText: "example.com" });
  await started;
  const queued = runBrowserAction({ id: "end-queued", sessionId: "s", op: "snapshot", policy }, { runKey: "end-run", chat, userText: "example.com" });
  shareAgentTab(chat, null); agent.endBrowserAgentRun("end-run"); store.activateTab(second); shareAgentTab(chat, second);
  release({ installed: true, result: { ok: true, text: "page", links: [] } }); await Promise.all([pending, queued]);
  assert.deepEqual(calls, [first]);
});

test("a target change before action registration invalidates the captured version", async () => {
  agent.endBrowserAgentRun("end-run");
  const store = useBrowserStore.getState(); const chat = store.chat; const first = store.activeTabId!;
  const second = store.openTabIn(chat, "https://example.com/delayed")!;
  store.activateTab(first); shareAgentTab(chat, first);
  const targetVersion = agent.agentChat(chat).targetVersion;
  store.activateTab(second); shareAgentTab(chat, second);
  store.activateTab(first); shareAgentTab(chat, first);
  assert.equal(shareAgentTab(chat, first), true);
  let calls = 0; let result: { ok: boolean } | undefined;
  globals.__frameAgent = async () => { calls++; return { installed: true, result: { ok: true, text: "page", links: [] } }; };
  globals.__authFetch = async (_url: string, init: { body: string }) => { result = JSON.parse(init.body).result; return new Response(null, { status: 200 }); };
  await runBrowserAction({ id: "delayed-action", sessionId: "s", op: "snapshot", policy }, { runKey: "delayed-run", chat, userText: "example.com", targetVersion });
  assert.equal(calls, 0); assert.equal(result?.ok, false); assert.notEqual(agent.agentChat(chat).runKey, "delayed-run");
  agent.endBrowserAgentRun("delayed-run");
});

test("drag cleanup does not reinstall an absent agent", async () => {
  agent.endBrowserAgentRun("delayed-run");
  const store = useBrowserStore.getState(); const chat = store.chat; shareAgentTab(chat, store.activeTabId!);
  let cancellations = 0; let installedCleanup = 0;
  globals.__frameAgent = async (_tab: string, op: Record<string, unknown>, code?: string) => {
    if (op.op === "describe") return { installed: true, result: { ok: true, info: { role: "button", name: "Move", tag: "div" }, hit: "r1", sig: "s", doc: "doc" } };
    if (op.op === "pointer" && op.phase === "move") agent.stopBrowserAgent(chat);
    if (op.op === "pointer" && op.phase === "cancel") { cancellations++; if (code) installedCleanup++; return { installed: false, result: null }; }
    return { installed: true, result: { ok: true, doc: "doc" } };
  };
  globals.__authFetch = async () => new Response(null, { status: 200 });
  await runBrowserAction({ id: "cleanup-absent", sessionId: "s", op: "drag", x: 10, y: 10, toX: 100, toY: 10, policy }, { runKey: "cleanup-run", chat, userText: "example.com" });
  agent.endBrowserAgentRun("cleanup-run");
  assert.equal(cancellations, 1); assert.equal(installedCleanup, 0);
});

test("late navigation after switching tabs cannot approve the new tab's host", async () => {
  const store = useBrowserStore.getState(); const chat = store.chat;
  async function trial(host: string, late: boolean) {
    const first = store.openTabIn(chat, `https://${host}/first`)!;
    const second = store.openTabIn(chat, `https://${host}/second`)!;
    store.activateTab(first); shareAgentTab(chat, first);
    let release!: (value: unknown) => void; let sent!: () => void;
    const started = new Promise<void>((resolve) => { sent = resolve; });
    globals.__frameAgent = async (_tab: string, op: Record<string, unknown>) => {
      if (op.op === "describe") return { installed: true, result: { ok: true, info: { role: "textbox", name: "Search", tag: "input" } } };
      if (op.op === "type" && late) { sent(); return new Promise((resolve) => { release = resolve; }); }
      return { installed: true, result: { ok: true, text: "page", links: [] } };
    };
    globals.__authFetch = async () => new Response(null, { status: 200 });
    let pending: Promise<void> | undefined;
    if (late) {
      pending = runBrowserAction({ id: `${host}-type`, sessionId: "s", op: "type", ref: "r1", text: "hello", policy }, { runKey: `${host}-old`, chat, userText: host });
      await started;
    }
    store.activateTab(second); shareAgentTab(chat, second);
    if (pending) {
      useBrowserStore.getState().updateTab(first, { displayUrl: `https://${host}/changed` });
      release({ installed: true, result: { ok: true } });
      await pending; agent.endBrowserAgentRun(`${host}-old`);
    }
    let finished = false;
    const next = runBrowserAction({ id: `${host}-next`, sessionId: "s", op: "snapshot", policy }, { runKey: `${host}-next`, chat, userText: "" }).then(() => { finished = true; });
    while (!finished && !agent.agentChat(chat).question) await new Promise((resolve) => setTimeout(resolve, 10));
    const question = agent.agentChat(chat).question;
    if (question) agent.answerQuestion(question.id, false);
    await next; agent.endBrowserAgentRun(`${host}-next`);
    return question?.ask ?? null;
  }
  assert.equal(await trial("control.example.org", false), "open");
  assert.equal(await trial("late.example.org", true), "open");
});
