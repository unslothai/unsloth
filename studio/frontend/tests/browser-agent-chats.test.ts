// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction, STOPPED } = await import("../src/features/browser/agent/executor.ts");
const { agentChat, browserAgentEnabled, endBrowserAgentRun, patchAgentChat, setBrowserAgentEnabled, stopBrowserAgent, useBrowserAgentStore } = await import(
  "../src/features/browser/agent/agent-store.ts"
);
const { chatKeyOfThread, chatTabs, currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };
type Posted = { action_id: string; result: { ok: boolean; url?: string; error?: string } };
const posted: Posted[] = [];
globals.__authFetch = async (_url: string, init: { body: string }) => {
  posted.push(JSON.parse(init.body));
  return new Response(null, { status: 200 });
};
const answer = (op: Record<string, unknown>) =>
  op.op === "status"
    ? { ok: true, quietMs: 1000, doc: "d1" }
    : { ok: true, text: "page", links: [], width: 800, height: 600, doc: "d1" };
globals.__callNative = async (command: string, args: { command: { op: Record<string, unknown> } }) =>
  command === "browser_view_agent" ? { installed: true, result: answer(args.command.op) } : null;
globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => ({ installed: true, result: answer(op) });
const kept: string[] = [];
globals.__keep = (tabId: string) => {
  kept.push(tabId);
  setTimeout(() => useBrowserStore.getState().updateTab(tabId, { loading: false }), 30);
};

const navigate = (id: string, scope: string, url: string) =>
  runBrowserAction({ id, sessionId: "s", op: "navigate", url, direction: "down", policy }, { runKey: `run-${scope}`, scope, userText: `open ${url}` });
const resultOf = (id: string) => posted.find((post) => post.action_id === id)?.result;
const urlOf = (tabId: string | null) => {
  const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === tabId);
  const entry = tab && currentEntry(tab);
  return entry?.kind === "web" ? entry.url : null;
};

test("two chats browse at once, each in its own tab; the one off screen leaves the user's tabs alone", async () => {
  globals.__native = true;
  const store = useBrowserStore.getState();
  store.showChat(chatKeyOfThread("shown"));
  store.openUrl("https://mine.example/");
  const mine = useBrowserStore.getState().activeTabId;

  await Promise.all([navigate("p1", "away", "https://away.example/"), navigate("p2", "shown", "https://here.example/")]);
  assert.equal(resultOf("p1")?.ok, true, resultOf("p1")?.error);
  assert.equal(resultOf("p2")?.ok, true, resultOf("p2")?.error);
  const away = agentChat(chatKeyOfThread("away")).tabId;
  const here = agentChat(chatKeyOfThread("shown")).tabId;
  assert.equal(urlOf(away), "https://away.example/");
  assert.equal(urlOf(here), "https://here.example/");
  assert.deepEqual(
    chatTabs(useBrowserStore.getState()).map((tab) => tab.id),
    [mine, here],
  );
  assert.equal(useBrowserStore.getState().activeTabId, here);
  assert.ok(kept.includes(away ?? ""));
  endBrowserAgentRun("run-away");
  assert.equal(((globals.__released ?? []) as string[]).includes(away ?? ""), false);
  store.showChat(chatKeyOfThread("away"));
  assert.equal(useBrowserStore.getState().activeTabId, away);
  assert.equal(useBrowserStore.getState().open, true);
  store.showChat(chatKeyOfThread("shown"));
});

test("a question in a chat off screen waits in that chat, with a toast; Stop there ends only its run", async () => {
  globals.__native = true;
  useBrowserAgentStore.setState({ scriptAccess: "ask" });
  const away = chatKeyOfThread("away");
  await navigate("q0", "away", "https://away.example/next");
  const pending = runBrowserAction(
    { id: "q1", sessionId: "s", op: "evaluate", script: "return 1;", direction: "down", policy },
    { runKey: "run-away", scope: "away", userText: "" },
  );
  let question = agentChat(away).question;
  for (let i = 0; !question && i < 200; i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    question = agentChat(away).question;
  }
  assert.equal(question?.ask, "script");
  assert.equal(agentChat(chatKeyOfThread("shown")).question, null, "the chat on screen isn't asked");
  const toasts = (globals.__toasts ?? []) as { options?: { id?: string } }[];
  assert.ok(toasts.some((toast) => toast.options?.id === `browser-agent-question:${away}`));
  stopBrowserAgent(chatKeyOfThread("shown"));
  assert.equal(agentChat(away).question?.id, question?.id);
  stopBrowserAgent(away);
  await pending;
  assert.equal(resultOf("q1")?.error, STOPPED);
});

test("without native pages a chat off screen keeps working", async () => {
  globals.__native = false;
  useBrowserStore.getState().showChat(chatKeyOfThread("elsewhere"));
  await navigate("w1", "elsewhere", "https://elsewhere.example/");
  useBrowserStore.getState().showChat(chatKeyOfThread("shown"));
  await runBrowserAction({ id: "w2", sessionId: "s", op: "snapshot", policy }, { runKey: "run-elsewhere", scope: "elsewhere", userText: "" });
  assert.equal(resultOf("w2")?.ok, true, resultOf("w2")?.error);
});

test("per-chat toggles and delayed actions stay isolated after a switch", async () => {
  globals.__native = false;
  const store = useBrowserStore.getState();
  const a = chatKeyOfThread("overlap-a"), b = chatKeyOfThread("overlap-b");
  useBrowserAgentStore.setState({ enabled: true });
  setBrowserAgentEnabled(a, false);
  assert.equal(browserAgentEnabled(a), false);
  assert.equal(browserAgentEnabled(b), true);
  setBrowserAgentEnabled(a, true);
  store.showChat(a);
  await navigate("overlap-nav-a", "overlap-a", "https://a.example/");
  store.showChat(b);
  await navigate("overlap-nav-b", "overlap-b", "https://b.example/");
  const tabA = agentChat(a).tabId, tabB = agentChat(b).tabId;
  assert.ok(tabA && tabB && tabA !== tabB);
  const started: string[] = [];
  const release = new Map<string, () => void>();
  globals.__frameAgent = async (tabId: string, op: Record<string, unknown>) => {
    if (op.op === "snapshot") {
      started.push(tabId);
      await new Promise<void>((resolve) => release.set(tabId, resolve));
    }
    return { installed: true, result: answer(op) };
  };
  const action = (id: string, scope: string) => runBrowserAction(
    { id, sessionId: "s", op: "snapshot", policy }, { runKey: `run-${scope}`, scope, userText: "" },
  );
  const first = action("overlap-a", "overlap-a"), second = action("overlap-b", "overlap-b");
  for (let i = 0; started.length < 2 && i < 200; i++) await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(new Set(started), new Set([tabA, tabB]));
  store.showChat(a);
  setBrowserAgentEnabled(b, false);
  endBrowserAgentRun("run-overlap-b");
  release.get(tabA)?.();
  release.get(tabB)?.();
  await Promise.all([first, second]);
  assert.equal(resultOf("overlap-a")?.ok, true, resultOf("overlap-a")?.error);
  assert.equal(resultOf("overlap-b")?.error, STOPPED);
  assert.equal(browserAgentEnabled(a), true);
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => ({ installed: true, result: answer(op) });
});

test("a forged target belonging to another chat cannot be read", async () => {
  globals.__native = false;
  const a = chatKeyOfThread("overlap-a"), b = chatKeyOfThread("overlap-b");
  patchAgentChat(b, { tabId: agentChat(a).tabId });
  await runBrowserAction({ id: "forged", sessionId: "s", op: "snapshot", policy }, { runKey: "forged-run", scope: "overlap-b", userText: "" });
  assert.equal(resultOf("forged")?.ok, false);
  assert.match(resultOf("forged")?.error ?? "", /No page is open/);
});

test("a compare pane's run drives its pair's browser, whatever the sidebar shows", async () => {
  globals.__native = true;
  await runBrowserAction(
    { id: "c1", sessionId: "s", op: "navigate", url: "https://pair.example/", direction: "down", policy },
    { runKey: "run-pane", scope: "pane-thread", chat: "compare:pair1", userText: "open https://pair.example/" },
  );
  assert.equal(resultOf("c1")?.ok, true, resultOf("c1")?.error);
  const tabId = agentChat("compare:pair1").tabId;
  assert.equal(useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.chat, "compare:pair1");
  assert.equal(agentChat(chatKeyOfThread("pane-thread")).tabId, null);
});

test("a new chat's run keeps its page when the chat is saved between actions", async () => {
  globals.__native = true;
  const store = useBrowserStore.getState();
  store.closePanel();
  store.showChat("single:draft");
  await navigate("d1", "__default", "https://draft.example/");
  assert.equal(resultOf("d1")?.ok, true, resultOf("d1")?.error);
  const tabId = agentChat("single:draft").tabId;
  assert.ok(tabId);
  store.adoptChat("single:draft", chatKeyOfThread("saved"));
  assert.equal(agentChat(chatKeyOfThread("saved")).tabId, tabId);
  assert.equal("single:draft" in useBrowserAgentStore.getState().chats, false);
  await runBrowserAction(
    { id: "d2", sessionId: "s", op: "snapshot", direction: "down", policy },
    { runKey: "run-__default", scope: "saved", userText: "" },
  );
  assert.equal(resultOf("d2")?.ok, true, resultOf("d2")?.error);
  assert.equal(useBrowserStore.getState().tabs.filter((tab) => tab.chat === chatKeyOfThread("saved")).length, 1);
});

test("a new chat's run stays in it after a switch, and Stop in the saved chat ends the action in flight", async () => {
  globals.__native = true;
  const store = useBrowserStore.getState();
  store.closePanel();
  store.showChat("single:draft2");
  store.showChat(chatKeyOfThread("elsewhere2"));
  await runBrowserAction(
    { id: "b1", sessionId: "s", op: "navigate", url: "https://bound.example/", direction: "down", policy },
    { runKey: "run-bound", scope: "__default", chat: "single:draft2", userText: "open https://bound.example/" },
  );
  assert.equal(resultOf("b1")?.ok, true, resultOf("b1")?.error);
  const tabId = agentChat("single:draft2").tabId;
  assert.equal(useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.chat, "single:draft2");
  assert.equal(chatTabs(useBrowserStore.getState()).length, 0);

  useBrowserAgentStore.setState({ scriptAccess: "ask" });
  const pending = runBrowserAction(
    { id: "b2", sessionId: "s", op: "evaluate", script: "return 1;", direction: "down", policy },
    { runKey: "run-bound", scope: "__default", chat: "single:draft2", userText: "" },
  );
  for (let i = 0; !agentChat("single:draft2").question && i < 200; i++) await new Promise((resolve) => setTimeout(resolve, 10));
  store.adoptChat("single:draft2", chatKeyOfThread("bound"));
  assert.ok(agentChat(chatKeyOfThread("bound")).question);
  stopBrowserAgent(chatKeyOfThread("bound"));
  await pending;
  assert.equal(resultOf("b2")?.error, STOPPED);
  assert.equal(agentChat(chatKeyOfThread("bound")).question, null);
});


test("a temporary chat's run keeps its pages out of history after a switch to a saved chat", async () => {
  globals.__native = true;
  const store = useBrowserStore.getState();
  store.showChat(chatKeyOfThread("saved-shown"));
  const run = (id: string, url: string) =>
    runBrowserAction(
      { id, sessionId: "s", op: "navigate", url, direction: "down", policy },
      { runKey: "run-hidden-temp", scope: "hidden-temp", temporary: true, userText: `open ${url}` },
    );
  await run("t1", "https://private.example/");
  await run("t2", "https://private.example/next");
  assert.equal(resultOf("t1")?.ok, true, resultOf("t1")?.error);
  assert.equal(resultOf("t2")?.ok, true, resultOf("t2")?.error);
  const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === agentChat(chatKeyOfThread("hidden-temp")).tabId);
  assert.deepEqual(
    tab?.history.map((entry) => (entry.kind === "web" ? [entry.url, entry.temporary] : null)),
    [
      ["https://private.example/", true],
      ["https://private.example/next", true],
    ],
  );
});

test("a saved chat's run keeps its pages in history while a temporary chat is on screen", async () => {
  globals.__native = true;
  const { useChatRuntimeStore } = await import("@/features/chat");
  const store = useBrowserStore.getState();
  store.showChat(chatKeyOfThread("temp-shown"));
  useChatRuntimeStore.getState().setIncognito(true);
  try {
    const run = (id: string, url: string) =>
      runBrowserAction(
        { id, sessionId: "s", op: "navigate", url, direction: "down", policy },
        { runKey: "run-hidden-saved", scope: "hidden-saved", temporary: false, userText: `open ${url}` },
      );
    await run("s1", "https://kept.example/");
    await run("s2", "https://kept.example/next");
    assert.equal(resultOf("s2")?.ok, true, resultOf("s2")?.error);
    const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === agentChat(chatKeyOfThread("hidden-saved")).tabId);
    assert.deepEqual(
      tab?.history.map((entry) => (entry.kind === "web" ? [entry.url, entry.temporary] : null)),
      [
        ["https://kept.example/", undefined],
        ["https://kept.example/next", undefined],
      ],
    );
  } finally {
    useChatRuntimeStore.getState().setIncognito(false);
  }
});

test("a run's temporary flag comes from its own thread, not a chat shown later", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /const browserTemporary = resolvedThreadId\s*\?\s*isThreadIncognito\(resolvedThreadId\) \|\|\s*\(resolvedThreadId === runtime\.activeThreadId && runtime\.incognito\)\s*:\s*runtime\.incognito;/,
  );
});

test("native background input stays virtual while the selected page keeps normal input", async () => {
  globals.__native = true;
  const store = useBrowserStore.getState();
  const away = chatKeyOfThread("virtual-away");
  const shown = chatKeyOfThread("virtual-shown");
  store.showChat(shown);
  const a = store.openTabIn(away, "https://away.example/")!;
  const b = store.openTabIn(shown, "https://here.example/")!;
  store.updateTab(a, { loading: false });
  store.updateTab(b, { loading: false });
  patchAgentChat(away, { tabId: a, enabled: true, runKey: null });
  patchAgentChat(shown, { tabId: b, enabled: true, runKey: null });
  globals.__shownNative = b;
  const commands: { tabId: string; command: { op: Record<string, unknown> } }[] = [];
  const previous = globals.__callNative;
  globals.__callNative = async (name: string, args: typeof commands[number]) => {
    if (name !== "browser_view_agent") return null;
    commands.push(args);
    return { installed: true, result: answer(args.command.op) };
  };
  try {
    for (const [chat, id] of [[away, a], [shown, b]]) {
      await runBrowserAction({ id: `virtual-${id}`, sessionId: chat, op: "snapshot", policy }, { runKey: `virtual-${id}`, chat, fullAccess: true, userText: "Read the page" });
      assert.equal(resultOf(`virtual-${id}`)?.ok, true);
    }
    assert.ok(commands.some((call) => call.tabId === a));
    assert.ok(commands.some((call) => call.tabId === b));
    assert.ok(commands.filter((call) => call.tabId === a).every((call) => call.command.op.virtualInput === true));
    assert.ok(commands.filter((call) => call.tabId === b).every((call) => call.command.op.virtualInput !== true));
    globals.__shownNative = a;
    commands.length = 0;
    await runBrowserAction({ id: "virtual-switching", sessionId: away, op: "snapshot", policy }, { runKey: `virtual-${a}`, chat: away, fullAccess: true, userText: "Read while my previous view is being hidden" });
    assert.equal(resultOf("virtual-switching")?.ok, true);
    assert.ok(commands.length > 0);
    assert.ok(commands.every((call) => call.tabId === a && call.command.op.virtualInput === true));
    globals.__shownNative = null;
    commands.length = 0;
    await runBrowserAction({ id: "virtual-observing", sessionId: shown, op: "snapshot", policy }, { runKey: `virtual-${b}`, chat: shown, fullAccess: true, userText: "Read my page while I inspect another chat" });
    assert.equal(resultOf("virtual-observing")?.ok, true);
    assert.ok(commands.length > 0);
    assert.ok(commands.every((call) => call.tabId === b && call.command.op.virtualInput === true));
  } finally {
    delete globals.__shownNative;
    globals.__callNative = previous;
    endBrowserAgentRun(`virtual-${a}`);
    endBrowserAgentRun(`virtual-${b}`);
  }
});
