// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

// Chats use the browser tools side by side, each in its own tab. A chat off screen works on in the
// background (native pages), leaving the user's tabs alone, and its questions wait in its own bar.
register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction, STOPPED } = await import("../src/features/browser/agent/executor.ts");
const { agentChat, endBrowserAgentRun, stopBrowserAgent, useBrowserAgentStore } = await import(
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
// The page of a native view answers through browser_view_agent; a framed one through the frame.
globals.__callNative = async (command: string, args: { command: { op: Record<string, unknown> } }) =>
  command === "browser_view_agent" ? { installed: true, result: answer(args.command.op) } : null;
globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => ({ installed: true, result: answer(op) });
const kept: string[] = [];
// A page opened in the background reports its load a beat later.
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
  // The chat on screen shows its own agent tab beside the user's; the other chat's tab is not in its strip.
  assert.deepEqual(
    chatTabs(useBrowserStore.getState()).map((tab) => tab.id),
    [mine, here],
  );
  assert.equal(useBrowserStore.getState().activeTabId, here);
  // The background page stays open while its run lasts, and is let go when it ends.
  assert.ok(kept.includes(away ?? ""));
  endBrowserAgentRun("run-away");
  assert.ok((globals.__released as string[]).includes(away ?? ""));
  // Going back to that chat shows its agent tab in an open panel.
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
  // Stop in the shown chat's bar leaves the other chat's question open.
  stopBrowserAgent(chatKeyOfThread("shown"));
  assert.equal(agentChat(away).question?.id, question?.id);
  stopBrowserAgent(away);
  await pending;
  assert.equal(resultOf("q1")?.error, STOPPED);
});

test("without native pages a chat off screen is told to wait for its chat", async () => {
  globals.__native = false;
  await navigate("w1", "elsewhere", "https://elsewhere.example/");
  assert.equal(resultOf("w1")?.ok, false);
  assert.match(resultOf("w1")?.error ?? "", /only runs while the chat is open here/);
  assert.equal(agentChat(chatKeyOfThread("elsewhere")).tabId, null);
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
  // The next action names the saved chat and reads the same page, not a second one.
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
  // The run started in the new chat, which the user has since left: its page opens there, not here.
  await runBrowserAction(
    { id: "b1", sessionId: "s", op: "navigate", url: "https://bound.example/", direction: "down", policy },
    { runKey: "run-bound", scope: "__default", chat: "single:draft2", userText: "open https://bound.example/" },
  );
  assert.equal(resultOf("b1")?.ok, true, resultOf("b1")?.error);
  const tabId = agentChat("single:draft2").tabId;
  assert.equal(useBrowserStore.getState().tabs.find((tab) => tab.id === tabId)?.chat, "single:draft2");
  assert.equal(chatTabs(useBrowserStore.getState()).length, 0);

  // Saved while an action waits on the user: Stop there ends it, and its question goes with it.
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
  // The saved chat on screen is not temporary: only the run itself says its pages are.
  store.showChat(chatKeyOfThread("saved-shown"));
  const run = (id: string, url: string) =>
    runBrowserAction(
      { id, sessionId: "s", op: "navigate", url, direction: "down", policy },
      { runKey: "run-hidden-temp", scope: "hidden-temp", temporary: true, userText: `open ${url}` },
    );
  // The first opens the run's tab in its own chat; the second moves that tab on.
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
