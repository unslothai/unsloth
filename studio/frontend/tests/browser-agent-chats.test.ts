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
