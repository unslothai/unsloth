// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { answerQuestion, stopBrowserAgent, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");
const { chatKeyOfThread, useBrowserStore } = await import("../src/features/browser/store.ts");

const globals = globalThis as Record<string, unknown>;
// The stub page loads at once, as tab-view marks a real tab loaded; else each action waits out LOAD_TIMEOUT_MS.
useBrowserStore.subscribe(({ tabs }) => {
  for (const tab of tabs) if (tab.loading) queueMicrotask(() => useBrowserStore.getState().updateTab(tab.id, { loading: false }));
});
globals.__authFetch = async () => new Response(null, { status: 200 });
const policy = { allowedDomains: [], blockedDomains: [] };
const buy = { role: "button", name: "Buy now", tag: "button", href: "", formMethod: "", formAction: "", formHasPassword: false, submits: false };
const sent: string[] = [];
let info: Record<string, unknown> = buy;
let onClick: (() => void) | null = null;
globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
  sent.push(String(op.op));
  if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
  if (op.op === "describe") {
    onClick?.();
    return { installed: true, result: { ok: true, info, hit: "r5", sig: "s" } };
  }
  return { installed: true, result: { ok: true, text: "page", links: [] } };
};

type Run = { runKey: string; scope: string; userText: string };
const chatOf = (scope: string) => chatKeyOfThread(scope);
const show = (scope: string) => useBrowserStore.getState().showChat(chatKeyOfThread(scope));
const pending = () => Object.values(useBrowserAgentStore.getState().chats).find((chat) => chat.question)?.question ?? null;
const navigate = (id: string, run: Run, url = "https://shop.example.com/") =>
  runBrowserAction({ id, sessionId: "s", op: "navigate", url, direction: "down", policy }, run);
const click = (id: string, run: Run) => runBrowserAction({ id, sessionId: "s", op: "click", x: 40, y: 20, direction: "down", policy }, run);
const run = (scope: string): Run => ({ runKey: `run-${scope}`, scope, userText: "open https://shop.example.com/" });

/** What the action asks, answered by `reply`, and whether the bar offered Always allow; null when it doesn't ask. */
async function askedBy(action: Promise<void>, reply: (id: number) => void) {
  let finished = false;
  const done = action.then(() => {
    finished = true;
  });
  let asked = pending();
  while (!finished && !asked) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    asked = pending();
  }
  if (asked) reply(asked.id);
  await done;
  return asked ? `${asked.ask}${asked.always ? "+always" : ""}` : null;
}

test("Always allow stops asking about that action on that site, in that chat only", async () => {
  info = buy;
  const a = run("chat-a");
  show("chat-a");
  await navigate("a0", a);
  assert.equal(await askedBy(click("a1", a), (id) => answerQuestion(id, true, true)), "click+always");
  const clicks = sent.filter((op) => op === "click").length;
  assert.equal(await askedBy(click("a2", a), (id) => answerQuestion(id, false)), null, "asked again after Always allow");
  assert.equal(sent.filter((op) => op === "click").length, clicks + 1);

  const b = run("chat-b");
  show("chat-b");
  await navigate("b0", b);
  assert.equal(await askedBy(click("b1", b), (id) => answerQuestion(id, false)), "click+always", "another chat inherited Always allow");

  show("chat-a");
  assert.equal(await askedBy(click("a3", a), (id) => answerQuestion(id, false)), null, "Always allow lost on coming back to the chat");
});

test("Always allow on a click covers only the button it named", async () => {
  const c = run("chat-c");
  show("chat-c");
  info = buy;
  await navigate("c0", c);
  assert.equal(await askedBy(click("c1", c), (id) => answerQuestion(id, true, true)), "click+always");
  info = { ...buy, name: "Delete account" };
  assert.equal(await askedBy(click("c2", c), (id) => answerQuestion(id, true)), "click+always");
  info = buy;
  assert.equal(await askedBy(click("c3", c), (id) => answerQuestion(id, true)), null);
});

test("Always allow isn't offered or kept for a new chat's first turn before it has an id", async () => {
  info = buy;
  const first = run("__default");
  useBrowserStore.getState().showChat("new:e");
  await navigate("e0", first);
  assert.equal(await askedBy(click("e1", first), (id) => answerQuestion(id, true, true)), "click");
  assert.equal(await askedBy(click("e2", first), (id) => answerQuestion(id, true)), "click");
});

test("Always allow isn't offered or kept for a native Back whose destination is unknown", async () => {
  const g = run("chat-g");
  show("chat-g");
  await navigate("g0", g);
  const tabId = useBrowserAgentStore.getState().chats[chatOf("chat-g")]?.tabId ?? "";
  globals.__native = true;
  globals.__callNative = async (command: string, args: { command: { op: Record<string, unknown> } }) =>
    command === "browser_view_agent" ? (globals.__frameAgent as (t: string, op: Record<string, unknown>) => unknown)(tabId, args.command.op) : null;
  try {
    // History from before this chat: the view can go back, but not to where.
    const back = (id: string) => {
      useBrowserStore.getState().updateTab(tabId, { nativeHistory: { back: true, forward: false } });
      return runBrowserAction({ id, sessionId: "s", op: "back", direction: "down", policy }, g);
    };
    assert.equal(await askedBy(back("g1"), (id) => answerQuestion(id, true, true)), "open");
    // The user browsed on in the view, so where Back leads is unknown again.
    useBrowserStore.getState().updateTab(tabId, { displayUrl: "https://shop.example.com/elsewhere" });
    assert.equal(await askedBy(back("g2"), (id) => answerQuestion(id, false)), "open", "an unnamed Back reused Always allow");
  } finally {
    globals.__native = false;
    delete globals.__callNative;
  }
});

test("Always allow doesn't answer for an action that was stopped", async () => {
  info = buy;
  const d = run("chat-d");
  show("chat-d");
  await navigate("d0", d);
  assert.equal(await askedBy(click("d1", d), (id) => answerQuestion(id, true, true)), "click+always");
  const clicks = sent.filter((op) => op === "click").length;
  // Stop lands while the next click on that button is being read, before it is asked about.
  onClick = () => stopBrowserAgent(chatOf("chat-d"));
  try {
    await click("d2", d);
  } finally {
    onClick = null;
  }
  assert.equal(sent.filter((op) => op === "click").length, clicks, "a stopped click ran on Always allow");
});
