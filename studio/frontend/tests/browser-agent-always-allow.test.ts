// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { answerQuestion, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };
const buy = { role: "button", name: "Buy now", tag: "button", href: "", formMethod: "", formAction: "", formHasPassword: false, submits: false };

async function question() {
  for (let i = 0; i < 100; i++) {
    const asked = useBrowserAgentStore.getState().question;
    if (asked) return asked;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  return null;
}

/** What the action asks, answered by `reply`; null when it finishes without asking. */
async function askedBy(action: Promise<void>, reply: (id: number) => void) {
  let finished = false;
  const done = action.then(() => {
    finished = true;
  });
  let asked = useBrowserAgentStore.getState().question;
  while (!finished && !asked) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    asked = useBrowserAgentStore.getState().question;
  }
  if (asked) reply(asked.id);
  await done;
  return asked?.ask ?? null;
}

test("Always allow stops asking about that action on that site, in that chat only", async () => {
  globals.__authFetch = async () => new Response(null, { status: 200 });
  const sent: string[] = [];
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(String(op.op));
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info: buy, hit: "r5", sig: "s" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const click = (id: string, run: { runKey: string; scope: string; userText: string }) =>
    runBrowserAction({ id, sessionId: "s", op: "click", x: 40, y: 20, direction: "down", policy }, run);
  const chat = { runKey: "r", scope: "chat-a", userText: "open https://shop.example.com/" };
  await runBrowserAction({ id: "n0", sessionId: "s", op: "navigate", url: "https://shop.example.com/", direction: "down", policy }, chat);

  const first = click("c1", chat);
  const asked = await question();
  assert.equal(asked?.ask, "click");
  answerQuestion(asked.id, true, true);
  await first;
  assert.equal(sent.filter((op) => op === "click").length, 1);

  const second = click("c2", chat);
  const again = await question();
  if (again) answerQuestion(again.id, false);
  await second;
  assert.equal(again, null, "asked again after Always allow");
  assert.equal(sent.filter((op) => op === "click").length, 2);

  const other = { runKey: "r", scope: "chat-b", userText: "" };
  const third = click("c3", other);
  const asks: string[] = [];
  for (let next = await question(); next; next = await question()) {
    asks.push(next.ask);
    answerQuestion(next.id, next.ask !== "click");
  }
  await third;
  assert.ok(asks.includes("click"), `another chat inherited Always allow: ${asks.join(", ")}`);

  assert.equal(await askedBy(click("c4", chat), (id) => answerQuestion(id, false)), null, "Always allow lost on coming back to the chat");
});

test("Always allow on a click covers only the button it named", async () => {
  globals.__authFetch = async () => new Response(null, { status: 200 });
  let info: Record<string, unknown> = buy;
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info, hit: "r5", sig: "s" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const chat = { runKey: "r", scope: "chat-c", userText: "open https://shop.example.com/" };
  await runBrowserAction({ id: "d0", sessionId: "s", op: "navigate", url: "https://shop.example.com/", direction: "down", policy }, chat);
  const askedFor = (id: string, next: Record<string, unknown>, always: boolean) => {
    info = next;
    const click = runBrowserAction({ id, sessionId: "s", op: "click", x: 40, y: 20, direction: "down", policy }, chat);
    return askedBy(click, (asked) => answerQuestion(asked, true, always));
  };

  assert.equal(await askedFor("d1", buy, true), "click");
  assert.equal(await askedFor("d2", { ...buy, name: "Delete account" }, false), "click");
  assert.equal(await askedFor("d3", buy, false), null);
});

test("Always allow isn't kept for a new chat's first turn before it has an id", async () => {
  globals.__authFetch = async () => new Response(null, { status: 200 });
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info: buy, hit: "r5", sig: "s" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const chat = { runKey: "r", scope: "__default", userText: "open https://shop.example.com/" };
  await runBrowserAction({ id: "e0", sessionId: "s", op: "navigate", url: "https://shop.example.com/", direction: "down", policy }, chat);
  const click = (id: string) => runBrowserAction({ id, sessionId: "s", op: "click", x: 40, y: 20, direction: "down", policy }, chat);
  assert.equal(await askedBy(click("e1"), (id) => answerQuestion(id, true, true)), "click");
  assert.equal(await askedBy(click("e2"), (id) => answerQuestion(id, true)), "click");
});
