// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A search engine's redirect link opens its target (the store unwraps it), so the new-site
// question has to be about the target, not about the search engine the link starts on.
import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { answerQuestion, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");
const { chatKeyOfThread, useBrowserStore } = await import("../src/features/browser/store.ts");

const globals = globalThis as Record<string, unknown>;
useBrowserStore.subscribe(({ tabs }) => {
  for (const tab of tabs) if (tab.loading) queueMicrotask(() => useBrowserStore.getState().updateTab(tab.id, { loading: false }));
});
globals.__authFetch = async () => new Response(null, { status: 200 });
const policy = { allowedDomains: [], blockedDomains: [] };
let links: string[] = [];
globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
  if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
  return { installed: true, result: { ok: true, text: "page", links } };
};

type Run = { runKey: string; scope: string; userText: string };
const run = (scope: string): Run => ({ runKey: `run-${scope}`, scope, userText: "search https://www.google.com/ for qwen" });
const show = (scope: string) => useBrowserStore.getState().showChat(chatKeyOfThread(scope));
const pending = () => Object.values(useBrowserAgentStore.getState().chats).find((chat) => chat.question)?.question ?? null;
const navigate = (id: string, context: Run, url: string) =>
  runBrowserAction({ id, sessionId: "s", op: "navigate", url, direction: "down", policy }, context);
const snapshot = (id: string, context: Run) =>
  runBrowserAction({ id, sessionId: "s", op: "snapshot", direction: "down", policy }, context);
const opened = () => useBrowserStore.getState().tabs.map((tab) => tab.history.map((entry) => ("url" in entry ? entry.url : "")).join(" ")).join(" ");

/** The host the action asks about (denied), and whether the target was already open when it asked; null when it doesn't ask. */
async function askedBy(action: Promise<void>, target: string) {
  let finished = false;
  const done = action.then(() => {
    finished = true;
  });
  let asked = pending();
  while (!finished && !asked) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    asked = pending();
  }
  const openedFirst = opened().includes(target);
  if (asked) answerQuestion(asked.id, false);
  await done;
  return asked ? { host: asked.host, openedFirst } : null;
}

test("a redirect link to a new site asks about that site before the tab loads it", async () => {
  const context = run("a");
  show("a");
  const target = "https://collector.example/x";
  const asked = await askedBy(navigate("a1", context, `https://www.google.com/url?q=${encodeURIComponent(target)}`), "collector.example");
  assert.deepEqual(asked, { host: "collector.example", openedFirst: false });
  assert.equal(opened().includes("collector.example"), false, "the denied target was opened anyway");
});

test("a DuckDuckGo redirect link is gated on its target too", async () => {
  const context = run("b");
  show("b");
  const asked = await askedBy(navigate("b1", context, "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fcollector2.example%2F"), "collector2.example");
  assert.equal(asked?.host, "collector2.example");
});

test("a plain search page the user asked for still opens without a question", async () => {
  const context = run("c");
  show("c");
  assert.equal(await askedBy(navigate("c1", context, "https://www.google.com/search?q=qwen"), "google.com"), null);
});

test("a redirect link the open page shows opens its target without a question, like any link there", async () => {
  const context = run("d");
  show("d");
  assert.equal(await askedBy(navigate("d1", context, "https://www.google.com/search?q=qwen"), "google.com"), null);
  const wrapped = "https://www.google.com/url?q=https://result.example/page&sa=U";
  links = [wrapped];
  await snapshot("d2", context);
  links = [];
  assert.equal(await askedBy(navigate("d3", context, wrapped), "result.example"), null);
  assert.equal(opened().includes("https://result.example/page"), true, "the linked result did not open");
  // Being on the page links only that result: another site the model names still asks.
  assert.equal(
    (await askedBy(navigate("d4", context, `https://www.google.com/url?q=${encodeURIComponent("https://unlisted.example/")}`), "unlisted.example"))
      ?.host,
    "unlisted.example",
  );
});

test("a redirect link the page shows is still refused when its target is blocked", async () => {
  const context = run("f");
  show("f");
  assert.equal(await askedBy(navigate("f1", context, "https://www.google.com/search?q=qwen"), "google.com"), null);
  const wrapped = "https://www.google.com/url?q=https://blocked.example/page&sa=U";
  links = [wrapped];
  await snapshot("f2", context);
  links = [];
  await runBrowserAction(
    { id: "f3", sessionId: "s", op: "navigate", url: wrapped, direction: "down", policy: { allowedDomains: [], blockedDomains: ["blocked.example"] } },
    context,
  );
  assert.equal(pending(), null);
  assert.equal(opened().includes("blocked.example/page"), false, "the blocked target was opened");
});

test("a clicked search-result redirect link opens its target without a question", async () => {
  const context = run("e");
  show("e");
  assert.equal(await askedBy(navigate("e1", context, "https://www.google.com/search?q=qwen"), "google.com"), null);
  const wrapped = "https://www.google.com/url?q=https://clicked.example/page&sa=U";
  const link = { role: "link", name: "Result", tag: "a", href: wrapped, formMethod: "", formAction: "", formHasPassword: false, submits: false };
  const original = globals.__frameAgent;
  links = [wrapped];
  globals.__frameAgent = async (tabId: string, op: Record<string, unknown>) => {
    if (op.op === "describe") return { installed: true, result: { ok: true, info: link, hit: "r5", sig: "s" } };
    if (op.op === "click") return { installed: true, result: { ok: true, navigate: wrapped } };
    return (original as (t: string, o: Record<string, unknown>) => unknown)(tabId, op);
  };
  try {
    await snapshot("e2", context);
    const click = runBrowserAction({ id: "e3", sessionId: "s", op: "click", ref: "r5", direction: "down", policy }, context);
    assert.equal(await askedBy(click, "clicked.example"), null);
    assert.equal(opened().includes("https://clicked.example/page"), true, "the clicked result did not open");
  } finally {
    globals.__frameAgent = original;
    links = [];
  }
});
