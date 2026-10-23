// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { agentChat, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const CHAT = "new";

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };
type Posted = { action_id: string; result: { ok: boolean; url?: string; text?: string; note?: string; error?: string } };
const posted: Posted[] = [];
globals.__authFetch = async (_url: string, init: { body: string }) => {
  posted.push(JSON.parse(init.body));
  return new Response(null, { status: 200 });
};
useBrowserStore.subscribe(({ tabs }) => {
  for (const tab of tabs) if (tab.loading) queueMicrotask(() => useBrowserStore.getState().updateTab(tab.id, { loading: false }));
});
const steps: Array<{ kind: string; detail: string } | null> = [];
const search = { role: "button", name: "Search", tag: "button", type: "", href: "", autocomplete: "", formMethod: "get", formAction: "", formHasPassword: false, submits: false };
globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
  steps.push(agentChat(CHAT).step);
  if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
  if (op.op === "describe") return { installed: true, result: { ok: true, info: search } };
  return { installed: true, result: { ok: true, text: "page", links: [] } };
};
const run = { runKey: "r", userText: "open https://example.com/" };
const resultOf = (id: string) => posted.find((post) => post.action_id === id)?.result;

test("the bar names each step: the host being opened, then the element's words once the page described it", async () => {
  await runBrowserAction({ id: "s0", sessionId: "s", op: "navigate", url: "https://example.com/", direction: "down", policy }, run);
  assert.equal(resultOf("s0")?.ok, true, resultOf("s0")?.error);
  assert.deepEqual(steps[0], { kind: "open", detail: "example.com" });
  assert.equal(agentChat(CHAT).step, null, "the step outlives its action");
  steps.length = 0;
  await runBrowserAction({ id: "s1", sessionId: "s", op: "click", ref: "r7", direction: "down", policy }, run);
  assert.equal(resultOf("s1")?.ok, true, resultOf("s1")?.error);
  assert.deepEqual(steps[0], { kind: "click", detail: "r7" });
  assert.ok(steps.some((step) => step?.kind === "click" && step.detail === "Search"), JSON.stringify(steps));
});

test("Back with no page behind it stays put and says so", async () => {
  await runBrowserAction({ id: "b1", sessionId: "s", op: "back", direction: "down", policy }, run);
  const result = resultOf("b1");
  assert.equal(result?.ok, true, result?.error);
  assert.equal(result?.url, "https://example.com/");
  assert.match(result?.note ?? "", /no previous page/);
  const tab = useBrowserStore.getState().tabs.find((candidate) => candidate.id === agentChat(CHAT).tabId);
  assert.equal(tab?.history.length, 1);
});

test("Back with no page behind it still holds the page it stays on to the site rules", async () => {
  const block = { browsing: "block", actions: "ask", downloads: "ask", unlabelled: "ask" } as const;
  useBrowserAgentStore.getState().setSiteRule("example.com", block);
  try {
    await runBrowserAction({ id: "b2", sessionId: "s", op: "back", direction: "down", policy }, run);
    const result = resultOf("b2");
    assert.equal(result?.ok, false);
    assert.equal(result?.text, undefined);
    assert.match(result?.error ?? "", /blocked example\.com/);
  } finally {
    useBrowserAgentStore.getState().setSiteRule("example.com", null);
  }
});

test("Back through a router's own history goes back in the page", async () => {
  const tabId = agentChat(CHAT).tabId ?? "";
  const store = useBrowserStore.getState();
  store.updateTab(tabId, { pageHistory: { back: true, forward: false } });
  const sent: Array<Record<string, unknown>> = [];
  globals.__frameCommand = (id: string, command: Record<string, unknown>) => {
    sent.push(command);
    store.updateTab(id, { displayUrl: "https://example.com/list" });
    return true;
  };
  try {
    await runBrowserAction({ id: "b3", sessionId: "s", op: "back", direction: "down", policy }, run);
  } finally {
    delete globals.__frameCommand;
    store.updateTab(tabId, { pageHistory: null });
  }
  const result = resultOf("b3");
  assert.equal(result?.ok, true, result?.error);
  assert.deepEqual(sent, [{ command: "go", delta: -1 }]);
  assert.doesNotMatch(result?.note ?? "", /no previous page/);
});
