// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { agentChat, answerQuestion, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const CHAT = "new";

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };
useBrowserStore.subscribe(({ tabs }) => {
  for (const tab of tabs) if (tab.loading) queueMicrotask(() => useBrowserStore.getState().updateTab(tab.id, { loading: false }));
});

async function evaluate(url: string, id: string) {
  const posted: { result: { ok: boolean; text?: string; error?: string } }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  const sent: string[] = [];
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(String(op.op));
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    if (op.op === "evaluated") return { installed: true, result: { ok: true, pending: false, text: "42" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const run = { runKey: "r", userText: `open ${url}` };
  await runBrowserAction({ id: `${id}n`, sessionId: "s", op: "navigate", url, direction: "down", policy }, run);
  const pending = runBrowserAction({ id, sessionId: "s", op: "evaluate", script: "return 42;", direction: "down", policy }, run);
  let question = agentChat(CHAT).question;
  for (let i = 0; !question && i < 50; i++) {
    const done = await Promise.race([pending.then(() => true), new Promise((resolve) => setTimeout(() => resolve(false), 10))]);
    if (done) break;
    question = agentChat(CHAT).question;
  }
  if (question) answerQuestion(question.id, false);
  await pending;
  return { asked: question?.ask ?? null, ran: sent.includes("evaluate"), result: posted.at(-1)?.result };
}

test("Run scripts Always allow runs the script without asking", async () => {
  useBrowserAgentStore.setState({ scriptAccess: "allow" });
  const { asked, ran, result } = await evaluate("https://example.com/", "a1");
  useBrowserAgentStore.setState({ scriptAccess: "ask" });
  assert.equal(asked, null);
  assert.ok(ran);
  assert.equal(result?.ok, true);
});

test("a site's Scripts rule skips the prompt only on that site", async () => {
  const { DEFAULT_RULE } = await import("../src/features/browser/agent/permissions.ts");
  useBrowserAgentStore.setState({ scriptAccess: "ask" });
  useBrowserAgentStore.getState().setSiteRule("dash.example", { ...DEFAULT_RULE, browsing: "allow", scripts: "allow" });
  useBrowserAgentStore.getState().setSiteRule("other.example", { ...DEFAULT_RULE, browsing: "allow" });
  try {
    const trusted = await evaluate("https://eu.dash.example/", "a2");
    assert.equal(trusted.asked, null);
    assert.ok(trusted.ran);
    const other = await evaluate("https://other.example/", "a3");
    assert.equal(other.asked, "script");
    assert.ok(!other.ran);
  } finally {
    useBrowserAgentStore.getState().setSiteRule("dash.example", null);
    useBrowserAgentStore.getState().setSiteRule("other.example", null);
  }
});
