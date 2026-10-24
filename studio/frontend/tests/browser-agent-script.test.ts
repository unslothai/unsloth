// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { agentChat, answerQuestion, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");
// Actions without a scope act for the chat on screen.
const CHAT = "new";

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };

test("a script approved after Run scripts was set to Never does not run", async () => {
  const posted: { result: { ok: boolean; error?: string } }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  const sent: string[] = [];
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(String(op.op));
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const run = { runKey: "r", userText: "open https://example.com/" };
  await runBrowserAction({ id: "e0", sessionId: "s", op: "navigate", url: "https://example.com/", direction: "down", policy }, run);
  useBrowserAgentStore.setState({ scriptAccess: "ask" });
  const pending = runBrowserAction({ id: "e1", sessionId: "s", op: "evaluate", script: "return 1;", direction: "down", policy }, run);
  let question = agentChat(CHAT).question;
  for (let i = 0; !question && i < 200; i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    question = agentChat(CHAT).question;
  }
  assert.equal(question?.ask, "script");
  useBrowserAgentStore.setState({ scriptAccess: "never" });
  answerQuestion(question.id, true);
  await pending;
  assert.equal(posted.at(-1)?.result.ok, false);
  assert.match(posted.at(-1)?.result.error ?? "", /doesn't let the agent run scripts/);
  assert.ok(!sent.includes("evaluate"));
});
