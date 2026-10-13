// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { answerQuestion, stopBrowserAgent, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };
const buy = { role: "button", name: "Buy now", tag: "button", href: "", formMethod: "", formAction: "", formHasPassword: false, submits: false };

test("Stop pressed while an approved point click is rechecked keeps the click from going out", async () => {
  const posted: { result: { ok: boolean; error?: string } }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  const sent: string[] = [];
  let describes = 0;
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(String(op.op));
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    if (op.op === "describe") {
      // The second describe is the recheck after the user's answer: Stop lands during it.
      if (++describes === 2) stopBrowserAgent();
      return { installed: true, result: { ok: true, info: buy, hit: "r5", sig: "s" } };
    }
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const run = { runKey: "r", userText: "open https://shop.example.com/" };
  await runBrowserAction({ id: "c0", sessionId: "s", op: "navigate", url: "https://shop.example.com/", direction: "down", policy }, run);
  const pending = runBrowserAction({ id: "c1", sessionId: "s", op: "click", x: 40, y: 20, direction: "down", policy }, run);
  let question = useBrowserAgentStore.getState().question;
  for (let i = 0; !question && i < 200; i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    question = useBrowserAgentStore.getState().question;
  }
  assert.ok(question, "the Buy button was not asked about");
  answerQuestion(question.id, true);
  await pending;
  assert.equal(posted.at(-1)?.result.ok, false);
  assert.ok(!sent.includes("click"), sent.join(", "));
});
