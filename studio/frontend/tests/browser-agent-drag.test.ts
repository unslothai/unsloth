// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const { stopBrowserAgent } = await import("../src/features/browser/agent/agent-store.ts");

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };
const info = { role: "button", name: "Move", tag: "div", href: "", formMethod: "", formAction: "", formHasPassword: false, submits: false };

test("a drag whose press navigates the tab is not released on the new page", async () => {
  const posted: { result: { ok: boolean; error?: string } }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  const sent: string[] = [];
  let doc = "r1";
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(`${op.op}${op.phase ? `:${op.phase}` : ""}`);
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info, hit: "r5", sig: "s", doc } };
    if (op.op === "pointer" && op.phase === "down") {
      // The page's own mousedown handler loads another document at the same address.
      doc = "r9001";
    }
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const run = { runKey: "r", userText: "open https://example.com/board" };
  await runBrowserAction({ id: "d0", sessionId: "s", op: "navigate", url: "https://example.com/board", direction: "down", policy }, run);
  sent.length = 0;
  await runBrowserAction({ id: "d1", sessionId: "s", op: "drag", x: 10, y: 10, toX: 200, toY: 10, direction: "down", policy }, run);
  assert.equal(posted.at(-1)?.result.ok, false);
  assert.match(posted.at(-1)?.result.error ?? "", /page changed during the drag/);
  assert.ok(sent.includes("pointer:down"));
  assert.ok(!sent.includes("pointer:up"), `released on the new page: ${sent.join(", ")}`);
  assert.equal(useBrowserStore.getState().tabs.length > 0, true);
});

test("a drag between fractional points reaches its end and is released", async () => {
  const posted: { result: { ok: boolean; error?: string } }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  const sent: Record<string, unknown>[] = [];
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(op);
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info, hit: "r5", sig: "s", doc: "r1" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  // On the tab the test above opened.
  const run = { runKey: "r", userText: "open https://example.com/board" };
  // 381.1 + (1.9 - 381.1) * 19 / 19 is 1.8999999999999773 in floating point.
  await runBrowserAction({ id: "f1", sessionId: "s", op: "drag", x: 381.1, y: 10, toX: 1.9, toY: 10, direction: "down", policy }, run);
  assert.equal(posted.at(-1)?.result.error, undefined);
  const up = sent.find((op) => op.op === "pointer" && op.phase === "up");
  assert.deepEqual(up && [up.x, up.y], [1.9, 10]);
  // The document check rides on the describes the drag makes anyway: no status round trips before the drop.
  assert.equal(sent.slice(0, sent.indexOf(up!)).filter((op) => op.op === "status").length, 0);
});

/** Drags on the tab the first test opened, with `onOp` seeing each page command; returns what was sent and the result. */
async function drag(id: string, runKey: string, onOp: (op: Record<string, unknown>, sent: string[]) => void) {
  const posted: { result: { ok: boolean; error?: string } }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  const sent: string[] = [];
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(`${op.op}${op.phase ? `:${op.phase}` : ""}`);
    onOp(op, sent);
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info, hit: "r5", sig: "s", doc: "r1" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const tabId = useBrowserStore.getState().tabs[0].id;
  useBrowserStore.getState().updateTab(tabId, { displayUrl: null });
  await runBrowserAction(
    { id, sessionId: "s", op: "drag", x: 10, y: 10, toX: 200, toY: 10, direction: "down", policy },
    { runKey, userText: "open https://example.com/board" },
  );
  return { sent, result: posted.at(-1)?.result };
}

test("a drag whose page moves its address without a new document is cancelled, not released", async () => {
  const { sent, result } = await drag("p1", "r", (op) => {
    if (op.op === "pointer" && op.phase === "down") {
      const tabId = useBrowserStore.getState().tabs[0].id;
      useBrowserStore.getState().updateTab(tabId, { displayUrl: "https://example.com/board#moved" });
    }
  });
  assert.equal(result?.ok, false);
  assert.ok(sent.includes("pointer:cancel"), sent.join(", "));
  assert.ok(!sent.includes("pointer:up"));
});

test("Stop during the checks before the drop cancels the drag instead of releasing it", async () => {
  const { sent, result } = await drag("p2", "r", (op, sent) => {
    if (op.op === "describe" && sent.includes("pointer:move")) stopBrowserAgent();
  });
  assert.equal(result?.ok, false);
  assert.ok(sent.includes("pointer:cancel"), sent.join(", "));
  assert.ok(!sent.includes("pointer:up"));
});
