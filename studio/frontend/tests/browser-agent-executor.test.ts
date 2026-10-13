// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");

const globals = globalThis as Record<string, unknown>;

type Posted = { result: { ok: boolean; url?: string; text?: string; error?: string } };

/** Runs one navigate with the page answering `status` through `onStatus`; returns the posted result. */
async function navigate(id: string, url: string, onStatus: (tabId: string, call: number) => number) {
  const posted: Posted[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  let calls = 0;
  globals.__frameAgent = async (tabId: string, op: Record<string, unknown>) => {
    if (op.op === "status") return { installed: true, result: { ok: true, pending: onStatus(tabId, ++calls), quietMs: 1000 } };
    return { installed: true, result: { ok: true, text: "Qwen models", links: [] } };
  };
  await runBrowserAction(
    { id, sessionId: "s", op: "navigate", url, direction: "down", policy: { allowedDomains: [], blockedDomains: [] } },
    { runKey: "r", userText: `open ${url}` },
  );
  assert.equal(posted.length, 1);
  return { result: posted[0].result, calls };
}

test("a router that rewrites the address while a new page fills in is read at its new address", async () => {
  // The list's request is still open; the router moves the address on, then the list arrives.
  const { result, calls } = await navigate("a1", "https://example.com/models", (tabId, call) => {
    if (call === 2) useBrowserStore.getState().updateTab(tabId, { displayUrl: "https://example.com/models?search=qwen" });
    return call < 3 ? 1 : 0;
  });
  assert.equal(result.error, undefined);
  assert.equal(result.ok, true);
  assert.equal(result.url, "https://example.com/models?search=qwen");
  assert.equal(result.text, "Qwen models");
  assert.ok(calls >= 3, "the wait stopped at the address change instead of waiting for the list");
});

test("a page that navigates away while it fills in is not read as the new page", async () => {
  const { result } = await navigate("a2", "https://example.com/docs", (tabId, call) => {
    if (call === 2) useBrowserStore.getState().navigate(tabId, { url: "https://other.example.org/" }, { replace: false });
    return 1;
  });
  assert.equal(result.ok, false);
  assert.equal(result.text, undefined);
});
