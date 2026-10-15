// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };

const stills: unknown[] = [];

async function cursorFlags(url: string): Promise<unknown[]> {
  globals.__authFetch = async () => new Response(null, { status: 200 });
  const flags: unknown[] = [];
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    flags.push(op.cursor);
    stills.push(op.cursorStill);
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, doc: "r1" } };
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const run = { runKey: "r", userText: `open ${url}` };
  await runBrowserAction({ id: url, sessionId: "s", op: "navigate", url, direction: "down", policy }, run);
  return flags;
}

test("the cursor is on by default and every page command carries the setting", async () => {
  assert.equal(useBrowserAgentStore.getState().showCursor, true);
  const flags = await cursorFlags("https://example.com/a");
  assert.ok(flags.length > 0);
  assert.ok(flags.every((flag) => flag === true), flags.join(", "));
  // Reduce motion travels with the cursor; off here, where there is no document to read it from.
  assert.ok(stills.every((still) => still === false), stills.join(", "));
});

test("turning the cursor off in Settings reaches the page on the next command", async () => {
  useBrowserAgentStore.getState().setShowCursor(false);
  try {
    const flags = await cursorFlags("https://example.com/b");
    assert.ok(flags.length > 0);
    assert.ok(flags.every((flag) => flag === false), flags.join(", "));
  } finally {
    useBrowserAgentStore.getState().setShowCursor(true);
  }
});
