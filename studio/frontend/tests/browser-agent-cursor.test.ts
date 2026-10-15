// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");

const { useBrowserStore } = await import("../src/features/browser/store.ts");
// No page loads in node: a tab sent somewhere finishes loading on the next tick, as a quick page would.
const realNavigate = useBrowserStore.getState().navigate;
useBrowserStore.setState({
  navigate: (...args: Parameters<typeof realNavigate>) => {
    const result = realNavigate(...args);
    setTimeout(() => useBrowserStore.getState().updateTab(args[0], { loading: false }), 0);
    return result;
  },
});

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };

const stills: unknown[] = [];

async function cursorFlags(url: string): Promise<unknown[]> {
  globals.__authFetch = async () => new Response(null, { status: 200 });
  const flags: unknown[] = [];
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    flags.push(op.cursor);
    stills.push(op.cursorStill);
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, requests: 0, doc: "r1" } };
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


/** From the page's answer to the click until the tab is told to navigate. */
async function linkClickHold(showCursor: boolean, from: string): Promise<number> {
  useBrowserAgentStore.getState().setShowCursor(showCursor);
  globals.__authFetch = async () => new Response(null, { status: 200 });
  const link = { role: "link", name: "Docs", tag: "a", href: `${from}docs`, formMethod: "", formAction: "", formHasPassword: false, submits: false };
  let clickedAt = 0;
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000, requests: 0, doc: "r1" } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info: link, hit: "r5", sig: "s" } };
    if (op.op === "click") {
      clickedAt = Date.now();
      return { installed: true, result: { ok: true, navigate: `${from}docs` } };
    }
    return { installed: true, result: { ok: true, text: "page", links: [] } };
  };
  const run = { runKey: "r", userText: `open ${from} and click Docs` };
  await runBrowserAction({ id: `${from}open`, sessionId: "s", op: "navigate", url: from, direction: "down", policy }, run);
  // Nothing loads here, so the click's wait for the new page is cut off once the tab is sent on.
  const stop = new AbortController();
  const original = useBrowserStore.getState().navigate;
  let navigatedAt = 0;
  useBrowserStore.setState({
    navigate: (...args: Parameters<typeof original>) => {
      if (!navigatedAt && clickedAt) {
        navigatedAt = Date.now();
        stop.abort();
      }
      return original(...args);
    },
  });
  try {
    await runBrowserAction({ id: `${from}click`, sessionId: "s", op: "click", ref: "r5", direction: "down", policy }, { ...run, signal: stop.signal });
  } finally {
    useBrowserStore.setState({ navigate: original });
  }
  assert.ok(clickedAt && navigatedAt, "the link click never sent the tab on");
  return navigatedAt - clickedAt;
}

test("a link click holds the navigation briefly so the cursor's press shows, only with the cursor on", async () => {
  try {
    assert.ok((await linkClickHold(true, "https://example.com/c/")) >= 300);
    assert.ok((await linkClickHold(false, "https://example.com/d/")) < 300);
  } finally {
    useBrowserAgentStore.getState().setShowCursor(true);
  }
});
