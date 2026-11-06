// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const globals = globalThis as Record<string, unknown>;
const storage = new Map<string, string>();
globals.localStorage = { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => storage.set(key, value), removeItem: (key: string) => storage.delete(key) };
globals.window = Object.assign(new EventTarget(), { __TAURI_INTERNALS__: {}, localStorage: globals.localStorage, location: { protocol: "tauri:", href: "tauri://localhost" } });
register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const { patchAgentChat, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");
const { isTauri } = await import("../src/lib/api-base.ts");

test("background screenshots without a native page cannot capture the foreground chat", async () => {
  assert.equal(isTauri, true);
  useBrowserAgentStore.setState({ showCursor: false });
  const a = "single:shot-a";
  const b = "single:shot-b";
  const store = useBrowserStore.getState();
  store.showChat(a);
  const tabId = store.openTabIn(a, "https://a.example/")!;
  store.updateTab(tabId, { loading: false });
  patchAgentChat(a, { tabId, shared: true, enabled: true, runKey: null, stoppedRun: null });
  store.showChat(b);
  const foreground = { chat: b };
  globals.document = { querySelectorAll: () => [foreground] };
  globals.__frameAgent = async () => ({ installed: true, result: { ok: true, quietMs: 1000, requests: 0 } });
  globals.__callNative = globals.__frameAgent;
  const captures: unknown[][] = [];
  globals.__screenshotPage = async (...args: unknown[]) => { captures.push(args); return null; };
  globals.__captureHiddenPage = async (...args: unknown[]) => { captures.push(args); return null; };
  const posts: { result: { ok: boolean; error?: string } }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posts.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  let sequence = 0;
  const screenshot = () => runBrowserAction(
    { id: `shot-${++sequence}`, sessionId: a, op: "screenshot", policy: { allowedDomains: [], blockedDomains: [] } },
    { runKey: "shots", chat: a, fullAccess: true, userText: "Screenshot my page" },
  );
  globals.__native = false;
  await screenshot();
  assert.equal(captures.length, 0);
  assert.equal(posts.at(-1)?.result.ok, false);
  assert.match(posts.at(-1)?.result.error ?? "", /this chat's browser panel is not on screen; use browser_snapshot/);

  globals.__native = true;
  await screenshot();
  assert.equal(captures.length, 1);
  assert.equal((captures[0][0] as { id: string }).id, tabId);
  assert.equal(captures[0].length, 1);

  globals.__native = false;
  store.showChat(a);
  foreground.chat = a;
  await screenshot();
  assert.equal(captures.length, 2);
  assert.equal(captures[1][1], foreground);
});
