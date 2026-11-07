// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction, STOPPED } = await import("../src/features/browser/agent/executor.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const { agentChat, patchAgentChat, setBrowserAgentEnabled } = await import("../src/features/browser/agent/agent-store.ts");

test("revoking browser use during run preparation invalidates its captured target", async () => {
  const source = readFileSync(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url), "utf8");
  const selection = source.indexOf("const browserTools = browserAgentToolsForTurn(runBrowserChat)");
  const version = source.indexOf("const browserTargetVersion", selection);
  assert.ok(selection > 0 && version > selection);
  const preparation = source.indexOf("async *run(");
  const hydration = source.indexOf("await useChatRuntimeStore.getState().hydratePersistedSettings()", preparation);
  assert.ok(version < hydration, "the target must be captured before settings hydration can yield");
  assert.doesNotMatch(source.slice(selection, version), /\bawait\b/);
  const capture = new Function("runBrowserChat", "browserAgentToolsForTurn", "agentChat", `${source.slice(selection, source.indexOf(";", version) + 1)}; return { browserTools, browserTargetVersion };`);
  const store = useBrowserStore.getState();
  const chat = store.chat;
  const tab = store.openTabIn(chat, "https://example.com/form")!;
  patchAgentChat(chat, { tabId: tab, shared: true });
  setBrowserAgentEnabled(chat, true);
  let release!: () => void;
  const loading = new Promise<void>((resolve) => { release = resolve; });
  const posts: Array<{ result: { ok: boolean; error?: string } }> = [];
  let pageCalls = 0;
  Object.assign(globalThis, {
    __frameAgent: async () => { pageCalls++; return { installed: true, result: { ok: true, text: "private contents" } }; },
    __authFetch: async (_url: string, init: { body: string }) => { posts.push(JSON.parse(init.body)); return new Response(null, { status: 200 }); },
  });
  const prepared = (async () => {
    const captured = capture(chat, () => ["browser_snapshot"], agentChat);
    await loading;
    await runBrowserAction({ id: "prepared", sessionId: "s", op: "snapshot", policy: { allowedDomains: [], blockedDomains: [] } }, { runKey: "prepared", chat, targetVersion: captured.browserTargetVersion, userText: "example.com" });
  })();
  setBrowserAgentEnabled(chat, false);
  release();
  await prepared;
  assert.equal(posts[0].result.error, STOPPED);
  assert.equal(posts[0].result.ok, false);
  assert.equal(pageCalls, 0);
});
