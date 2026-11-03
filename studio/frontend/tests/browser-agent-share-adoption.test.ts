// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { useBrowserStore } = await import("../src/features/browser/store.ts");
const agent = await import("../src/features/browser/agent/agent-store.ts");
const { shareAgentTab } = await import("../src/features/browser/agent/tab-sharing.ts");

test("a shared draft tab follows the chat's first save without loading the executor", () => {
  const store = useBrowserStore.getState();
  const draft = store.chat;
  const tab = store.openTabIn(draft, "https://example.com/form")!;
  assert.equal(shareAgentTab(draft, tab), true);
  store.adoptChat(draft, "single:saved");
  assert.equal(agent.agentChat("single:saved").tabId, tab);
});
