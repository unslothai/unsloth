// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type RefreshModule = {
  refreshContextUsage: (options?: { invalidate?: boolean }) => Promise<void>;
};

test("a prompt setting change clears local usage even during a run", async () => {
  const cleared: unknown[] = [];
  const state = {
    activeThreadId: "thread-1",
    contextUsage: { totalTokens: 42 },
    loadedContextLength: 4096,
    modelLoading: false,
    models: [],
    params: { checkpoint: "local-model" },
    runningByThreadId: { "thread-1": true },
    setContextUsage: (usage: unknown) => cleared.push(usage),
  };
  const refreshModule = loadWithStubs<RefreshModule>(
    new URL(
      "../src/features/chat/utils/refresh-context-usage.ts",
      import.meta.url,
    ),
    {
      "../api/chat-adapter": {},
      "../api/chat-api": {},
      "../external-providers": {
        isExternalModelId: () => false,
      },
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => state },
      },
      "./chat-history-storage": {},
    },
  );

  await refreshModule.refreshContextUsage({ invalidate: true });

  assert.deepEqual(cleared, [null]);
});

test("a prompt setting change leaves external usage intact", async () => {
  const cleared: unknown[] = [];
  const state = {
    activeThreadId: "thread-1",
    contextUsage: { totalTokens: 42 },
    loadedContextLength: 4096,
    modelLoading: false,
    models: [],
    params: { checkpoint: "external:model" },
    runningByThreadId: {},
    setContextUsage: (usage: unknown) => cleared.push(usage),
  };
  const refreshModule = loadWithStubs<RefreshModule>(
    new URL(
      "../src/features/chat/utils/refresh-context-usage.ts",
      import.meta.url,
    ),
    {
      "../api/chat-adapter": {},
      "../api/chat-api": {},
      "../external-providers": {
        isExternalModelId: () => true,
      },
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => state },
      },
      "./chat-history-storage": {},
    },
  );

  await refreshModule.refreshContextUsage({ invalidate: true });

  assert.deepEqual(cleared, []);
});
