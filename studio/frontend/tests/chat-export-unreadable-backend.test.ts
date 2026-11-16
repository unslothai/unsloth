// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { MessageRecord, ThreadRecord } from "../src/features/chat/types";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";

function loadChatApi(response: Response) {
  return loadWithStubs<{ buildBackendChatExport: () => Promise<unknown> }>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "./skill-load-event": { skillLoadCardEvent },
      "./thread-read-cache.ts": loadWithStubs(
        new URL("../src/features/chat/api/thread-read-cache.ts", import.meta.url),
        {},
      ),
      "@/features/auth": { authFetch: async () => response },
      "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
      "../types": {},
      "../types/api": {},
      "../utils/chat-history-revision": {
        notifyChatHistoryUpdated: () => {},
        isCoalescedHistoryEvent: () => false,
      },
      "../utils/openapi-support": { readAdvertisedPatchGuards: () => null },
      "../utils/load-warning-toast": { showLoadWarning: () => {} },
      "../lib/load-request": {},
      "./generation-length.ts": {},
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => undefined },
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": {
        disposableTimeoutSignal: () => ({ signal: new AbortController().signal, dispose: () => {} }),
      },
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": { consumeNativePathToken: () => undefined },
      "@/features/settings/low-disk-check": { checkDiskSpace: () => Promise.resolve() },
      "@/lib/model-lifecycle-events": {},
    },
  );
}

const legacyThread = { id: "old-legacy", title: "Old browser chat" } as ThreadRecord;
const legacyMessage = {
  id: "old-message",
  threadId: "old-legacy",
  role: "user",
  content: [{ type: "text", text: "hi" }],
  createdAt: 1,
} as MessageRecord;

function loadStorage(buildBackendChatExport: () => Promise<unknown>) {
  return loadWithStubs<{ buildStoredChatExport: () => Promise<{ threadCount: number }> }>(
    new URL("../src/features/chat/utils/chat-history-storage.ts", import.meta.url),
    {
      "../api/chat-api": {
        ChatThreadDeletedError: class extends Error {},
        buildBackendChatExport,
        notifyChatHistoryUpdated: () => {},
      },
      "../db": {
        DEXIE_DB_NAME: "unsloth-chat",
        db: {
          threads: { toArray: async () => [legacyThread] },
          messages: { toArray: async () => [legacyMessage] },
          transaction: async () => undefined,
        },
      },
      "./chat-thread-tombstones": {
        isChatThreadDeleted: () => false,
        markChatThreadDeleted: () => {},
        markChatThreadsDeleted: () => {},
      },
      "./thread-record-write-coordinator": {
        ThreadRecordWriteCoordinator: class {},
      },
      "../stores/fork-boundary-store": { setForkBoundary: () => {} },
    },
  );
}

test("a chat export whose body cannot be read fails instead of saving no chats", async () => {
  const api = loadChatApi(
    new Response('{"exportedAt":"2026-11-06T00:00:00Z","threads":[{"id":"t1"', {
      status: 200,
    }),
  );
  await assert.rejects(api.buildBackendChatExport());
});

test("a failed server export fails instead of saving only old browser chats", async () => {
  const storage = loadStorage(async () => {
    throw new Error("Request failed (500)");
  });
  await assert.rejects(storage.buildStoredChatExport(), /500/);
});
