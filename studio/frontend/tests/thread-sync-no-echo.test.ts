// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { MessageRepository } from "@assistant-ui/core/internal";
import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import * as savedUsage from "../src/features/chat/utils/saved-context-usage.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Record_ = Record<string, unknown>;
type SyncOptions = { pruneMissing?: boolean; deletedMessageIds?: string[]; echo?: boolean };

type ApiModule = {
  syncChatMessages: (threadId: string, messages: Record_[], options?: SyncOptions) => Promise<Record_[]>;
};

function jsonResponse(status: number, body: unknown) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: () => null },
    async json() {
      return body;
    },
  };
}

function apiHarness(response: ReturnType<typeof jsonResponse>) {
  const requests: { url: string; init?: RequestInit }[] = [];
  const module = loadWithStubs<ApiModule>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "./skill-load-event": { skillLoadCardEvent },
      "./thread-read-cache.ts": loadWithStubs(
        new URL("../src/features/chat/api/thread-read-cache.ts", import.meta.url),
        {},
      ),
      "@/lib/sse-framing": loadWithStubs(new URL("../src/lib/sse-framing.ts", import.meta.url), {}),
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          requests.push({ url, init });
          return response;
        },
      },
      "@/lib/format-fastapi-error": {
        formatApiErrorBody: (body: unknown) =>
          (body as { detail?: string } | null)?.detail ?? null,
      },
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
  return { module, requests };
}

const sent = [
  { id: "m1", threadId: "t1", parentId: null, role: "user", content: [], createdAt: 1 },
  { id: "m2", threadId: "t1", parentId: "m1", role: "assistant", content: [], createdAt: 2 },
];
const stored = [{ ...sent[0], content: [{ type: "text", text: "server copy" }] }];

test("echo: false asks the backend to skip the echo and resolves to the sent records", async () => {
  const { module, requests } = apiHarness(jsonResponse(200, { messages: [] }));
  const result = await module.syncChatMessages("t1", sent, { echo: false });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/api/chat/threads/t1/messages?echoMessages=false");
  assert.equal(requests[0].init?.method, "PUT");
  assert.deepEqual(result, sent);
});

test("an older backend that ignores the flag and echoes every row is still accepted", async () => {
  const { module } = apiHarness(jsonResponse(200, { messages: stored }));
  const result = await module.syncChatMessages("t1", sent, { echo: false });
  assert.deepEqual(result, sent);
});

test("the default sync and every delete keep the echo", async () => {
  for (const options of [
    {},
    { echo: false, pruneMissing: true },
    { echo: false, deletedMessageIds: ["m2"] },
  ] as SyncOptions[]) {
    const { module, requests } = apiHarness(jsonResponse(200, { messages: stored }));
    const result = await module.syncChatMessages("t1", sent, options);
    assert.equal(requests[0].url, "/api/chat/threads/t1/messages", JSON.stringify(options));
    assert.deepEqual(result, stored);
  }
});

test("a failed no-echo sync still throws", async () => {
  const { module } = apiHarness(jsonResponse(500, { detail: "boom" }));
  await assert.rejects(module.syncChatMessages("t1", sent, { echo: false }));
});

type Exported = {
  headId: string | null;
  messages: { parentId: string | null; message: Record_ }[];
};

type ExportModule = {
  syncExportedRepositoryToBackend: (
    remoteId: string,
    exp: Exported,
    options?: { pruneMissing?: boolean; deletedMessageIds?: string[] },
  ) => Promise<void>;
  deleteThreadMessage: (args: {
    thread: { export: () => Exported; import: (data: Exported) => void };
    messageId: string;
    remoteId: string | undefined;
  }) => Promise<void>;
};

function exportHarness() {
  const options: SyncOptions[] = [];
  const module = loadWithStubs<ExportModule>(
    new URL("../src/features/chat/utils/delete-thread-message.ts", import.meta.url),
    {
      "@assistant-ui/core/internal": { MessageRepository },
      "../api/chat-api": { listChatMessages: async () => [] },
      "./chat-history-storage": {
        ensureStoredChatThread: async () => {},
        saveStoredChatMessage: async (record: Record_) => record,
        syncStoredChatMessages: async (_id: string, records: Record_[], opts: SyncOptions) => {
          options.push(opts);
          return records;
        },
      },
      "./research-message-sync": researchSync,
      "./saved-context-usage": savedUsage,
    },
  );
  return { module, options };
}

function chat(): Exported {
  return {
    headId: "a1",
    messages: [
      {
        parentId: null,
        message: { id: "u1", role: "user", content: [{ type: "text", text: "hi" }], createdAt: new Date(1000) },
      },
      {
        parentId: "u1",
        message: {
          id: "a1",
          role: "assistant",
          content: [{ type: "text", text: "hello" }],
          status: { type: "complete", reason: "stop" },
          metadata: {},
          createdAt: new Date(2000),
        },
      },
    ],
  };
}

test("the autosave sends echo: false and the delete path keeps the echo", async () => {
  const { module, options } = exportHarness();
  await module.syncExportedRepositoryToBackend("t1", chat());
  assert.equal(options.at(-1)?.echo, false);

  await module.syncExportedRepositoryToBackend("t1", chat(), { pruneMissing: true });
  assert.equal(options.at(-1)?.echo, true);

  const exported = chat();
  await module.deleteThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a1",
    remoteId: "t1",
  });
  assert.equal(options.at(-1)?.pruneMissing, true);
  assert.equal(options.at(-1)?.echo, true, "publishForkBoundary reads the echoed rows");
});
