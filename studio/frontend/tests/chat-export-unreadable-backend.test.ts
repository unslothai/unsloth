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
      "@/lib/sse-framing": loadWithStubs(new URL("../src/lib/sse-framing.ts", import.meta.url), {}),
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

test("a chat backup longer than the browser's longest string still downloads", async () => {
  const videoChars = 300 * 1024 * 1024;
  const block = new TextEncoder().encode("A".repeat(1024 * 1024));
  const encoder = new TextEncoder();
  async function* body() {
    yield encoder.encode(
      '{"exportedAt":"2026-11-24T00:00:00Z","version":1,"threadCount":1,"projects":[],"threads":[{"id":"t1"}],"messages":[',
    );
    for (const id of ["m1", "m2"]) {
      yield encoder.encode(
        `${id === "m1" ? "\n" : ",\n"}{"id":"${id}","threadId":"t1","role":"user","createdAt":1,"content":[{"type":"video","data":"`,
      );
      for (let sent = 0; sent < videoChars; sent += block.length) yield block;
      yield encoder.encode('"}]}');
    }
    yield encoder.encode("\n]}");
  }
  const chunks = body();
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await chunks.next();
      if (done) controller.close();
      else controller.enqueue(value);
    },
  });
  const api = loadChatApi(new Response(stream));
  const backup = (await api.buildBackendChatExport()) as {
    threads: unknown[];
    messages: { id: string; content: { data: string }[] }[];
  };
  assert.deepEqual(
    backup.messages.map((message) => [message.id, message.content[0].data.length]),
    [
      ["m1", videoChars],
      ["m2", videoChars],
    ],
  );

  let saved: unknown;
  const download = loadWithStubs<{ downloadChatExport: () => Promise<void> }>(
    new URL("../src/features/chat/utils/export-chat-history.ts", import.meta.url),
    {
      "@/lib/native-files": {
        downloadBlobStreaming: async (content: unknown) => {
          saved = content;
        },
      },
      "./archived-chat-export": { filterArchivedChatExport: () => undefined },
      "./chat-history-storage": { buildStoredChatExport: async () => backup },
    },
  );
  await download.downloadChatExport();
  assert.ok(saved instanceof Blob);
  assert.ok(saved.size > 2 * videoChars);
});

test("a chat backup writes strings longer than one slice as the same JSON", async () => {
  const { chatExportBlob } = loadWithStubs<{
    chatExportBlob: (data: unknown) => Blob;
  }>(new URL("../src/features/chat/utils/export-chat-history.ts", import.meta.url), {
    "@/lib/native-files": { downloadBlobStreaming: async () => undefined },
    "./archived-chat-export": { filterArchivedChatExport: () => undefined },
    "./chat-history-storage": { buildStoredChatExport: async () => undefined },
  });
  const slice = 1 << 24;
  const data = "x".repeat(slice - 1) + '😀"\\\n' + "y".repeat(slice);
  const backup = {
    exportedAt: "x",
    version: 1,
    threadCount: 1,
    projects: [],
    threads: [{ id: "t1" }],
    messages: [
      { id: "m1", content: [{ type: "video", data }], skip: undefined },
      { id: "m2", content: "short" },
    ],
  };
  const text = await chatExportBlob(backup).text();
  assert.deepEqual(JSON.parse(text), JSON.parse(JSON.stringify(backup)));
});
