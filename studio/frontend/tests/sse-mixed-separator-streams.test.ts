// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// SSE frame separators can mix line endings: a blank line after \r\n field
// lines is a three-byte boundary ("\r\n\n", or "\n\r\n" for \n fields before a
// \r\n blank), not the two or four bytes the old drain loops assumed.

const openStreamResponse = loadWithStubs<
  typeof import("../src/lib/open-stream-response.ts")
>(new URL("../src/lib/open-stream-response.ts", import.meta.url), {});

const sseFraming = loadWithStubs<typeof import("../src/lib/sse-framing.ts")>(
  new URL("../src/lib/sse-framing.ts", import.meta.url),
  {},
);

function sseResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

type ExportApi = typeof import("../src/features/export/api/export-api.ts");

function loadExportApi(response: Response): ExportApi {
  return loadWithStubs<ExportApi>(
    new URL("../src/features/export/api/export-api.ts", import.meta.url),
    {
      "@/features/auth": { authFetch: async () => response },
      "@/lib/format-fastapi-error": { readFastApiError: async () => "error" },
      "@/lib/open-stream-response": openStreamResponse,
      "@/lib/sse-framing": sseFraming,
    },
  );
}

type RecipeApi = typeof import("../src/features/recipe-studio/api/index.ts");

function loadRecipeApi(response: Response): RecipeApi {
  return loadWithStubs<RecipeApi>(
    new URL("../src/features/recipe-studio/api/index.ts", import.meta.url),
    {
      "@/features/auth": { authFetch: async () => response },
      "@/lib/api-base": { apiUrl: (path: string) => path },
      "@/lib/format-fastapi-error": {
        formatFastApiDetail: () => "error",
        readFastApiError: async () => "error",
      },
      "@/lib/open-stream-response": openStreamResponse,
      "@/lib/sse-framing": sseFraming,
    },
  );
}

type ChatApi = {
  streamChatCompletions: (
    payload: unknown,
    signal: AbortSignal,
    loadedContextLength?: number | null,
  ) => AsyncGenerator<unknown>;
};

function loadChatApi(response: Response): ChatApi {
  return loadWithStubs<ChatApi>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "./skill-load-event": { skillLoadCardEvent },
      "./thread-read-cache.ts": loadWithStubs(
        new URL(
          "../src/features/chat/api/thread-read-cache.ts",
          import.meta.url,
        ),
        {},
      ),
      "./generation-length.ts": loadWithStubs(
        new URL(
          "../src/features/chat/api/generation-length.ts",
          import.meta.url,
        ),
        {},
      ),
      "@/lib/sse-framing": sseFraming,
      "@/features/auth": {
        authFetch: async () => response,
        getAuthSessionEpoch: () => 0,
      },
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
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => undefined },
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": {
        disposableTimeoutSignal: () => ({
          signal: new AbortController().signal,
          dispose: () => {},
        }),
      },
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => {} },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": {
        consumeNativePathToken: () => undefined,
      },
      "@/features/settings/low-disk-check": {
        checkDiskSpace: () => Promise.resolve(),
      },
      "@/lib/model-lifecycle-events": {},
    },
  );
}

test("export log stream keeps events split on a mixed line-ending boundary", async () => {
  const body =
    'event: log\r\ndata: {"line":"a","stream":"stdout"}\r\n\n' +
    'event: log\ndata: {"line":"b","stream":"stderr"}\n\r\n' +
    "event: complete\ndata: {}\n\n";
  const api = loadExportApi(sseResponse(body));

  const events: Array<{ event: string }> = [];
  await api.streamExportLogs({
    signal: new AbortController().signal,
    onEvent: (event) => events.push(event),
  });
  assert.deepEqual(
    events.map((event) => event.event),
    ["log", "log", "complete"],
  );
});

test("recipe job stream keeps the event name across a mixed line-ending boundary", async () => {
  const body =
    'event: progress\r\ndata: {"n":1}\r\n\n' +
    'event: progress\ndata: {"n":2}\n\r\n' +
    'event: done\ndata: {"ok":true}\n\n';
  const api = loadRecipeApi(sseResponse(body));

  const events: Array<{ event: string }> = [];
  await api.streamRecipeJobEvents({
    jobId: "job-1",
    signal: new AbortController().signal,
    onEvent: (event) => events.push(event),
  });
  assert.deepEqual(
    events.map((event) => event.event),
    ["progress", "progress", "done"],
  );
});

test("chat completions stream reaches [DONE] across a mixed line-ending boundary", async () => {
  const body =
    'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"}}]}\r\n\n' +
    'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"yo"}}]}\n\r\n' +
    "data: [DONE]\n\n";
  const api = loadChatApi(sseResponse(body));

  const chunks: unknown[] = [];
  for await (const chunk of api.streamChatCompletions(
    { model: "m", messages: [] },
    new AbortController().signal,
  )) {
    chunks.push(chunk);
  }
  assert.equal(chunks.length, 2);
  const choice = (
    chunks[1] as { choices: Array<{ delta: { content: string } }> }
  ).choices[0];
  assert.equal(choice?.delta.content, "yo");
});
