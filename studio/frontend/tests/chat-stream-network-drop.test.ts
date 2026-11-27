// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

let nextResponse: () => Response = () => new Response(null, { status: 500 });

const { streamChatCompletions, StreamInterruptedError: Interrupted } =
  loadWithStubs<{
    streamChatCompletions: (
      payload: unknown,
      signal: AbortSignal,
    ) => AsyncGenerator<unknown>;
    StreamInterruptedError: new () => Error;
  }>(new URL("../src/features/chat/api/chat-api.ts", import.meta.url), {
    "./skill-load-event": { skillLoadCardEvent },
    "./thread-read-cache.ts": loadWithStubs(
      new URL("../src/features/chat/api/thread-read-cache.ts", import.meta.url),
      {},
    ),
    "@/features/auth": { authFetch: async () => nextResponse() },
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
    "./generation-length.ts": {
      lengthStopCause: () => "unknown",
      windowEvidenceCount: () => null,
    },
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
    "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
    "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
    "@/features/native-intents/api": {
      consumeNativePathToken: () => undefined,
    },
    "@/features/settings/low-disk-check": {
      checkDiskSpace: () => Promise.resolve(),
    },
    "@/lib/model-lifecycle-events": {},
  });

function droppingStream(error: unknown): ReadableStream<Uint8Array> {
  let sent = false;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (!sent) {
        sent = true;
        controller.enqueue(
          new TextEncoder().encode(
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
          ),
        );
        return;
      }
      controller.error(error);
    },
  });
}

async function drain(signal: AbortSignal) {
  const chunks: unknown[] = [];
  for await (const chunk of streamChatCompletions(
    { model: "m", messages: [], stream: true } as never,
    signal,
  )) {
    chunks.push(chunk);
  }
  return chunks;
}

test("a socket dropped mid-stream reads as an interruption, not a failed generation", async () => {
  nextResponse = () =>
    new Response(droppingStream(new TypeError("Load failed")), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  await assert.rejects(drain(new AbortController().signal), Interrupted);
});

test("other stream errors and a Stop keep their own error", async () => {
  nextResponse = () =>
    new Response(droppingStream(new RangeError("decoder")), {
      status: 200,
    });
  await assert.rejects(drain(new AbortController().signal), RangeError);

  const stop = new AbortController();
  stop.abort();
  nextResponse = () =>
    new Response(droppingStream(new TypeError("aborted")), {
      status: 200,
    });
  await assert.rejects(
    drain(stop.signal),
    (error: unknown) => !(error instanceof Interrupted),
  );
});
