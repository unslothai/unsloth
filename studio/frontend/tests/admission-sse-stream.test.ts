// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The admission comments, read off a real SSE byte stream.
 *
 * `admission-status.test.ts` covers the vocabulary; this covers the seam that consumes it,
 * which is where the interesting mistakes live. Two in particular:
 *
 *   * an admission block carries NO `data:` line, and the reader's fast path skips any
 *     block that produced none. Testing the parser alone would never catch the signal
 *     being dropped one line later.
 *   * comments can arrive split across reader chunks, because SSE framing is a byte
 *     stream and nothing guarantees a comment lands whole in one read.
 */

import assert from "node:assert/strict";
import test from "node:test";

import * as admissionStatus from "../src/features/chat/utils/admission-status.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Module = {
  streamChatCompletions: (
    payload: unknown,
    signal: AbortSignal,
    loadedContextLength?: number | null,
  ) => AsyncGenerator<Record<string, unknown>>;
};

/** A Response double whose body yields exactly the byte slices it is given. */
function sseResponse(slices: string[]) {
  const encoder = new TextEncoder();
  let i = 0;
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: async () =>
          i < slices.length
            ? { done: false, value: encoder.encode(slices[i++]) }
            : { done: true, value: undefined },
        cancel: async () => {},
      }),
    },
  };
}

function harness(slices: string[]) {
  return loadWithStubs<Module>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "@/features/auth": { authFetch: async () => sseResponse(slices) },
      "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
      "../types": {},
      "../types/api": {},
      "../utils/admission-status": admissionStatus,
      "../utils/chat-history-revision": {
        notifyChatHistoryUpdated: () => {},
        isCoalescedHistoryEvent: () => false,
      },
      "./generation-length.ts": { maxTokensIsTheLimit: () => "" },
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => undefined },
      "@/features/hub/lib/abort-signals": {},
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": {
        consumeNativePathToken: () => undefined,
      },
      "@/lib/model-lifecycle-events": {},
    },
  );
}

async function collect(slices: string[]) {
  const module = harness(slices);
  const chunks: Record<string, unknown>[] = [];
  for await (const chunk of module.streamChatCompletions(
    { model: "default", messages: [] },
    new AbortController().signal,
  )) {
    chunks.push(chunk);
  }
  return chunks;
}

const CONTENT =
  'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n';
const DONE = "data: [DONE]\n\n";

test("a queued run reports waiting, then admitted, then streams", async () => {
  const chunks = await collect([
    ": admission-wait\n\n",
    ": admission-done\n\n",
    CONTENT,
    DONE,
  ]);
  const statuses = chunks
    .map((c) => c._admissionStatus)
    .filter((s) => s !== undefined);
  assert.deepEqual(statuses, ["waiting", "admitted"]);
});

test("an admission block is not swallowed by the empty-block fast path", async () => {
  // The regression this exists for: the reader skips blocks with no `data:` line, and an
  // admission block is exactly that. Handling it after the skip drops every one.
  const chunks = await collect([": admission-wait\n\n", CONTENT, DONE]);
  assert.equal(
    chunks.filter((c) => c._admissionStatus === "waiting").length,
    1,
  );
});

test("the content still arrives alongside the signals", async () => {
  // The signals must be additive: teaching the reader about comments must not cost a token.
  const chunks = await collect([": admission-wait\n\n", CONTENT, DONE]);
  const text = chunks
    .flatMap((c) => (c.choices as { delta?: { content?: string } }[]) ?? [])
    .map((ch) => ch.delta?.content ?? "")
    .join("");
  assert.equal(text, "hi");
});

test("a comment split across reads is still recognised", async () => {
  // SSE is a byte stream; nothing promises a comment lands whole in one read.
  const chunks = await collect([": admiss", "ion-wait\n", "\n", CONTENT, DONE]);
  assert.equal(
    chunks.filter((c) => c._admissionStatus === "waiting").length,
    1,
  );
});

test("keep-alive comments produce no admission chunk", async () => {
  const chunks = await collect([": keep-alive\n\n", CONTENT, DONE]);
  assert.equal(
    chunks.filter((c) => c._admissionStatus !== undefined).length,
    0,
  );
});

test("a paused run reports paused and then resumed", async () => {
  const chunks = await collect([
    ": admission-done\n\n",
    CONTENT,
    ": preempt-paused\n\n",
    ": preempt-resumed\n\n",
    CONTENT,
    DONE,
  ]);
  const statuses = chunks
    .map((c) => c._admissionStatus)
    .filter((s) => s !== undefined);
  assert.deepEqual(statuses, ["admitted", "paused", "resumed"]);
});

test("repeated waits while queued are each reported", async () => {
  // The backend re-emits the wait comment on an interval as its own keep-alive, so a
  // reader that reported only the first would let the indicator go stale.
  const chunks = await collect([
    ": admission-wait\n\n",
    ": admission-wait\n\n",
    ": admission-done\n\n",
    CONTENT,
    DONE,
  ]);
  assert.deepEqual(
    chunks.map((c) => c._admissionStatus).filter((s) => s !== undefined),
    ["waiting", "waiting", "admitted"],
  );
});

test("CRLF framing is handled like LF", async () => {
  // Some proxies rewrite line endings on the way through.
  const chunks = await collect([": admission-wait\r\n\r\n", CONTENT, DONE]);
  assert.equal(
    chunks.filter((c) => c._admissionStatus === "waiting").length,
    1,
  );
});
