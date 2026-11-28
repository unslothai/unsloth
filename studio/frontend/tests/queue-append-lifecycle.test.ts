// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The installed public ThreadRuntimeImpl.append() returns void and the local
// core does not await history.append(). These tests pin that contract and the
// metadata.custom string token that carries our own settlement signal.

import assert from "node:assert/strict";
import test from "node:test";
import {
  LocalThreadRuntimeCore,
  ThreadRuntimeImpl,
} from "@assistant-ui/core/internal";
import {
  QUEUE_APPEND_TOKEN_KEY,
  QueueAppendAbandonedError,
  abandonQueueAppend,
  createQueueAppendToken,
  failQueueAppend,
  queueAppendTokenForMessage,
  settleQueueAppend,
  stripQueueAppendToken,
  takeQueueAppendToken,
  waitForQueueAppend,
} from "../src/features/chat/utils/queue-append-lifecycle.ts";
import { queuedPromptMessage } from "../src/features/chat/utils/queued-text-attachments.ts";

function runtimeWithHistory(
  onAppend: (item: {
    message: { metadata?: { custom?: Record<string, unknown> } };
  }) => void,
) {
  const core = new LocalThreadRuntimeCore(
    { getModelContext: () => ({}) },
    {
      adapters: {
        chatModel: {
          run: async () => ({
            content: [],
            status: { type: "complete", reason: "stop" },
          }),
        },
        history: {
          load: async () => null as never,
          append: async (item: {
            message: { metadata?: { custom?: Record<string, unknown> } };
          }) => {
            onAppend(item);
          },
        },
      },
    },
  );
  // ThreadRuntimeImpl eagerly creates the thread composer. Its append path is
  // the contract under test; the unused composer only needs a subject shape.
  (core as unknown as { composer: unknown }).composer = {
    path: { ref: "thread.composer" },
    getState: () => ({}),
    subscribe: () => () => undefined,
  };
  const item = {
    id: "local-1",
    remoteId: "thread-1",
    externalId: undefined,
    title: undefined,
    status: "regular" as const,
    isMain: true,
  };
  const binding = {
    path: { ref: "thread" },
    getState: () => core,
    getStateState: () => core,
    outerSubscribe: (callback: () => void) => core.subscribe(callback),
    subscribe: (callback: () => void) => core.subscribe(callback),
  };
  const listBinding = {
    path: { ref: "threadListItem" },
    getState: () => item,
    subscribe: () => () => undefined,
  };
  return {
    core,
    runtime: new ThreadRuntimeImpl(binding as never, listBinding as never),
  };
}

test("the installed public append() returns void and the token reaches history", async () => {
  const appended: Array<{
    message: { metadata?: { custom?: Record<string, unknown> } };
  }> = [];
  const { runtime } = runtimeWithHistory((item) => {
    appended.push(item);
  });
  const token = createQueueAppendToken();
  const completion = waitForQueueAppend(token);
  const message = queuedPromptMessage({ prompt: "queued" }, token);

  const result = runtime.append({ ...message, startRun: false });
  assert.equal(result, undefined);
  assert.equal(appended.length, 1);
  const appendedItem = appended[0];
  assert.ok(appendedItem);
  const custom = appendedItem.message.metadata?.custom;
  assert.equal(custom?.[QUEUE_APPEND_TOKEN_KEY], token);
  assert.equal(queueAppendTokenForMessage(appendedItem.message), token);

  const persisted = stripQueueAppendToken(custom);
  assert.equal(persisted?.[QUEUE_APPEND_TOKEN_KEY], undefined);
  assert.equal(
    persisted,
    undefined,
    "a token-only custom object leaves no metadata",
  );

  const originalCustom = message.metadata?.custom;
  assert.equal(takeQueueAppendToken(message), token);
  assert.equal(originalCustom?.[QUEUE_APPEND_TOKEN_KEY], undefined);

  abandonQueueAppend(token);
  await assert.rejects(completion, QueueAppendAbandonedError);
});

test("history success settles the waiter", async () => {
  const { runtime } = runtimeWithHistory(() => undefined);
  const token = createQueueAppendToken();
  const completion = waitForQueueAppend(token);
  runtime.append({
    ...queuedPromptMessage({ prompt: "queued" }, token),
    startRun: false,
  });
  // The real adapter settles here; the test callback ran synchronously.
  settleQueueAppend(token);
  await completion;
});

test("history failure rejects the waiter", async () => {
  const { runtime } = runtimeWithHistory(() => undefined);
  const token = createQueueAppendToken();
  const completion = waitForQueueAppend(token);
  runtime.append({
    ...queuedPromptMessage({ prompt: "queued" }, token),
    startRun: false,
  });
  failQueueAppend(token, new Error("history failed"));
  await assert.rejects(completion, /history failed/);
});

test("ordinary queued messages have no metadata and no token", () => {
  const message = queuedPromptMessage({ prompt: "plain" });
  assert.equal(message.metadata, undefined);
  assert.equal(queueAppendTokenForMessage(message), null);
});

test("abandonment rejects with a typed error and is idempotent", async () => {
  const token = createQueueAppendToken();
  const completion = waitForQueueAppend(token);
  abandonQueueAppend(token);
  abandonQueueAppend(token);
  await assert.rejects(completion, QueueAppendAbandonedError);
});
