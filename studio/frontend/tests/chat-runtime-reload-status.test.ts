// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { readIncompleteInfo, restoredAssistantStatus } = await import(
  "../src/features/chat/utils/continuation.ts"
);
const {
  generationChunkHasSubstantiveDelta,
  generationIsSettled,
  generationNeedsRecovery,
  loadGenerationOverlaySnapshot,
  recoveredReasoningSummaryMetadata,
  recoveredGenerationFinalMetadata,
  generationRecoveryMetadata,
  shouldPreserveGenerationMetadata,
  subscribeGenerationRecoveryTriggers,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

test("first-token recovery ignores role and control chunks", () => {
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { role: "assistant" } }],
    }),
    false,
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [],
      usage: { completion_tokens: 1 },
    }),
    false,
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { content: "token" } }],
    }),
    true,
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { reasoning_content: "thought" } }],
    }),
    true,
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { reasoning_details: [{ text: "thought" }] } }],
    }),
    true,
  );
});

test("reload recovery preserves server reasoning durations", () => {
  const metadata = recoveredReasoningSummaryMetadata(
    {
      reasoningDuration: 1,
      reasoningDurations: [1],
    },
    3200,
  );
  assert.equal(metadata.reasoningDuration, 3);
  assert.deepEqual(metadata.reasoningDurations, [1, 3]);
  assert.equal(recoveredReasoningSummaryMetadata(metadata, -1), metadata);
});

test("stored assistant status remains truthful after reload", () => {
  for (const reason of ["interrupted", "length"] as const) {
    const metadata = { custom: { incomplete: { reason } } };
    assert.deepEqual(restoredAssistantStatus(metadata), {
      type: "incomplete",
      reason: "error",
    });
    assert.deepEqual(readIncompleteInfo(metadata), { reason });
  }

  assert.deepEqual(
    restoredAssistantStatus({
      custom: { incomplete: { reason: "cancelled" } },
    }),
    { type: "incomplete", reason: "cancelled" },
  );

  assert.deepEqual(restoredAssistantStatus({ custom: {} }), {
    type: "complete",
    reason: "unknown",
  });
  assert.deepEqual(restoredAssistantStatus(undefined), {
    type: "complete",
    reason: "unknown",
  });
});

test("terminal status settles only after the replay cursor catches up", () => {
  assert.equal(generationIsSettled("completed", 3, 5), false);
  assert.equal(generationIsSettled("completed", 5, 5), true);
  assert.equal(generationIsSettled("running", 5, 5), false);
});

test("active runs are read before messages so a concurrent create is visible", async () => {
  const calls: string[] = [];
  const stored = [{ id: "user-1" }];
  const snapshot = await loadGenerationOverlaySnapshot(
    "thread-1",
    () => {
      calls.push("runs");
      stored.push({ id: "assistant-1" });
      return Promise.resolve([{ assistantMessageId: "assistant-1" }]);
    },
    () => {
      calls.push("messages");
      return Promise.resolve([...stored]);
    },
  );
  assert.deepEqual(calls, ["runs", "messages"]);
  assert.equal(
    snapshot.messages.some(
      (message) => message.id === snapshot.activeRuns[0]?.assistantMessageId,
    ),
    true,
  );
});

test("terminal recovery restores final local usage and timing metadata", () => {
  const metadata = recoveredGenerationFinalMetadata({
    current: { generationSettled: true },
    run: {
      id: "run-1",
      requestPayload: { model: "local/model" },
      createdAt: 100,
      startedAt: 120,
      completedAt: 1120,
    },
    usage: {
      prompt_tokens: 8,
      completion_tokens: 12,
      total_tokens: 20,
      prompt_tokens_details: { cached_tokens: 3 },
    },
    timings: { predicted_per_second: 12, prompt_ms: 25 },
    firstChunkAt: 220,
    totalChunks: 4,
  });
  assert.deepEqual(metadata.contextUsage, {
    promptTokens: 8,
    completionTokens: 12,
    totalTokens: 20,
    cachedTokens: 3,
    cacheWriteTokens: 0,
    modelId: "local/model",
  });
  assert.deepEqual(metadata.serverTimings, {
    predicted_per_second: 12,
    prompt_ms: 25,
  });
  assert.deepEqual(metadata.timing, {
    streamStartTime: 120,
    firstTokenTime: 100,
    totalStreamTime: 1000,
    tokenCount: 12,
    tokensPerSecond: 12,
    totalChunks: 4,
    toolCallCount: 0,
  });
  assert.deepEqual(metadata.responseDetails, {
    modelId: "local/model",
    modelLabel: "local/model",
    responseModelId: "local/model",
    providerName: "Local model",
    providerType: "local",
    startedAt: 120,
    finishedAt: 1120,
    durationMs: 1000,
    cancelId: "run-1",
    toolCalls: [],
  });
});

test("reload, wake, and stale-tab recovery stays monotonic and truthful", () => {
  const apply = (
    status: "running" | "completed" | "failed" | "cancelled",
    cursor: number,
    lengthLimited = false,
  ) =>
    generationRecoveryMetadata({
      current: { generationRunId: "run-1" },
      runId: "run-1",
      status,
      cursor,
      lastEventSeq: 4,
      lengthLimited,
    });
  assert.deepEqual(
    [
      ["running", 2],
      ["completed", 2, true],
      ["completed", 4, true],
      ["failed", 4],
      ["cancelled", 4],
    ].map(([status, cursor, limited]) => {
      const metadata = apply(
        status as "running" | "completed" | "failed" | "cancelled",
        cursor as number,
        Boolean(limited),
      );
      return [generationNeedsRecovery(metadata), metadata.incomplete];
    }),
    [
      [true, { reason: "cancelled" }],
      [true, { reason: "cancelled" }],
      [false, { reason: "length" }],
      [false, { reason: "interrupted" }],
      [false, { reason: "cancelled" }],
    ],
  );

  const windowTarget = new EventTarget();
  const documentTarget = Object.assign(new EventTarget(), {
    visibilityState: "hidden",
  });
  let recoveries = 0;
  const unsubscribe = subscribeGenerationRecoveryTriggers(
    windowTarget,
    documentTarget,
    () => {
      recoveries += 1;
    },
  );
  windowTarget.dispatchEvent(new Event("online"));
  windowTarget.dispatchEvent(new Event("pageshow"));
  documentTarget.dispatchEvent(new Event("visibilitychange"));
  documentTarget.visibilityState = "visible";
  documentTarget.dispatchEvent(new Event("visibilitychange"));
  unsubscribe();
  windowTarget.dispatchEvent(new Event("online"));
  assert.equal(recoveries, 3);

  const existing = {
    generationRunId: "run-1",
    generationSeq: 4,
    generationStatus: "completed",
    generationSettled: true,
    serverManaged: true,
  };
  const incoming = { ...existing };
  assert.deepEqual(
    [
      incoming,
      { ...incoming, generationSeq: 3 },
      { ...incoming, generationRunId: "run-2" },
      { ...incoming, generationStatus: "running" },
      { ...incoming, generationSettled: false },
    ].map((candidate) => shouldPreserveGenerationMetadata(existing, candidate)),
    [false, true, true, true, true],
  );
});

test("recovery persists the replay prefix statistics it has applied", () => {
  const metadata = generationRecoveryMetadata({
    current: {
      generationRunId: "run-1",
      generationChunkCount: 4,
    },
    runId: "run-1",
    status: "running",
    cursor: 7,
    lastEventSeq: 9,
    lengthLimited: false,
    firstChunkAt: 220,
    totalChunks: 6,
  });
  assert.equal(metadata.generationFirstChunkAt, 220);
  assert.equal(metadata.generationChunkCount, 6);
});
