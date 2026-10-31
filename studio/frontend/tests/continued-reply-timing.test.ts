// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A continued reply (Max Tokens auto-continue, or Continue) shows one merged answer, so its timing
// tooltip, badge and response details describe the whole answer, not the last request.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const { priorPassTiming, readPriorPassTiming, withPriorServerTimings, withPriorTiming } = await import(
  "../src/features/chat/utils/continued-timing.ts"
);
const { CONTINUATION_RUN_CONFIG_KEY, readContinuationRequest } = await import(
  "../src/features/chat/utils/continuation.ts"
);

// A 128-token first pass cut by Max Tokens, as the adapter stores it.
const firstPass = {
  timing: { streamStartTime: 1000, firstTokenTime: 300, totalStreamTime: 4200, tokenCount: 128, totalChunks: 129, toolCallCount: 0 },
  custom: {
    timing: { streamStartTime: 1000, firstTokenTime: 300, totalStreamTime: 4200, tokenCount: 128, totalChunks: 129, toolCallCount: 0 },
    serverTimings: {
      prompt_n: 40, prompt_ms: 100, prompt_per_second: 400, cache_n: 0,
      predicted_n: 128, predicted_ms: 3800, predicted_per_second: 127 / 3.8,
      draft_n: 64, draft_n_accepted: 16,
    },
    responseDetails: { startedAt: 1000, finishedAt: 5200, durationMs: 4200 },
  },
};

// The 11-token continuation from the report, which the tooltip showed as the whole answer.
const secondPass = {
  timing: { streamStartTime: 9000, firstTokenTime: 120, totalStreamTime: 348, tokenCount: 11, totalChunks: 12, toolCallCount: 0, tokensPerSecond: 31.6 },
  serverTimings: {
    prompt_n: 3, prompt_ms: 20, prompt_per_second: 150, cache_n: 168,
    predicted_n: 11, predicted_ms: 300, predicted_per_second: 10 / 0.3,
  } as Record<string, number>,
};

test("the resumed reply's figures are read from its metadata", () => {
  const prior = priorPassTiming(firstPass);
  assert.deepEqual(prior, {
    totalStreamTime: 4200,
    totalChunks: 129,
    toolCallCount: 0,
    tokenCount: 128,
    firstTokenTime: 300,
    durationMs: 4200,
    startedAt: 1000,
    serverTimings: firstPass.custom.serverTimings,
  });
  assert.equal(priorPassTiming({ custom: {} }), undefined, "no timing, nothing to carry");
  assert.equal(priorPassTiming(undefined), undefined);
  // A diffusion model's steps and blocks are not a decode that continues.
  const diffusion = { ...firstPass, custom: { ...firstPass.custom, serverTimings: { diffusion: true, diffusion_steps: 8 } } };
  assert.equal(priorPassTiming(diffusion)?.serverTimings, undefined);
});

test("the carried figures survive the run config and reject junk", () => {
  const priorTiming = priorPassTiming(firstPass);
  const request = readContinuationRequest({
    custom: { [CONTINUATION_RUN_CONFIG_KEY]: { partial: "Once upon", priorTiming } },
  });
  assert.deepEqual(request?.priorTiming, priorTiming);
  const junk = readContinuationRequest({
    custom: { [CONTINUATION_RUN_CONFIG_KEY]: { partial: "Once upon", priorTiming: { totalStreamTime: "4s" } } },
  });
  assert.equal(junk?.priorTiming, undefined);
  assert.equal(
    readPriorPassTiming({ totalStreamTime: 10, tokenCount: -1, serverTimings: { predicted_n: Number.NaN } })?.tokenCount,
    undefined,
  );
});

test("a continued reply's tokens, time and speed cover the whole answer", () => {
  const prior = priorPassTiming(firstPass);
  const server = withPriorServerTimings(secondPass.serverTimings, prior?.serverTimings);
  const timing = withPriorTiming(secondPass.timing, prior, server?.predicted_per_second);
  // The report: 128 + 11 tokens shown as "11 tokens / 348 ms".
  assert.equal(timing.tokenCount, 139);
  assert.equal(timing.totalStreamTime, 4548);
  assert.equal(timing.firstTokenTime, 300, "the answer's first token came in the first pass");
  assert.equal(timing.totalChunks, 141);
  assert.ok(server);
  assert.equal(server.predicted_n, 139);
  assert.equal(server.predicted_ms, 4100);
  assert.equal(server.prompt_n, 43);
  assert.equal(server.draft_n, 64, "a pass without drafts adds none");
  assert.equal(server.draft_n_accepted, 16);
  // llama.cpp's rate is decode steps over decode time; the merge keeps that across passes.
  assert.ok(Math.abs(server.predicted_per_second - (127 + 10) / 4.1) < 1e-9);
  assert.equal(timing.tokensPerSecond, server.predicted_per_second);
});

test("a reply that was never continued keeps its own figures", () => {
  const timing = withPriorTiming(secondPass.timing, undefined, 99);
  assert.equal(timing, secondPass.timing);
  assert.equal(withPriorServerTimings(secondPass.serverTimings, undefined), secondPass.serverTimings);
  const diffusion = { diffusion: true, diffusion_steps: 8 };
  assert.equal(withPriorServerTimings(diffusion, { predicted_n: 5 }), diffusion);
});

test("a first pass with no token puts the first token in a later pass", () => {
  const prior = { totalStreamTime: 2000, totalChunks: 0, toolCallCount: 0 };
  const timing = withPriorTiming(secondPass.timing, prior);
  assert.equal(timing.firstTokenTime, 2120);
  assert.equal(timing.tokenCount, 11);
});

// The adapter runs only against a live provider stream, so its wiring is checked in the source.
test("Continue carries the figures and the adapter adds them on every way a pass ends", () => {
  const thread = readText("../src/components/assistant-ui/thread.tsx");
  assert.match(thread, /const priorTiming = priorPassTiming\(metadata\);\s*if \(priorTiming\) request\.priorTiming = priorTiming;/);
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /const passText = \(\) => cumulativeText\.slice\(continuationPartial\.length\);/);
  // Counted from this pass's own output, never the merged text, or the earlier passes count twice.
  assert.match(adapter, /meta\?\.usage\?\.completion_tokens \?\? estimateTokenCount\(passText\(\)\)/);
  assert.match(adapter, /estimateTokenCount\(continuation \? passText\(\) : partialText\)/);
  assert.match(adapter, /const finalTiming = withPriorTiming\(/);
  assert.match(adapter, /const partialTiming = withPriorTiming\(/);
  assert.match(adapter, /serverTimings: finalServerTimings \?\? undefined,/);
  assert.match(adapter, /startedAt: priorTiming\?\.startedAt \?\? streamStartTime,/);
});
