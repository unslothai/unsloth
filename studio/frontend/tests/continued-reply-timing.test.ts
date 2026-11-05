// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { replyFinishedTimestamp } from "../src/lib/message-timestamp.ts";

import { readText } from "./helpers/kit.ts";

const { addPass, priorPassTiming, readPriorPassTiming, withPriorServerTimings, withPriorTiming } = await import(
  "../src/features/chat/utils/continued-timing.ts"
);
const { CONTINUATION_RUN_CONFIG_KEY, readContinuationRequest } = await import(
  "../src/features/chat/utils/continuation.ts"
);

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

test("a failed continuation keeps its actual finish time after aggregating active duration", () => {
  const timing = withPriorTiming(secondPass.timing, priorPassTiming(firstPass));
  assert.equal(replyFinishedTimestamp({
    status: { type: "incomplete" },
    metadata: { custom: { timing } },
  }), 9348);
  assert.equal(timing.totalStreamTime, 4548);
});

test("a first pass with no token puts the first token in a later pass", () => {
  const prior = { totalStreamTime: 2000, totalChunks: 0, toolCallCount: 0 };
  const timing = withPriorTiming(secondPass.timing, prior);
  assert.equal(timing.firstTokenTime, 2120);
  assert.equal(timing.tokenCount, 11);
});

test("three passes add up when each Continue carries the passes before it", () => {
  const pass = (ms: number, tokens: number, first: number, start: number) => ({
    totalStreamTime: ms, totalChunks: tokens + 1, toolCallCount: 0, tokenCount: tokens, firstTokenTime: first,
    startedAt: start, durationMs: ms,
    serverTimings: { predicted_n: tokens, predicted_ms: ms - 50, predicted_per_second: (tokens - 1) / ((ms - 50) / 1000) },
  });
  const one = pass(500, 64, 120, 1000);
  const two = addPass(one, pass(400, 64, 90, 9000));
  const three = addPass(two, pass(300, 20, 80, 20000));
  assert.equal(three.tokenCount, 148);
  assert.equal(three.totalStreamTime, 1200);
  assert.equal(three.durationMs, 1200);
  assert.equal(three.firstTokenTime, 120, "the answer's first token came in the first pass");
  assert.equal(three.startedAt, 1000);
  assert.equal(three.serverTimings?.predicted_n, 148);
  assert.equal(three.serverTimings?.predicted_ms, 1050);
  const request = readContinuationRequest({ custom: { [CONTINUATION_RUN_CONFIG_KEY]: { partial: "x", priorTiming: two } } });
  assert.deepEqual(request?.priorTiming, two);
});

test("a stopped or broken-off reply hands Resume every pass so far", () => {
  const carried = addPass(priorPassTiming(firstPass), {
    totalStreamTime: 900, totalChunks: 40, toolCallCount: 0, tokenCount: 39, startedAt: 9000, durationMs: 900,
    serverTimings: { predicted_n: 39, predicted_ms: 800, predicted_per_second: 38 / 0.8 },
  });
  const stopped = { custom: { incomplete: { reason: "cancelled" }, resumeTiming: carried } };
  assert.deepEqual(priorPassTiming(stopped), carried);
  assert.equal(priorPassTiming(stopped)?.tokenCount, 128 + 39);
  const brokeOff = { custom: { timing: { totalStreamTime: 5100, tokenCount: 167, totalChunks: 170 }, resumeTiming: carried } };
  assert.equal(priorPassTiming(brokeOff)?.serverTimings?.predicted_n, 128 + 39);
  assert.equal(priorPassTiming(brokeOff)?.startedAt, 1000);
});

test("Continue carries the figures and the adapter adds them on every way a pass ends", () => {
  const thread = readText("../src/components/assistant-ui/thread.tsx");
  assert.match(thread, /const priorTiming = priorPassTiming\(metadata\);\s*if \(priorTiming\) request\.priorTiming = priorTiming;/);
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /const passText = \(\) => cumulativeText\.slice\(continuationPartial\.length\);/);
  assert.match(adapter, /meta\?\.usage\?\.completion_tokens \?\? estimateTokenCount\(passText\(\)\)/);
  assert.match(adapter, /estimateTokenCount\(continuation \? passText\(\) : partialText\)/);
  assert.match(adapter, /const finalTiming = withPriorTiming\(/);
  assert.match(adapter, /const partialTiming = withPriorTiming\(/);
  assert.match(adapter, /serverTimings: finalServerTimings \?\? undefined,/);
  assert.match(adapter, /startedAt: priorTiming\?\.startedAt \?\? streamStartTime,/);
  const live = adapter.slice(adapter.indexOf("const liveCustom = () => ({"), adapter.indexOf("const liveCustom = () => ({") + 400);
  assert.match(live, /resumeTiming: passSoFar \? addPass\(priorTiming, passSoFar\(\)\) : priorTiming,/);
  const partial = adapter.slice(adapter.indexOf("const partialTiming = withPriorTiming("));
  assert.match(partial, /resumeTiming: passSoFar \? addPass\(priorTiming, passSoFar\(\)\) : priorTiming,/);
});
