// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  sessionEtaSeconds,
  sessionStepsPerSecond,
} from "../src/features/studio/sections/progress-section-lib.ts";
import { consumeTrainingProgressStream } from "../src/features/training/lib/training-sse-stream.ts";
import { useTrainingRuntimeStore } from "../src/features/training/stores/training-runtime-store.ts";

function payload(step: number, extra: Record<string, unknown> = {}) {
  return {
    job_id: "job-resumed",
    step,
    total_steps: 1000,
    loss: 0.5,
    learning_rate: 0.0001,
    progress_percent: step / 10,
    epoch: 0.9,
    elapsed_seconds: 60,
    eta_seconds: 540,
    grad_norm: null,
    num_tokens: null,
    eval_loss: null,
    ...extra,
  };
}

test("a resumed run's throughput counts only the steps done in this session", () => {
  assert.equal(sessionStepsPerSecond(910, 900, 60), 10 / 60);
  assert.equal(sessionStepsPerSecond(100, 0, 60), 100 / 60);
  assert.equal(sessionStepsPerSecond(900, 900, 60), 0);
  assert.equal(sessionStepsPerSecond(0, 900, 60), null);
  assert.equal(sessionStepsPerSecond(910, 900, null), null);
  assert.equal(sessionStepsPerSecond(910, 900, 0), null);
});

test("the fallback ETA uses the same session window", () => {
  assert.equal(sessionEtaSeconds(910, 900, 1000, 60), 540);
  assert.equal(sessionEtaSeconds(900, 900, 1000, 60), null);
  assert.equal(sessionEtaSeconds(910, 900, 1000, null), null);
});

test("the session start step travels from the SSE frame into the store", async () => {
  const frames = [payload(910, { session_start_step: 900 }), payload(920)]
    .map((p) => `event: progress\nid: ${p.step}\ndata: ${JSON.stringify(p)}\n\n`)
    .join("");
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(frames));
      controller.close();
    },
  });
  useTrainingRuntimeStore.getState().resetRuntime();
  useTrainingRuntimeStore.getState().setStartPending("job-resumed", "Starting");
  assert.equal(useTrainingRuntimeStore.getState().sessionStartStep, 0);

  await consumeTrainingProgressStream({
    body,
    signal: new AbortController().signal,
    onEvent: ({ payload: parsed, id }) =>
      useTrainingRuntimeStore.getState().applyProgress(parsed, id ?? undefined),
  });

  const state = useTrainingRuntimeStore.getState();
  assert.equal(state.currentStep, 920);
  assert.equal(state.sessionStartStep, 900);

  useTrainingRuntimeStore.getState().setStartPending("job-next", "Starting");
  assert.equal(useTrainingRuntimeStore.getState().sessionStartStep, 0);
});
