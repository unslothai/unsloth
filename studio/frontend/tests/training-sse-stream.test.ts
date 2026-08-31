// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { consumeTrainingProgressStream } from "../src/features/training/lib/training-sse-stream.ts";
import type { TrainingProgressPayload } from "../src/features/training/types/runtime.ts";

function progressPayload(jobId: string, step: number): TrainingProgressPayload {
  return {
    job_id: jobId,
    step,
    total_steps: 10,
    loss: null,
    learning_rate: null,
    progress_percent: step * 10,
    epoch: null,
    elapsed_seconds: null,
    eta_seconds: null,
    grad_norm: null,
    num_tokens: null,
    eval_loss: null,
  };
}

function rawEvent(
  data: string,
  options: {
    event?: "progress" | "heartbeat" | "complete" | "error";
    id?: number;
    lineEnding?: "\n" | "\r\n";
  } = {},
): string {
  const lineEnding = options.lineEnding ?? "\n";
  return [
    `event: ${options.event ?? "progress"}`,
    `id: ${options.id ?? 0}`,
    `data: ${data}`,
    "",
    "",
  ].join(lineEnding);
}

function event(
  jobId: string,
  step: number,
  options: Parameters<typeof rawEvent>[1] = {},
): string {
  return rawEvent(JSON.stringify(progressPayload(jobId, step)), {
    id: step,
    ...options,
  });
}

test("aborting one buffered event prevents later callbacks from the same chunk", async () => {
  const controller = new AbortController();
  const chunk = new TextEncoder().encode(
    `${event("job-stale", 1)}${event("job-current", 2)}`,
  );
  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      streamController.enqueue(chunk);
      streamController.close();
    },
  });
  const received: string[] = [];

  await consumeTrainingProgressStream({
    body,
    signal: controller.signal,
    onEvent: ({ payload }) => {
      received.push(payload.job_id);
      controller.abort();
    },
  });

  assert.deepEqual(received, ["job-stale"]);
  assert.equal(body.locked, false);
});

test("an already aborted stream never dispatches buffered data", async () => {
  const controller = new AbortController();
  controller.abort();
  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      streamController.enqueue(new TextEncoder().encode(event("job-1", 1)));
      streamController.close();
    },
  });
  let events = 0;

  await consumeTrainingProgressStream({
    body,
    signal: controller.signal,
    onEvent: () => {
      events += 1;
    },
  });

  assert.equal(events, 0);
  assert.equal(body.locked, false);
});

test("a completed stream releases its reader", async () => {
  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      streamController.enqueue(new TextEncoder().encode(event("job-1", 1)));
      streamController.close();
    },
  });

  await consumeTrainingProgressStream({
    body,
    signal: new AbortController().signal,
    onEvent: () => undefined,
  });

  assert.equal(body.locked, false);
});

test("malformed frames are skipped without interrupting valid frames", async () => {
  const minimalWrongJob = {
    job_id: "job-other",
    step: 2,
    total_steps: 10,
    progress_percent: 20,
  };
  const chunks = [
    event("job-current", 1),
    rawEvent("{"),
    rawEvent("null", { lineEnding: "\r\n" }),
    rawEvent("{}"),
    rawEvent(
      JSON.stringify({
        job_id: "job-current",
        step: 2,
        total_steps: 10,
      }),
    ),
    rawEvent(
      '{"job_id":"job-current","step":2,"total_steps":10,"progress_percent":1e999}',
    ),
    rawEvent(
      JSON.stringify({
        ...progressPayload("job-current", 2),
        elapsed_seconds: "later",
      }),
    ),
    rawEvent(JSON.stringify(minimalWrongJob), {
      event: "heartbeat",
      id: 2,
    }),
    event("job-current", 3, { event: "complete" }),
  ];
  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      for (const chunk of chunks) {
        streamController.enqueue(new TextEncoder().encode(chunk));
      }
      streamController.close();
    },
  });
  const received: Array<{
    event: string;
    jobId: string;
    step: number;
    elapsedSeconds: number | null;
  }> = [];

  await consumeTrainingProgressStream({
    body,
    signal: new AbortController().signal,
    onEvent: ({ event: eventName, payload }) => {
      received.push({
        event: eventName,
        jobId: payload.job_id,
        step: payload.step,
        elapsedSeconds: payload.elapsed_seconds,
      });
    },
  });

  assert.deepEqual(received, [
    {
      event: "progress",
      jobId: "job-current",
      step: 1,
      elapsedSeconds: null,
    },
    {
      event: "heartbeat",
      jobId: "job-other",
      step: 2,
      elapsedSeconds: null,
    },
    {
      event: "complete",
      jobId: "job-current",
      step: 3,
      elapsedSeconds: null,
    },
  ]);
  assert.equal(body.locked, false);
});

test("non-progress events accept backend preparation and terminal values", async () => {
  const preparationPayload = {
    job_id: "job-terminal",
    step: 0,
    total_steps: 0,
    progress_percent: 0,
  };
  const terminalPayload = { ...preparationPayload, step: -1 };
  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      streamController.enqueue(
        new TextEncoder().encode(
          `${rawEvent(JSON.stringify(preparationPayload), {
            event: "heartbeat",
          })}${rawEvent(JSON.stringify(terminalPayload), {
            event: "complete",
          })}${rawEvent(JSON.stringify(terminalPayload), { event: "error" })}`,
        ),
      );
      streamController.close();
    },
  });
  const received: string[] = [];

  await consumeTrainingProgressStream({
    body,
    signal: new AbortController().signal,
    onEvent: ({ event: eventName, payload }) => {
      received.push(`${eventName}:${payload.step}:${payload.total_steps}`);
    },
  });

  assert.deepEqual(received, ["heartbeat:0:0", "complete:-1:0", "error:-1:0"]);
  assert.equal(body.locked, false);
});

test("callback exceptions propagate after the reader is cancelled and released", async () => {
  const callbackError = new Error("callback failed");
  let cancellations = 0;
  let callbacks = 0;
  const body = new ReadableStream<Uint8Array>({
    start(streamController) {
      streamController.enqueue(
        new TextEncoder().encode(
          `${event("job-current", 1)}${event("job-current", 2)}`,
        ),
      );
    },
    cancel() {
      cancellations += 1;
      throw new Error("cancel failed");
    },
  });

  await assert.rejects(
    consumeTrainingProgressStream({
      body,
      signal: new AbortController().signal,
      onEvent: () => {
        callbacks += 1;
        throw callbackError;
      },
    }),
    (error: unknown) => error === callbackError,
  );

  assert.equal(callbacks, 1);
  assert.equal(cancellations, 1);
  assert.equal(body.locked, false);
});
