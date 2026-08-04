// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { consumeTrainingProgressStream } from "../src/features/training/lib/training-sse-stream.ts";

function event(jobId: string, step: number): string {
  return [
    "event: progress",
    `id: ${step}`,
    `data: ${JSON.stringify({ job_id: jobId, step })}`,
    "",
    "",
  ].join("\n");
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
