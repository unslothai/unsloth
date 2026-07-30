// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createSingleFlight } from "../src/features/chat/api/single-flight.ts";

function deferred<T>() {
  let resolve: (value: T) => void = () => undefined;
  let reject: (error: unknown) => void = () => undefined;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

test("concurrent callers share one operation and its committed result", async () => {
  const flight = createSingleFlight<{ committed: boolean }>();
  const operation = deferred<{ committed: boolean }>();
  let starts = 0;
  const start = () => {
    starts += 1;
    return operation.promise;
  };

  const first = flight.run(new AbortController().signal, start);
  const second = flight.run(new AbortController().signal, start);
  assert.equal(starts, 1);

  operation.resolve({ committed: true });
  assert.deepEqual(await first, { committed: true });
  assert.deepEqual(await second, { committed: true });
});

test("an aborted waiter does not cancel or poison the shared operation", async () => {
  const flight = createSingleFlight<string>();
  const operation = deferred<string>();
  const firstController = new AbortController();
  const secondController = new AbortController();
  const reason = new Error("first caller stopped");
  let starts = 0;
  const start = () => {
    starts += 1;
    return operation.promise;
  };

  const first = flight.run(firstController.signal, start);
  const second = flight.run(secondController.signal, start);
  firstController.abort(reason);

  await assert.rejects(first, (error) => error === reason);
  operation.resolve("loaded");
  assert.equal(await second, "loaded");
  assert.equal(starts, 1);
});

test("an already-aborted caller never starts the operation", async () => {
  const flight = createSingleFlight<void>();
  const controller = new AbortController();
  const reason = new Error("stopped before claim");
  controller.abort(reason);
  let starts = 0;

  await assert.rejects(
    flight.run(controller.signal, async () => {
      starts += 1;
    }),
    (error) => error === reason,
  );
  assert.equal(starts, 0);
});

test("a failed operation clears the slot for one fresh retry", async () => {
  const flight = createSingleFlight<string>();
  const failure = new Error("load failed");
  let starts = 0;
  const fail = () => {
    starts += 1;
    return Promise.reject(failure);
  };

  const first = flight.run(new AbortController().signal, fail);
  const second = flight.run(new AbortController().signal, fail);
  await assert.rejects(first, (error) => error === failure);
  await assert.rejects(second, (error) => error === failure);
  assert.equal(starts, 1);

  assert.equal(
    await flight.run(new AbortController().signal, async () => {
      starts += 1;
      return "retried";
    }),
    "retried",
  );
  assert.equal(starts, 2);
});
