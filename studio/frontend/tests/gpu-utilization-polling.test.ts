// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createSingleFlight,
  type PollingTimer,
  startSerialPolling,
} from "../src/hooks/gpu-utilization-polling.ts";

function deferred(): {
  promise: Promise<void>;
  resolve: () => void;
} {
  let resolve = () => {};
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function fakeTimer(): PollingTimer & {
  callbacks: Array<() => void>;
  cleared: Array<ReturnType<typeof setTimeout>>;
} {
  const callbacks: Array<() => void> = [];
  const cleared: Array<ReturnType<typeof setTimeout>> = [];
  return {
    callbacks,
    cleared,
    setTimeout(callback) {
      callbacks.push(callback);
      return callbacks.length as unknown as ReturnType<typeof setTimeout>;
    },
    clearTimeout(timer) {
      cleared.push(timer);
    },
  };
}

test("serial polling schedules the next request only after completion", async () => {
  const timer = fakeTimer();
  const first = deferred();
  const second = deferred();
  let calls = 0;
  const stop = startSerialPolling(
    () => {
      calls += 1;
      return calls === 1 ? first.promise : second.promise;
    },
    5000,
    timer,
  );

  assert.equal(calls, 1);
  assert.equal(timer.callbacks.length, 0);

  first.resolve();
  await first.promise;
  await Promise.resolve();
  assert.equal(timer.callbacks.length, 1);

  timer.callbacks.shift()?.();
  assert.equal(calls, 2);
  assert.equal(timer.callbacks.length, 0);

  stop();
  second.resolve();
  await second.promise;
  await Promise.resolve();
  assert.equal(timer.callbacks.length, 0);
});

test("single-flight polling shares work across concurrent consumers", async () => {
  const first = deferred();
  let calls = 0;
  const poll = createSingleFlight(() => {
    calls += 1;
    return first.promise;
  });

  const firstRequest = poll();
  const secondRequest = poll();
  assert.equal(firstRequest, secondRequest);
  assert.equal(calls, 1);

  first.resolve();
  await firstRequest;
  await Promise.resolve();
  void poll();
  assert.equal(calls, 2);
});

test("stopping serial polling clears a scheduled request", async () => {
  const timer = fakeTimer();
  const first = deferred();
  let calls = 0;
  const stop = startSerialPolling(() => {
    calls += 1;
    return first.promise;
  }, 5000, timer);

  first.resolve();
  await first.promise;
  await Promise.resolve();
  const scheduled = timer.callbacks[0];
  stop();

  assert.deepEqual(timer.cleared, [1]);
  scheduled();
  assert.equal(calls, 1);
});
