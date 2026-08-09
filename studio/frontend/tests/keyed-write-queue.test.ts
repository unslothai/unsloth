// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { KeyedWriteQueue } from "../src/features/chat/utils/keyed-write-queue.ts";

function deferred(): {
  promise: Promise<void>;
  resolve: () => void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("writes for one key run in order", async () => {
  const queue = new KeyedWriteQueue();
  const first = deferred();
  const events: string[] = [];

  const firstWrite = queue.enqueue(["thread-1"], async () => {
    events.push("first-start");
    await first.promise;
    events.push("first-end");
  });
  const secondWrite = queue.enqueue(["thread-1"], () => {
    events.push("second");
    return Promise.resolve();
  });

  await Promise.resolve();
  assert.deepEqual(events, ["first-start"]);
  first.resolve();
  await Promise.all([firstWrite, secondWrite]);
  assert.deepEqual(events, ["first-start", "first-end", "second"]);
  assert.deepEqual(queue.keys(), []);
});

test("a multi-key write waits for every predecessor and blocks later writes", async () => {
  const queue = new KeyedWriteQueue();
  const first = deferred();
  const second = deferred();
  const batch = deferred();
  const batchStarted = deferred();
  const events: string[] = [];

  const firstWrite = queue.enqueue(["thread-1"], async () => {
    events.push("first-start");
    await first.promise;
    events.push("first-end");
  });
  const secondWrite = queue.enqueue(["thread-2"], async () => {
    events.push("second-start");
    await second.promise;
    events.push("second-end");
  });
  const batchWrite = queue.enqueue(["thread-1", "thread-2"], async () => {
    events.push("batch-start");
    batchStarted.resolve();
    await batch.promise;
    events.push("batch-end");
  });
  const laterWrite = queue.enqueue(["thread-1"], () => {
    events.push("later");
    return Promise.resolve();
  });

  await Promise.resolve();
  first.resolve();
  await firstWrite;
  assert.equal(events.includes("batch-start"), false);
  second.resolve();
  await secondWrite;
  await batchStarted.promise;
  assert.equal(events.at(-1), "batch-start");
  assert.equal(events.includes("later"), false);
  batch.resolve();
  await Promise.all([batchWrite, laterWrite]);
  assert.deepEqual(events.slice(-2), ["batch-end", "later"]);
});

test("a failed write does not poison the next write", async () => {
  const queue = new KeyedWriteQueue();
  const failure = new Error("write failed");
  const events: string[] = [];

  const failed = queue.enqueue(["thread-1"], () => Promise.reject(failure));
  const recovered = queue.enqueue(["thread-1"], () => {
    events.push("recovered");
    return Promise.resolve();
  });

  await assert.rejects(failed, failure);
  await recovered;
  assert.deepEqual(events, ["recovered"]);
});
