// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createRetryableSharedRead } from "../src/features/chat/utils/retryable-shared-read.ts";

test("shares an in-flight read and its successful result", async () => {
  let calls = 0;
  const read = createRetryableSharedRead(async () => {
    calls += 1;
    await Promise.resolve();
    return "thread";
  });

  const first = read();
  const second = read();
  assert.equal(first, second);
  assert.deepEqual(await Promise.all([first, second]), ["thread", "thread"]);
  assert.equal(await read(), "thread");
  assert.equal(calls, 1);
});

test("retries after a failed read", async () => {
  let calls = 0;
  const read = createRetryableSharedRead(async () => {
    calls += 1;
    if (calls === 1) {
      throw new Error("temporary failure");
    }
    return "thread";
  });

  await assert.rejects(read(), /temporary failure/);
  assert.equal(await read(), "thread");
  assert.equal(calls, 2);
});

test("retries after a successful value marked as non-cacheable", async () => {
  let attempts = 0;
  const read = createRetryableSharedRead(
    async () => ++attempts,
    (value) => value > 1,
  );

  assert.equal(await read(), 1);
  assert.equal(await read(), 2);
  assert.equal(await read(), 2);
  assert.equal(attempts, 2);
});

test("every concurrent caller sees a rejection, from one underlying read", async () => {
  let calls = 0;
  const read = createRetryableSharedRead(async () => {
    calls += 1;
    await Promise.resolve();
    throw new Error("backend down");
  });

  const settled = await Promise.allSettled([read(), read(), read()]);
  assert.deepEqual(
    settled.map((r) => r.status),
    ["rejected", "rejected", "rejected"],
  );
  assert.equal(calls, 1);
});

test("a read that throws synchronously rejects rather than escaping", async () => {
  let calls = 0;
  const read = createRetryableSharedRead(() => {
    calls += 1;
    throw new Error("synchronous failure");
  });

  await assert.rejects(read(), /synchronous failure/);
  await assert.rejects(read(), /synchronous failure/);
  assert.equal(calls, 2);
});

test("an undefined record still counts as a cached value", async () => {
  // Incognito and deleted threads resolve to undefined; that is an answer, not
  // a miss, so it must not be re-read on every consumer.
  let calls = 0;
  const read = createRetryableSharedRead(
    async () => {
      calls += 1;
      return { thread: undefined, cacheable: true };
    },
    (result) => result.cacheable,
  );

  assert.equal((await read()).thread, undefined);
  assert.equal((await read()).thread, undefined);
  assert.equal(calls, 1);
});
