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
