// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  settlesWithin,
  waitForSettledOrRunFallback,
} from "../src/features/chat/utils/bounded-settlement.ts";

test("reports work that confirms within the deadline", async () => {
  assert.equal(await settlesWithin(Promise.resolve(), 5), true);
});

test("reports work that remains unconfirmed at the deadline", async () => {
  const stalled = new Promise<void>(() => undefined);
  assert.equal(await settlesWithin(stalled, 5), false);
});

test("preserves a rejection that arrives within the deadline", async () => {
  const failure = new Error("delete failed");
  await assert.rejects(settlesWithin(Promise.reject(failure), 5), failure);
});

test("accepts an independent fallback only after it confirms", async () => {
  const stalled = new Promise<void>(() => undefined);
  let fallbackCalls = 0;

  await waitForSettledOrRunFallback(
    stalled,
    () => {
      fallbackCalls += 1;
      return Promise.resolve();
    },
    5,
  );

  assert.equal(fallbackCalls, 1);
});

test("rejects when the independent fallback also remains unconfirmed", async () => {
  const stalled = new Promise<void>(() => undefined);

  await assert.rejects(
    waitForSettledOrRunFallback(stalled, () => stalled, 5),
    /Timed out waiting for fallback work/,
  );
});
