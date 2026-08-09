// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  settleWithin,
  waitForSettledOrRunFallback,
} from "../src/features/chat/utils/bounded-settlement.ts";

test("bounds ignored work and returns confirmed work", async () => {
  const stalled = new Promise<void>(() => undefined);
  await settleWithin(stalled, 5);
  const first = waitForSettledOrRunFallback(
    Promise.resolve("first"),
    () => Promise.resolve("retry"),
    5,
  );
  assert.equal(await first, "first");
});

test("preserves a deletion failure that arrives before the fallback", async () => {
  const failure = new Error("delete failed");
  let fallbackCalled = false;
  await assert.rejects(
    waitForSettledOrRunFallback(
      Promise.reject(failure),
      () => {
        fallbackCalled = true;
        return Promise.resolve();
      },
      5,
    ),
    failure,
  );
  assert.equal(fallbackCalled, false);
});

test("returns only after the independent fallback confirms", async () => {
  const stalled = new Promise<string>(() => undefined);
  let confirmFallback!: (value: string) => void;
  const fallback = new Promise<string>((resolve) => {
    confirmFallback = resolve;
  });
  let confirmed = false;

  const waiting = waitForSettledOrRunFallback(stalled, () => fallback, 5).then(
    (result) => {
      confirmed = true;
      return result;
    },
  );
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(confirmed, false);

  confirmFallback("confirmed");
  assert.equal(await waiting, "confirmed");
  assert.equal(confirmed, true);
});

test("a timed-out legacy backfill retains its binding in thread listings", () => {
  const source = readFileSync(
    new URL("../src/features/chat/utils/chat-history-storage.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /pendingLegacyThreadBackfills\.set[\s\S]*timedOutLegacyBackfills\.set[\s\S]*\.\.\.timedOutLegacyBackfills\.get\(thread\.id\)/,
  );
  assert.doesNotMatch(source, /importDrainTimedOut \|\|\s*!backendThreads/);
});
