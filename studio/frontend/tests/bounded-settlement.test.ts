// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { waitForSettledBatch } from "../src/features/chat/utils/bounded-settlement.ts";

test("an early failure survives a sibling that exceeds the batch deadline", async () => {
  const failure = new Error("delete failed");
  const stalled = new Promise<void>(() => undefined);

  await assert.rejects(
    waitForSettledBatch([Promise.reject(failure), stalled], 5),
    failure,
  );
});

test("a pending batch can time out without rejecting", async () => {
  const stalled = new Promise<void>(() => undefined);
  await waitForSettledBatch([Promise.resolve(), stalled], 5);
});
