// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createTrainingStartRequestId } from "../src/features/training/lib/training-start-request-id.ts";

const UUID_V4_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

test("training start request IDs use randomUUID when available", () => {
  const expected = "12345678-1234-4123-8123-123456789abc";
  const cryptoSource = {
    randomUUID: () => expected,
    getRandomValues: <T extends ArrayBufferView>(array: T) => array,
  };

  assert.equal(createTrainingStartRequestId(cryptoSource), expected);
});

test("training start request IDs remain valid without randomUUID", () => {
  const cryptoSource = {
    getRandomValues: <T extends ArrayBufferView>(array: T) => {
      if (array instanceof Uint8Array) {
        array.fill(0);
      }
      return array;
    },
  };

  assert.equal(
    createTrainingStartRequestId(cryptoSource),
    "00000000-0000-4000-8000-000000000000",
  );
  assert.match(createTrainingStartRequestId(null), UUID_V4_PATTERN);
});
