// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { readIncompleteInfo, restoredAssistantStatus } = await import(
  "../src/features/chat/utils/continuation.ts"
);

test("stored assistant status remains truthful after reload", () => {
  for (const reason of ["interrupted", "length"] as const) {
    const metadata = { custom: { incomplete: { reason } } };
    assert.deepEqual(restoredAssistantStatus(metadata), {
      type: "incomplete",
      reason: "error",
    });
    assert.deepEqual(readIncompleteInfo(metadata), { reason });
  }

  assert.deepEqual(restoredAssistantStatus({ custom: {} }), {
    type: "complete",
    reason: "unknown",
  });
  assert.deepEqual(restoredAssistantStatus(undefined), {
    type: "complete",
    reason: "unknown",
  });
});
