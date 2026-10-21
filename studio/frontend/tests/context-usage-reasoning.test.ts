// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { contextUsageReasoningKey } from "../src/features/chat/utils/context-usage-reasoning.ts";

test("completion and recount reasoning fields identify the same effective mode", () => {
  for (const enabled of [true, false]) {
    assert.equal(
      contextUsageReasoningKey({ thinking: { type: enabled ? "enabled" : "disabled" } }),
      contextUsageReasoningKey({ enable_thinking: enabled }),
    );
  }
  assert.equal(
    contextUsageReasoningKey({ thinking: { type: "enabled" }, enable_thinking: false }),
    contextUsageReasoningKey({ enable_thinking: false }),
  );
});

test("reasoning effort, explicit omission, and preserved thinking keep separate count identities", () => {
  const modes = [
    {},
    { enable_thinking: false },
    { enable_thinking: true },
    { enable_thinking: true, reasoning_effort: "low" },
    { enable_thinking: true, reasoning_effort: "high" },
    { enable_thinking: true, reasoning_effort: "high", preserve_thinking: false },
    { enable_thinking: true, reasoning_effort: "high", preserve_thinking: true },
  ];
  assert.equal(new Set(modes.map(contextUsageReasoningKey)).size, modes.length);
});

test("omitted reasoning fields and serialized null defaults match", () => {
  assert.equal(
    contextUsageReasoningKey({ enable_thinking: null, reasoning_effort: null, preserve_thinking: null, thinking: null }),
    contextUsageReasoningKey({}),
  );
});
