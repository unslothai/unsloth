// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { effectiveSystemPrompt } = await import(
  "../src/features/training/lib/rl-system-prompt.ts"
);

const mapping = { question: "prompt", sys: "system" };

test("a row's system column is what the GRPO preview shows, as training uses it", () => {
  const row = { question: "2+2?", sys: "Return only the number." };
  assert.equal(
    effectiveSystemPrompt(row, mapping, "Use <answer>"),
    "Return only the number.",
  );
});

test("a conversation's own system message wins over the column and the preset", () => {
  const row = {
    question: [
      { role: "system", content: "Be terse." },
      { role: "user", content: "2+2?" },
    ],
    sys: "Return only the number.",
  };
  assert.equal(
    effectiveSystemPrompt(row, mapping, "Use <answer>"),
    "Be terse.",
  );
});

test("rows without their own system message show the configured prompt", () => {
  assert.equal(
    effectiveSystemPrompt(
      { question: "2+2?", sys: "" },
      mapping,
      "Use <answer>",
    ),
    "Use <answer>",
  );
  assert.equal(
    effectiveSystemPrompt(null, mapping, "Use <answer>"),
    "Use <answer>",
  );
});
