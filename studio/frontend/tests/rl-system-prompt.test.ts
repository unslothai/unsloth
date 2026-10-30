// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { configuredSystemPrompt, effectiveSystemPrompt } = await import(
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

test("with the reasoning format on, an empty prompt is the reasoning prompt training gets", () => {
  assert.equal(configuredSystemPrompt("", true, "Use <think>"), "Use <think>");
  assert.equal(configuredSystemPrompt("  ", true, "Use <think>"), "Use <think>");
  assert.equal(configuredSystemPrompt("", false, "Use <think>"), "");
  assert.equal(configuredSystemPrompt("Mine", true, "Use <think>"), "Mine");
});

test("the reasoning format offers no 'no system prompt' preset, since training would add one", () => {
  const src = readFileSync(
    new URL("../src/features/studio/sections/rl/prompt-format-section.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /value: "none",\s*label: presetLabel\("none"\),[^}]*disabled: s\.reasoning/);
  assert.match(src, /effectiveSystemPrompt\(previewRow, s\.mapping, configured\)/);
});

test("Try it previews rewards with the prompt training gets under the reasoning format", () => {
  const src = readFileSync(
    new URL("../src/features/studio/sections/rl/rewards-section.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /previewPrompt\(\s*previewRow,\s*mapping,[\s\S]{0,160}configuredSystemPrompt\(\s*grpoSystemPrompt,\s*reasoning,/,
  );
});
