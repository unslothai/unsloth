// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  DEFAULT_PER_MODEL_CONFIG,
  isDefaultConfig,
  isReasoningBudgetMessageValid,
  normalizePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("reasoning budget fields normalize to the backend defaults", () => {
  assert.deepEqual(normalizePerModelConfig({}), DEFAULT_PER_MODEL_CONFIG);
  assert.equal(
    normalizePerModelConfig({ reasoningBudget: -10 }).reasoningBudget,
    -1,
  );
  assert.equal(
    normalizePerModelConfig({ reasoningBudget: 12.9 }).reasoningBudget,
    12,
  );
  assert.equal(isDefaultConfig(DEFAULT_PER_MODEL_CONFIG), true);
  assert.equal(
    isDefaultConfig({ ...DEFAULT_PER_MODEL_CONFIG, reasoningBudget: 0 }),
    false,
  );
});

test("reasoning budget messages use the subprocess byte boundary", () => {
  assert.equal(isReasoningBudgetMessageValid("😀".repeat(2_048)), true);
  assert.equal(isReasoningBudgetMessageValid("😀".repeat(2_049)), false);
  assert.equal(isReasoningBudgetMessageValid("bad\0message"), false);
  assert.equal(isReasoningBudgetMessageValid("  PAD  "), true);
  assert.equal(
    normalizePerModelConfig({ reasoningBudgetMessage: "😀".repeat(2_049) })
      .reasoningBudgetMessage,
    "",
  );
  assert.equal(
    normalizePerModelConfig({ reasoningBudgetMessage: "  PAD  " })
      .reasoningBudgetMessage,
    "  PAD  ",
  );
});

test("reasoning budget and message round-trip through per-model storage", () => {
  store.clear();
  const config = {
    ...DEFAULT_PER_MODEL_CONFIG,
    reasoningBudget: 2048,
    reasoningBudgetMessage: "Reasoning budget exhausted",
  };
  assert.equal(savePerModelConfig("unsloth/Test-GGUF", "Q4_K_M", config), true);

  const resolved = resolveInitialConfig("unsloth/Test-GGUF", "Q4_K_M");
  assert.equal(resolved.remembered, true);
  assert.equal(resolved.config.reasoningBudget, 2048);
  assert.equal(
    resolved.config.reasoningBudgetMessage,
    "Reasoning budget exhausted",
  );
});
