// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore: store } =
  await import("../src/features/training/stores/training-config-store.ts");
const { isBaseModelName } =
  await import("../src/features/training/lib/base-model-name.ts");
const { GRPO_REASONING_SYSTEM_PROMPT, GRPO_DEFAULT_SYSTEM_PROMPT } =
  await import("../src/config/training.ts");

test("base model names are told apart from instruct ones", () => {
  for (const name of [
    "unsloth/Qwen3-4B-Base",
    "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
    "Qwen2.5-3B-base",
  ]) {
    assert.ok(isBaseModelName(name), name);
  }
  for (const name of [
    "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/gemma-3-1b-it",
    "org/database-llm",
    "base-org/Llama-3.2-1B-Instruct",
    null,
  ]) {
    assert.ok(!isBaseModelName(name), String(name));
  }
});

test("picking a base model turns the reasoning format on, as the notebooks do", () => {
  setAuthFetchHandler(() => new Response("{}", { status: 404 }));
  try {
    store.getState().reset();
    store.getState().setSelectedModel("unsloth/Qwen3-4B-Base");
    let state = store.getState();
    assert.equal(state.grpoReasoningFormat, true);
    assert.equal(state.grpoSystemPrompt, GRPO_REASONING_SYSTEM_PROMPT);
    // Still the user's switch for this model.
    store.getState().setGrpoReasoningFormat(false);
    store.getState().setSelectedModel("unsloth/Qwen3-4B-Base");
    assert.equal(store.getState().grpoReasoningFormat, false);
    store.getState().setSelectedModel("unsloth/Qwen2.5-3B-Instruct");
    state = store.getState();
    assert.equal(state.grpoReasoningFormat, false);
    assert.equal(state.grpoSystemPrompt, GRPO_DEFAULT_SYSTEM_PROMPT);
    store.getState().setSelectedModel("unsloth/Qwen3-4B-Base");
    store.getState().setSelectedModel("unsloth/gemma-3-1b-it");
    assert.equal(store.getState().grpoReasoningFormat, false);
  } finally {
    setAuthFetchHandler(null);
  }
});

test("the reasoning format turns on the notebooks' 100-step format warm-up", () => {
  store.getState().reset();
  store.getState().setGrpoReasoningFormat(true);
  assert.equal(store.getState().grpoFormatWarmupSteps, 100);
  // The user's own count stays; switching the format off runs no warm-up.
  store.getState().setGrpoFormatWarmupSteps(30);
  store.getState().setGrpoReasoningFormat(true);
  assert.equal(store.getState().grpoFormatWarmupSteps, 30);
  store.getState().setGrpoReasoningFormat(false);
  assert.equal(store.getState().grpoFormatWarmupSteps, 0);
  // Below the warm-up's context floor it stays off rather than block Start.
  store.getState().setContextLength(1024);
  store.getState().setGrpoReasoningFormat(true);
  assert.equal(store.getState().grpoFormatWarmupSteps, 0);
});
