// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_INFERENCE_PARAMS } from "../src/features/chat/types/runtime.ts";
import {
  SAVED_PROMPT_NAME_MAX_LENGTH,
  UNTITLED_SAVED_PROMPT_NAME,
  applySystemPromptToParams,
  savedPromptNameFromText,
} from "../src/features/chat/utils/saved-system-prompt.ts";

test("empty or whitespace-only text becomes Untitled Prompt", () => {
  assert.equal(savedPromptNameFromText(""), UNTITLED_SAVED_PROMPT_NAME);
  assert.equal(savedPromptNameFromText("   \n\t"), UNTITLED_SAVED_PROMPT_NAME);
});

test("the first non-empty line becomes the saved name", () => {
  assert.equal(
    savedPromptNameFromText("You are a concise editor.\nNever invent facts."),
    "You are a concise editor.",
  );
  assert.equal(
    savedPromptNameFromText("\n  Stay in character.  \nMore rules."),
    "Stay in character.",
  );
});

test("internal whitespace collapses and long names truncate", () => {
  assert.equal(savedPromptNameFromText("Speak   slowly"), "Speak slowly");
  const long = "A".repeat(SAVED_PROMPT_NAME_MAX_LENGTH + 12);
  assert.equal(
    savedPromptNameFromText(long).length,
    SAVED_PROMPT_NAME_MAX_LENGTH,
  );
});

test("applying a saved prompt only replaces the system prompt", () => {
  const next = applySystemPromptToParams(DEFAULT_INFERENCE_PARAMS, "Be brief.");
  assert.equal(next.systemPrompt, "Be brief.");
  assert.equal(next.temperature, DEFAULT_INFERENCE_PARAMS.temperature);
  assert.equal(next.topP, DEFAULT_INFERENCE_PARAMS.topP);
  assert.equal(next.maxTokens, DEFAULT_INFERENCE_PARAMS.maxTokens);
  assert.equal(next.systemVariables, DEFAULT_INFERENCE_PARAMS.systemVariables);
});
