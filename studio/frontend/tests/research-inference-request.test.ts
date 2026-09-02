// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { buildResearchInferenceRequest } from "../src/features/chat/research-inference-request.ts";

const clamp = (effort: "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max") =>
  effort === "xhigh" ? "high" as const : effort;

test("Codex research keeps provider routing and clamps generation settings", () => {
  assert.deepEqual(
    buildResearchInferenceRequest({
      checkpoint: "external::provider::gpt-5.6-sol",
      external: {
        providerId: "provider",
        providerType: "openai_codex",
        modelId: "gpt-5.6-sol",
      },
      temperature: 0.2,
      topP: 0.9,
      maxTokens: 20000,
      reasoningRequested: true,
      supportsReasoningOff: true,
      reasoningStyle: "reasoning_effort",
      reasoningEffort: "xhigh",
      reasoningEffortLevels: ["low", "medium", "high"],
      clampReasoningEffort: clamp,
    }),
    {
      model: "gpt-5.6-sol",
      providerId: "provider",
      providerType: "openai_codex",
      externalModel: "gpt-5.6-sol",
      temperature: 0.2,
      topP: 0.9,
      maxTokens: 8192,
      reasoningEffort: "high",
    },
  );
});

test("invalid optional settings do not leak into a local research request", () => {
  assert.deepEqual(
    buildResearchInferenceRequest({
      checkpoint: "local/model.gguf",
      temperature: 3,
      topP: 0,
      maxTokens: 0,
      reasoningRequested: false,
      supportsReasoningOff: true,
      reasoningStyle: "enable_thinking",
      reasoningEffort: "none",
      reasoningEffortLevels: ["none", "low"],
      clampReasoningEffort: clamp,
    }),
    { model: "local/model.gguf", enableThinking: false },
  );
});

// A reasoning_effort provider has no enableThinking to carry an off, so a
// research request that emits nothing leaves ollama on its Think-true default
// (server/routes.go sets it whenever the model can think and no control
// arrives), which is the #9649 report reappearing in the synthesis call.
test("Thinking off reaches research synthesis as an explicit none", () => {
  assert.deepEqual(
    buildResearchInferenceRequest({
      checkpoint: "external::provider::gpt-oss:20b",
      external: {
        providerId: "provider",
        providerType: "ollama",
        modelId: "gpt-oss:20b",
      },
      temperature: 0.2,
      topP: 0.9,
      maxTokens: 4096,
      reasoningRequested: false,
      supportsReasoningOff: true,
      reasoningStyle: "reasoning_effort",
      reasoningEffort: "medium",
      reasoningEffortLevels: ["low", "medium", "high", "max"],
      clampReasoningEffort: clamp,
    }),
    {
      model: "gpt-oss:20b",
      providerId: "provider",
      providerType: "ollama",
      externalModel: "gpt-oss:20b",
      temperature: 0.2,
      topP: 0.9,
      maxTokens: 4096,
      reasoningEffort: "none",
    },
  );
});

// gpt-5 rejects reasoning_effort "none", so its caps clear supportsReasoningOff
// and the off must stay unsent rather than 400 the synthesis call.
test("a provider without an off value sends no effort when reasoning is off", () => {
  assert.deepEqual(
    buildResearchInferenceRequest({
      checkpoint: "external::provider::gpt-5",
      external: {
        providerId: "provider",
        providerType: "openai",
        modelId: "gpt-5",
      },
      temperature: 0.2,
      topP: 0.9,
      maxTokens: 4096,
      reasoningRequested: false,
      supportsReasoningOff: false,
      reasoningStyle: "reasoning_effort",
      reasoningEffort: "medium",
      reasoningEffortLevels: ["low", "medium", "high"],
      clampReasoningEffort: clamp,
    }),
    {
      model: "gpt-5",
      providerId: "provider",
      providerType: "openai",
      externalModel: "gpt-5",
      temperature: 0.2,
      topP: 0.9,
      maxTokens: 4096,
    },
  );
});
