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
      reasoningStyle: "enable_thinking",
      reasoningEffort: "none",
      reasoningEffortLevels: ["none", "low"],
      clampReasoningEffort: clamp,
    }),
    { model: "local/model.gguf", enableThinking: false },
  );
});
