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
        maxOutputTokens: 128000,
        maxOutputTokensFromSavedCap: false,
        maxOutputTokensPublished: null,
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
      maxOutputTokens: 128000,
      maxOutputTokensFromSavedCap: false,
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

test("the report ceiling the connection resolved reaches the run config", () => {
  const request = buildResearchInferenceRequest({
    checkpoint: "external::provider::gemini-3.6-flash",
    external: {
      providerId: "provider",
      providerType: "gemini",
      modelId: "gemini-3.6-flash",
      maxOutputTokens: 65536,
      maxOutputTokensFromSavedCap: false,
      maxOutputTokensPublished: null,
    },
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 4096,
    reasoningRequested: false,
    reasoningStyle: "none",
    reasoningEffort: "low",
    reasoningEffortLevels: ["low", "medium", "high"],
    clampReasoningEffort: clamp,
  });

  assert.equal(request.maxOutputTokens, 65536);
  assert.equal(request.maxTokens, 4096);
});

test("an undocumented model with no connection override sends no ceiling", () => {
  const request = buildResearchInferenceRequest({
    checkpoint: "local-model",
    external: {
      providerId: "provider",
      providerType: "custom",
      modelId: "some-self-hosted-model",
      // What getGroundedExternalMaxOutputTokens returns when nothing documents the model.
      maxOutputTokens: null,
      maxOutputTokensFromSavedCap: false,
      maxOutputTokensPublished: null,
    },
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 4096,
    reasoningRequested: false,
    reasoningStyle: "none",
    reasoningEffort: "low",
    reasoningEffortLevels: ["low", "medium", "high"],
    clampReasoningEffort: clamp,
  });
  assert.equal("maxOutputTokens" in request, false);
});

test("an explicit connection override is still sent", () => {
  const request = buildResearchInferenceRequest({
    checkpoint: "local-model",
    external: {
      providerId: "provider",
      providerType: "custom",
      modelId: "some-self-hosted-model",
      maxOutputTokens: 20000,
      maxOutputTokensFromSavedCap: false,
      maxOutputTokensPublished: null,
    },
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 4096,
    reasoningRequested: false,
    reasoningStyle: "none",
    reasoningEffort: "low",
    reasoningEffortLevels: ["low", "medium", "high"],
    clampReasoningEffort: clamp,
  });
  assert.equal(request.maxOutputTokens, 20000);
});

test("the request says whether the saved cap is what grounded its ceiling", () => {
  const build = (maxOutputTokensFromSavedCap: boolean, maxOutputTokens: number | null) =>
    buildResearchInferenceRequest({
      checkpoint: "external::p1::some-self-hosted-model",
      external: {
        providerId: "p1",
        providerType: "custom",
        modelId: "some-self-hosted-model",
        maxOutputTokens,
        maxOutputTokensFromSavedCap,
        maxOutputTokensPublished: null,
      },
      temperature: 0.2,
      topP: 0.9,
      maxTokens: 4096,
      reasoningRequested: false,
      reasoningStyle: "none",
      reasoningEffort: "medium",
      reasoningEffortLevels: ["low", "medium", "high"],
      clampReasoningEffort: clamp,
    });

  assert.equal(build(true, 30000).maxOutputTokensFromSavedCap, true);
  assert.equal(build(false, 30000).maxOutputTokensFromSavedCap, false);
  // No ceiling to qualify, so the flag has nothing to say and is left off entirely.
  assert.equal("maxOutputTokensFromSavedCap" in build(true, null), false);
});

test("the published ceiling rides along, unfolded, when the model has one", () => {
  const request = buildResearchInferenceRequest({
    checkpoint: "external::p1::gemini-3.6-flash",
    external: {
      providerId: "p1",
      providerType: "gemini",
      modelId: "gemini-3.6-flash",
      // What the connection actually spends: the override folded into the published cap.
      maxOutputTokens: 8192,
      maxOutputTokensFromSavedCap: false,
      maxOutputTokensPublished: 65536,
    },
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 4096,
    reasoningRequested: false,
    reasoningStyle: "none",
    reasoningEffort: "low",
    reasoningEffortLevels: ["low", "medium", "high"],
    clampReasoningEffort: clamp,
  });

  // The backend needs the pair to tell a capped connection from a 8192-token model.
  assert.equal(request.maxOutputTokens, 8192);
  assert.equal(request.maxOutputTokensPublished, 65536);
});
