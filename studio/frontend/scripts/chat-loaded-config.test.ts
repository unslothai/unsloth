// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { PerModelConfig } from "../src/features/chat/model-config/per-model-config.ts";
import {
  loadedRuntimeConfigMatches,
  normalizeRuntimePerModelLoadConfig,
} from "../src/features/chat/model-runtime/per-model-load-config.ts";
import { DEFAULT_INFERENCE_PARAMS } from "../src/features/chat/types/runtime.ts";

const defaultConfig: PerModelConfig = {
  kvCacheDtype: null,
  speculativeType: "auto",
  specDraftNMax: null,
  customContextLength: null,
  chatTemplateOverride: null,
  trustRemoteCode: false,
};

function runtimeState(overrides: {
  checkpoint?: string;
  ggufVariant?: string | null;
  loadedKvCacheDtype?: string | null;
  loadedSpeculativeType?: string | null;
  loadedSpecDraftNMax?: number | null;
  loadedCustomContextLength?: number | null;
  loadedChatTemplateOverride?: string | null;
  trustRemoteCode?: boolean;
}) {
  return {
    params: {
      ...DEFAULT_INFERENCE_PARAMS,
      checkpoint: overrides.checkpoint ?? "Org/Model",
      trustRemoteCode: overrides.trustRemoteCode ?? false,
    },
    activeGgufVariant: overrides.ggufVariant ?? null,
    loadedKvCacheDtype: overrides.loadedKvCacheDtype ?? null,
    loadedSpeculativeType: overrides.loadedSpeculativeType ?? "auto",
    loadedSpecDraftNMax: overrides.loadedSpecDraftNMax ?? null,
    loadedCustomContextLength: overrides.loadedCustomContextLength ?? null,
    loadedChatTemplateOverride: overrides.loadedChatTemplateOverride ?? null,
  };
}

test("loaded runtime config matches identical per-model config", () => {
  assert.equal(
    loadedRuntimeConfigMatches({
      state: runtimeState({}),
      modelId: "org/model",
      config: defaultConfig,
    }),
    true,
  );
});

test("loaded runtime config uses default config when omitted", () => {
  assert.equal(
    loadedRuntimeConfigMatches({
      state: runtimeState({}),
      modelId: "Org/Model",
    }),
    true,
  );
});

test("loaded runtime config compares loaded context, not staged context", () => {
  assert.equal(
    loadedRuntimeConfigMatches({
      state: runtimeState({ loadedCustomContextLength: 8192 }),
      modelId: "Org/Model",
      config: { ...defaultConfig, customContextLength: 8192 },
    }),
    true,
  );
  assert.equal(
    loadedRuntimeConfigMatches({
      state: runtimeState({ loadedCustomContextLength: 8192 }),
      modelId: "Org/Model",
      config: { ...defaultConfig, customContextLength: 4096 },
    }),
    false,
  );
});

test("loaded runtime config ignores staged context length from runtime state", () => {
  const state = {
    ...runtimeState({ loadedCustomContextLength: 8192 }),
    customContextLength: 4096,
  };

  assert.equal(
    loadedRuntimeConfigMatches({
      state,
      modelId: "Org/Model",
      config: { ...defaultConfig, customContextLength: 8192 },
    }),
    true,
  );
});

test("loaded runtime config detects load-setting changes", () => {
  assert.equal(
    loadedRuntimeConfigMatches({
      state: runtimeState({ loadedKvCacheDtype: "q8_0" }),
      modelId: "Org/Model",
      config: { ...defaultConfig, kvCacheDtype: "q4_0" },
    }),
    false,
  );
});

test("speculative draft default matches backend-selected draft", () => {
  assert.equal(
    loadedRuntimeConfigMatches({
      state: runtimeState({
        loadedSpeculativeType: "mtp",
        loadedSpecDraftNMax: 3,
      }),
      modelId: "Org/Model",
      config: { ...defaultConfig, speculativeType: "mtp" },
    }),
    true,
  );
});

test("ngram-simple speculative mode round-trips as an exact mode", () => {
  assert.equal(
    loadedRuntimeConfigMatches({
      state: runtimeState({ loadedSpeculativeType: "ngram-simple" }),
      modelId: "Org/Model",
      config: { ...defaultConfig, speculativeType: "ngram-simple" },
    }),
    true,
  );
});

test("runtime config normalization trims blank chat templates", () => {
  assert.deepEqual(
    normalizeRuntimePerModelLoadConfig({
      ...defaultConfig,
      chatTemplateOverride: "   ",
    }),
    {
      kvCacheDtype: null,
      speculativeType: "auto",
      specDraftNMax: null,
      customContextLength: null,
      chatTemplateOverride: null,
      trustRemoteCode: false,
    },
  );
});
