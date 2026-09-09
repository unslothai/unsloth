// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);
const { normalizeSavedChatSettings, sanitizeChatSettings } = await import(
  "../src/features/chat/utils/chat-settings-storage.ts"
);
const { normalizeSavedThreadScopedSettings, sanitizeThreadScopedSettings } =
  await import("../src/features/chat/utils/thread-scoped-settings.ts");
const { getReplayedParams } = await import(
  "../src/features/chat/lib/per-model-params.ts"
);
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);
const {
  applyPresetForProvider,
  BUILTIN_PRESETS,
  isSamePresetConfig,
  toPresetParams,
  mergeBackendRecommendedInference,
} = await import("../src/features/chat/presets/preset-policy.ts");

test("original legacy values become Custom independently at every saved scope", () => {
  for (const minP of [0, 0.01, 0.2]) {
    const saved = normalizeSavedChatSettings({
      inferenceParams: { minP },
      inferenceParamsByModel: { model: { minP } },
      customPresets: [{ name: "Saved", params: { minP } }],
    });
    const expected = { minP, minPMode: "custom" };
    assert.deepEqual(saved.inferenceParams, expected);
    assert.deepEqual(saved.inferenceParamsByModel?.model, expected);
    assert.deepEqual(saved.customPresets?.[0]?.params, expected);
    assert.deepEqual(normalizeSavedThreadScopedSettings({ minP }), expected);
    const replayed = getReplayedParams(
      true,
      { model: { minP } },
      DEFAULT_INFERENCE_PARAMS,
      "model",
      true,
    );
    assert.equal(replayed.minP, minP);
    assert.equal(replayed.minPMode, "custom");
  }
});

test("empty records inherit and pure outgoing numeric patches do not invent intent", () => {
  assert.deepEqual(normalizeSavedThreadScopedSettings({}), {});
  assert.deepEqual(
    normalizeSavedChatSettings({
      inferenceParamsByModel: { model: { temperature: 0.4 } },
    }).inferenceParamsByModel?.model,
    { temperature: 0.4 },
  );
  assert.deepEqual(
    sanitizeChatSettings({
      inferenceParams: { minP: 0.3 },
      inferenceParamsByModel: { model: { minP: 0 } },
      customPresets: [{ name: "Saved", params: { minP: 0.2 } }],
    }),
    {
      inferenceParams: { minP: 0.3 },
      inferenceParamsByModel: { model: { minP: 0 } },
      customPresets: [{ name: "Saved", params: { minP: 0.2 } }],
    },
  );
  assert.deepEqual(sanitizeThreadScopedSettings({ minP: 0.3 }), { minP: 0.3 });
});

test("explicit modes survive every persisted scope and invalid modes are rejected", () => {
  for (const minPMode of ["server-default", "custom"] as const) {
    const pair = { minPMode, minP: 0.2 };
    const saved = {
      inferenceParams: pair,
      inferenceParamsByModel: { model: pair },
      customPresets: [{ name: "Saved", params: pair }],
    };
    assert.deepEqual(
      normalizeSavedChatSettings(sanitizeChatSettings(saved)),
      saved,
    );
    assert.deepEqual(normalizeSavedThreadScopedSettings(pair), pair);
  }
  assert.deepEqual(
    normalizeSavedThreadScopedSettings({ minPMode: "invalid", minP: 0 }),
    { minPMode: "custom", minP: 0 },
  );
  assert.deepEqual(
    normalizeSavedChatSettings({ inferenceParams: { minPMode: "invalid" } }),
    {},
  );
});

test("preset mode participates in save, equality, reset and provider-aware application", () => {
  const custom = {
    ...DEFAULT_INFERENCE_PARAMS,
    minPMode: "custom" as const,
    minP: 0.2,
  };
  const server = { ...custom, minPMode: "server-default" as const };
  assert.equal(isSamePresetConfig(custom, server), false);
  assert.deepEqual(toPresetParams(server).minPMode, "server-default");
  for (const params of [custom, server]) {
    assert.equal(
      applyPresetForProvider(
        DEFAULT_INFERENCE_PARAMS,
        { name: "Saved", params },
        "vllm",
      ).minPMode,
      params.minPMode,
    );
  }
  assert.equal(
    applyPresetForProvider(custom, BUILTIN_PRESETS[0]!, "vllm").minPMode,
    "server-default",
  );
  for (const provider of [null, "openai", "ollama"]) {
    const applied = applyPresetForProvider(
      custom,
      BUILTIN_PRESETS[0]!,
      provider,
    );
    assert.equal(applied.minPMode, "custom");
    assert.equal(applied.minP, DEFAULT_INFERENCE_PARAMS.minP);
  }
});

test("model recommendations update the retained number without changing either mode", () => {
  for (const minPMode of ["server-default", "custom"] as const) {
    const next = mergeBackendRecommendedInference({
      current: { ...DEFAULT_INFERENCE_PARAMS, minPMode },
      response: { inference: { min_p: 0.4 } },
      modelId: "model",
      presetSource: "builtin-default",
      loadedContextLength: null,
    });
    assert.equal(next.minP, 0.4);
    assert.equal(next.minPMode, minPMode);
  }
});
