// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  TRAINING_CONFIG_PERSISTENCE_VERSION,
  mergeTrainingConfig,
  migrateTrainingConfig,
  partializeTrainingConfig,
} = await import(
  "../src/features/training/stores/training-config-persistence.ts"
);

test("persists the applied model defaults identity and summary baseline", () => {
  const persisted = partializeTrainingConfig({
    selectedModel: "org/model",
    modelDefaultsAppliedFor: "org/model",
    advancedSettingsBaseline: { loraRank: 32, saveSteps: 25 },
    isLoadingModelDefaults: true,
    wandbToken: "secret-token",
    setLearningRate: () => undefined,
  } as never);

  assert.equal(persisted.modelDefaultsAppliedFor, "org/model");
  assert.deepEqual(persisted.advancedSettingsBaseline, {
    loraRank: 32,
    saveSteps: 25,
  });
  assert.equal("isLoadingModelDefaults" in persisted, false);
  assert.equal("wandbToken" in persisted, false);
  assert.equal("setLearningRate" in persisted, false);
});

test("migration preserves tuned values while protecting them from model defaults", () => {
  const migrated = migrateTrainingConfig(
    {
      selectedModel: "org/model",
      learningRate: 0.000031,
      loraRank: 48,
      wandbToken: "legacy-secret-token",
    },
    16,
  );

  assert.equal(TRAINING_CONFIG_PERSISTENCE_VERSION, 18);
  assert.equal(migrated.learningRate, 0.000031);
  assert.equal(migrated.loraRank, 48);
  assert.equal(migrated.modelDefaultsAppliedFor, "org/model");
  assert.equal(migrated.advancedSettingsBaseline, null);
  assert.equal("wandbToken" in migrated, false);
});

test("merge never restores a persisted W&B token", () => {
  const merged = mergeTrainingConfig({ wandbToken: "persisted-secret-token" }, {
    trainingMethod: "qlora",
    wandbToken: "",
  } as never);

  assert.equal(merged.wandbToken, "");
});

test("merge rejects defaults metadata for a different selected model", () => {
  const matching = mergeTrainingConfig(
    {
      selectedModel: "org/current",
      modelDefaultsAppliedFor: "org/current",
      advancedSettingsBaseline: { loraRank: 32 },
    },
    { trainingMethod: "qlora" } as never,
  );
  const merged = mergeTrainingConfig(
    {
      selectedModel: "org/current",
      modelDefaultsAppliedFor: "org/stale",
      advancedSettingsBaseline: { loraRank: 64 },
    },
    { trainingMethod: "qlora" } as never,
  );

  assert.equal(matching.modelDefaultsAppliedFor, "org/current");
  assert.deepEqual(matching.advancedSettingsBaseline, { loraRank: 32 });
  assert.equal(merged.modelDefaultsAppliedFor, null);
  assert.equal(merged.advancedSettingsBaseline, null);
});
