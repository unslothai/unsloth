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
    trainOnCompletionsDefaultPendingFor: null,
    trainOnCompletions: true,
    trainingMethodProvenance: {
      learningRateManuallySet: true,
      modelAdapterLearningRate: 0.00001,
      datasetFormatBeforeCpt: "sharegpt",
    },
    isLoadingModelDefaults: true,
    wandbToken: "secret-token",
    setLearningRate: () => undefined,
  } as never);

  assert.equal(persisted.modelDefaultsAppliedFor, "org/model");
  assert.deepEqual(persisted.advancedSettingsBaseline, {
    loraRank: 32,
    saveSteps: 25,
  });
  assert.equal(persisted.trainOnCompletions, true);
  assert.deepEqual(persisted.trainingMethodProvenance, {
    learningRateManuallySet: true,
    modelAdapterLearningRate: 0.00001,
    datasetFormatBeforeCpt: "sharegpt",
  });
  assert.equal("isLoadingModelDefaults" in persisted, false);
  assert.equal("trainOnCompletionsDefaultPendingFor" in persisted, false);
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

  assert.equal(TRAINING_CONFIG_PERSISTENCE_VERSION, 21);
  assert.equal(migrated.learningRate, 0.000031);
  assert.equal(migrated.loraRank, 48);
  assert.equal(migrated.modelDefaultsAppliedFor, "org/model");
  assert.equal(migrated.advancedSettingsBaseline, null);
  assert.deepEqual(migrated.trainingMethodProvenance, {
    learningRateManuallySet: true,
    modelAdapterLearningRate: null,
    datasetFormatBeforeCpt: null,
    targetModulesBeforeCpt: null,
  });
  assert.equal("wandbToken" in migrated, false);
});

test("migration keeps method-default learning rates automatic", () => {
  const migrated = migrateTrainingConfig(
    { trainingMethod: "full", learningRate: 0.00002 },
    18,
  );

  assert.equal(
    migrated.trainingMethodProvenance.learningRateManuallySet,
    false,
  );
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

test("merge restores completion training from legacy model defaults metadata", () => {
  const legacy = mergeTrainingConfig(
    {
      selectedModel: "org/model",
      modelDefaultsAppliedFor: "org/model",
      advancedSettingsBaseline: { trainOnCompletions: true },
    },
    { trainingMethod: "qlora", trainOnCompletions: false } as never,
  );
  const explicit = mergeTrainingConfig(
    {
      selectedModel: "org/model",
      modelDefaultsAppliedFor: "org/model",
      advancedSettingsBaseline: { trainOnCompletions: true },
      trainOnCompletions: false,
    },
    { trainingMethod: "qlora", trainOnCompletions: true } as never,
  );

  assert.equal(legacy.trainOnCompletions, true);
  assert.equal(legacy.trainOnCompletionsDefaultPendingFor, null);
  assert.equal(explicit.trainOnCompletions, false);
  assert.equal(explicit.trainOnCompletionsDefaultPendingFor, null);
});

test("defers an unavailable legacy completion default without persisting a placeholder", () => {
  const migrated = migrateTrainingConfig(
    {
      selectedModel: "org/model",
      learningRate: 0.000031,
      loraRank: 48,
    },
    16,
  );
  const merged = mergeTrainingConfig(migrated, {
    trainingMethod: "qlora",
    trainOnCompletions: false,
  } as never);
  const persisted = partializeTrainingConfig(merged);

  assert.equal(merged.modelDefaultsAppliedFor, "org/model");
  assert.equal(merged.advancedSettingsBaseline, null);
  assert.equal(merged.trainOnCompletionsDefaultPendingFor, "org/model");
  assert.equal("trainOnCompletions" in persisted, false);
  assert.equal("trainOnCompletionsDefaultPendingFor" in persisted, false);
  assert.equal(persisted.learningRate, 0.000031);
  assert.equal(persisted.loraRank, 48);
});

test("does not defer an explicitly persisted completion setting", () => {
  const merged = mergeTrainingConfig(
    {
      selectedModel: "org/model",
      modelDefaultsAppliedFor: "org/model",
      advancedSettingsBaseline: null,
      trainOnCompletions: false,
    },
    { trainingMethod: "qlora", trainOnCompletions: true } as never,
  );

  assert.equal(merged.trainOnCompletions, false);
  assert.equal(merged.trainOnCompletionsDefaultPendingFor, null);
});

test("persistence normalizes streaming for non-Hub dataset sources", () => {
  for (const datasetSource of ["upload", "s3"] as const) {
    const browseDatasetSelection = {
      dataset: "org/remembered",
      knownCached: true,
      localPath: "/cache/datasets--org--remembered",
      source: "huggingface" as const,
    };
    const current = {
      browseDatasetSelection,
      datasetSource: "huggingface",
      datasetStreaming: false,
      evalSteps: 0,
      selectedModel: null,
      trainingMethod: "qlora",
      trainOnCompletions: false,
      wandbToken: "",
    };
    const persisted = partializeTrainingConfig({
      ...current,
      datasetSource,
      datasetStreaming: true,
      evalSteps: 0.1,
    } as never);
    const merged = mergeTrainingConfig(
      {
        browseDatasetSelection,
        datasetSource,
        datasetStreaming: true,
        evalSteps: 0.1,
      },
      current as never,
    );

    assert.equal(persisted.datasetStreaming, false);
    assert.equal(persisted.evalSteps, 0.1);
    assert.equal(merged.datasetStreaming, false);
    assert.equal(merged.evalSteps, 0.1);
    assert.deepEqual(merged.browseDatasetSelection, browseDatasetSelection);
  }
});

test("persistence preserves valid Hub streaming", () => {
  const current = {
    datasetSource: "huggingface",
    datasetStreaming: false,
    selectedModel: null,
    trainingMethod: "qlora",
    trainOnCompletions: false,
    wandbToken: "",
  };
  const merged = mergeTrainingConfig(
    { datasetSource: "huggingface", datasetStreaming: true },
    current as never,
  );

  assert.equal(merged.datasetStreaming, true);
});
