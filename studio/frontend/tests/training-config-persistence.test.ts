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
const { buildTrainingMethodPatch } = await import(
  "../src/features/training/stores/training-method-transition.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
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

  assert.equal(TRAINING_CONFIG_PERSISTENCE_VERSION, 22);
  assert.equal(migrated.learningRate, 0.000031);
  assert.equal(migrated.loraRank, 48);
  assert.equal(migrated.modelDefaultsAppliedFor, "org/model");
  assert.equal(migrated.advancedSettingsBaseline, null);
  assert.deepEqual(migrated.trainingMethodProvenance, {
    learningRateManuallySet: true,
    modelAdapterLearningRate: null,
    datasetFormatBeforeCpt: null,
    targetModulesBeforeCpt: null,
    loraRankBeforeCpt: null,
    loraAlphaBeforeCpt: null,
    loraVariantBeforeCpt: null,
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

test("a recovered CPT session restores its LoRA params on the way out", () => {
  const persisted = {
    selectedModel: "org/model",
    modelDefaultsAppliedFor: "org/model",
    trainingMethod: "cpt",
    loraRank: 128,
    loraAlpha: 32,
    loraVariant: "rslora",
    advancedSettingsBaseline: {
      loraRank: 8,
      loraAlpha: 8,
      loraVariant: "lora",
    },
    trainingMethodProvenance: {
      learningRateManuallySet: false,
      modelAdapterLearningRate: null,
      datasetFormatBeforeCpt: null,
      targetModulesBeforeCpt: null,
    },
  };
  const merged = mergeTrainingConfig(
    migrateTrainingConfig(persisted, 21),
    initialTrainingConfigState as never,
  );

  assert.equal(merged.trainingMethodProvenance.loraRankBeforeCpt, 8);
  const restored = { ...merged, ...buildTrainingMethodPatch(merged, "qlora") };
  assert.equal(restored.loraRank, 8);
  assert.equal(restored.loraAlpha, 8);
  assert.equal(restored.loraVariant, "lora");
});

test("a session persisted inside CPT recovers its pre-CPT LoRA params", () => {
  const migrated = migrateTrainingConfig(
    {
      selectedModel: "org/model",
      modelDefaultsAppliedFor: "org/model",
      trainingMethod: "cpt",
      advancedSettingsBaseline: {
        loraRank: 8,
        loraAlpha: 8,
        loraVariant: "lora",
      },
      trainingMethodProvenance: {
        learningRateManuallySet: false,
        modelAdapterLearningRate: null,
        datasetFormatBeforeCpt: null,
        targetModulesBeforeCpt: null,
      },
    },
    21,
  );

  assert.equal(migrated.trainingMethodProvenance.loraRankBeforeCpt, 8);
  assert.equal(migrated.trainingMethodProvenance.loraAlphaBeforeCpt, 8);
  assert.equal(migrated.trainingMethodProvenance.loraVariantBeforeCpt, "lora");
});

test("a baseline captured inside CPT is not mistaken for pre-CPT params", () => {
  const migrated = migrateTrainingConfig(
    {
      selectedModel: "org/model",
      modelDefaultsAppliedFor: "org/model",
      trainingMethod: "cpt",
      advancedSettingsBaseline: {
        loraRank: 128,
        loraAlpha: 32,
        loraVariant: "rslora",
      },
      trainingMethodProvenance: {
        learningRateManuallySet: false,
        modelAdapterLearningRate: null,
        datasetFormatBeforeCpt: null,
        targetModulesBeforeCpt: null,
      },
    },
    21,
  );

  assert.equal(migrated.trainingMethodProvenance.loraRankBeforeCpt, undefined);
});

test("an empty model identifier is not treated as a baseline match", () => {
  const persisted = {
    selectedModel: "",
    modelDefaultsAppliedFor: "",
    trainingMethod: "cpt",
    advancedSettingsBaseline: {
      loraRank: 8,
      loraAlpha: 8,
      loraVariant: "lora",
    },
    trainingMethodProvenance: {
      learningRateManuallySet: false,
      modelAdapterLearningRate: null,
      datasetFormatBeforeCpt: null,
      targetModulesBeforeCpt: null,
    },
  };
  const migrated = migrateTrainingConfig(persisted, 21);

  // The merge drops it, so the migration must not have copied it first.
  assert.equal(migrated.trainingMethodProvenance.loraRankBeforeCpt, undefined);
  const merged = mergeTrainingConfig(
    migrated,
    initialTrainingConfigState as never,
  );
  assert.equal(merged.advancedSettingsBaseline, null);
  assert.equal(merged.trainingMethodProvenance.loraRankBeforeCpt, null);
});

test("recovery does not overwrite a pre-CPT value the record already has", () => {
  const migrated = migrateTrainingConfig(
    {
      selectedModel: "org/model",
      modelDefaultsAppliedFor: "org/model",
      trainingMethod: "cpt",
      advancedSettingsBaseline: {
        loraRank: 8,
        loraAlpha: 8,
        loraVariant: "lora",
      },
      trainingMethodProvenance: {
        learningRateManuallySet: false,
        modelAdapterLearningRate: null,
        datasetFormatBeforeCpt: null,
        targetModulesBeforeCpt: null,
        loraRankBeforeCpt: 48,
        loraAlphaBeforeCpt: 0,
        loraVariantBeforeCpt: "loftq",
      },
    },
    21,
  );

  assert.equal(migrated.trainingMethodProvenance.loraRankBeforeCpt, 48);
  assert.equal(migrated.trainingMethodProvenance.loraAlphaBeforeCpt, 8);
  assert.equal(migrated.trainingMethodProvenance.loraVariantBeforeCpt, "loftq");
});

test("a baseline left over from another model is not used as pre-CPT params", () => {
  const migrated = migrateTrainingConfig(
    {
      selectedModel: "org/other-model",
      modelDefaultsAppliedFor: "org/model",
      trainingMethod: "cpt",
      advancedSettingsBaseline: {
        loraRank: 8,
        loraAlpha: 8,
        loraVariant: "lora",
      },
      trainingMethodProvenance: {
        learningRateManuallySet: false,
        modelAdapterLearningRate: null,
        datasetFormatBeforeCpt: null,
        targetModulesBeforeCpt: null,
      },
    },
    21,
  );

  assert.equal(migrated.trainingMethodProvenance.loraRankBeforeCpt, undefined);
});
