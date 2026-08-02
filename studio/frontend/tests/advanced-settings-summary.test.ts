// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { CPT_TARGET_MODULES, DEFAULT_HYPERPARAMS } = await import(
  "../src/config/training.ts"
);
const { countNonDefaultAdvancedSettings } = await import(
  "../src/features/studio/wizard/advanced-settings-summary.ts"
);

const defaultState = {
  trainingMethod: "qlora" as const,
  optimizerType: DEFAULT_HYPERPARAMS.optimizerType,
  lrSchedulerType: DEFAULT_HYPERPARAMS.lrSchedulerType,
  weightDecay: DEFAULT_HYPERPARAMS.weightDecay,
  warmupSteps: DEFAULT_HYPERPARAMS.warmupSteps,
  saveSteps: DEFAULT_HYPERPARAMS.saveSteps,
  evalSteps: DEFAULT_HYPERPARAMS.evalSteps,
  randomSeed: DEFAULT_HYPERPARAMS.randomSeed,
  packing: DEFAULT_HYPERPARAMS.packing,
  trainOnCompletions: DEFAULT_HYPERPARAMS.trainOnCompletions,
  gradientCheckpointing: DEFAULT_HYPERPARAMS.gradientCheckpointing,
  visionImageSize: DEFAULT_HYPERPARAMS.visionImageSize,
  finetuneVisionLayers: DEFAULT_HYPERPARAMS.finetuneVisionLayers,
  finetuneLanguageLayers: DEFAULT_HYPERPARAMS.finetuneLanguageLayers,
  finetuneAttentionModules: DEFAULT_HYPERPARAMS.finetuneAttentionModules,
  finetuneMLPModules: DEFAULT_HYPERPARAMS.finetuneMLPModules,
  loraRank: DEFAULT_HYPERPARAMS.loraRank,
  loraAlpha: DEFAULT_HYPERPARAMS.loraAlpha,
  loraDropout: DEFAULT_HYPERPARAMS.loraDropout,
  loraVariant: DEFAULT_HYPERPARAMS.loraVariant,
  targetModules: DEFAULT_HYPERPARAMS.targetModules,
};

test("advanced settings summary counts submitted non-default values", () => {
  assert.equal(countNonDefaultAdvancedSettings(defaultState), 0);
  assert.equal(
    countNonDefaultAdvancedSettings({
      ...defaultState,
      loraRank: 32,
      optimizerType: "adamw_torch",
    }),
    2,
  );
});

test("advanced settings summary uses method-aware LoRA defaults", () => {
  assert.equal(
    countNonDefaultAdvancedSettings({
      ...defaultState,
      trainingMethod: "full",
      loraRank: 64,
    }),
    0,
  );
  assert.equal(
    countNonDefaultAdvancedSettings({
      ...defaultState,
      trainingMethod: "cpt",
      loraRank: 128,
      loraAlpha: 32,
      loraVariant: "rslora",
      targetModules: CPT_TARGET_MODULES,
    }),
    0,
  );
});

test("advanced settings summary uses applied model defaults as its baseline", () => {
  const modelDefaults = {
    loraRank: 32,
    loraAlpha: 32,
    saveSteps: 30,
    trainOnCompletions: true,
    targetModules: [...DEFAULT_HYPERPARAMS.targetModules, "shared_mlp"],
  };
  const modelState = { ...defaultState, ...modelDefaults };

  assert.equal(countNonDefaultAdvancedSettings(modelState, modelDefaults), 0);
  assert.equal(
    countNonDefaultAdvancedSettings(
      { ...modelState, saveSteps: 60 },
      modelDefaults,
    ),
    1,
  );
});
