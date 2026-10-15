// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { DrafterCapability } from "../src/features/training/types/api.ts";
import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload } =
  await import("../src/features/training/api/mappers.ts");
const { initialTrainingConfigState } =
  await import("../src/features/training/stores/training-config-policy.ts");
const {
  TRAINING_CONFIG_PERSISTENCE_VERSION,
  migrateTrainingConfig,
  partializeTrainingConfig,
} =
  await import("../src/features/training/stores/training-config-persistence.ts");
const { parseYamlConfig, serializeConfigToYaml } =
  await import("../src/features/training/lib/yaml-config.ts");
const { drafterStateFrom } =
  await import("../src/features/training/lib/drafter-config.ts");
const { phaseColors } =
  await import("../src/features/studio/sections/progress-section-lib.ts");

const MODEL = "unsloth/Qwen3-1.7B";

function capability(patch: Partial<DrafterCapability> = {}): DrafterCapability {
  return {
    model_name: MODEL,
    has_native_mtp: false,
    strategy: "eagle3",
    supported: true,
    reason: null,
    vllm: true,
    llama_cpp: true,
    ...patch,
  };
}

function config(patch: Partial<TrainingConfigState> = {}): TrainingConfigState {
  return {
    ...initialTrainingConfigState,
    modelType: "text",
    selectedModel: MODEL,
    datasetSource: "huggingface",
    dataset: "unsloth/test",
    ...patch,
  };
}

const ON = {
  drafterEnabled: true,
  drafterStrategy: "auto",
  drafterDataSource: "regenerate",
  drafterMaxExamples: 1500,
  drafterEpochs: 2,
  drafterLearningRate: 0.0002,
  drafterDraftSteps: 4,
} as const;

test("drafter training is off by default and the payload has no drafter key", () => {
  assert.equal(initialTrainingConfigState.drafterEnabled, false);
  const payload = buildTrainingStartPayload(config(), null);
  assert.equal(Object.hasOwn(payload, "drafter"), false);
});

test("a capability check still pending or for another model does not drop a ticked drafter", () => {
  for (const patch of [
    { ...ON },
    { ...ON, drafterCapability: capability({ model_name: "other/model" }) },
  ]) {
    const payload = buildTrainingStartPayload(config(patch), null);
    assert.equal(Object.hasOwn(payload, "drafter"), true);
  }
});

test("settings the request cannot carry leave the payload exactly as without the feature", () => {
  const off = buildTrainingStartPayload(config(), null);
  const cases: Partial<TrainingConfigState>[] = [
    // Settings edited but the box unticked.
    { ...ON, drafterEnabled: false, drafterCapability: capability() },
    // Ticked, but the backend said this model cannot train one.
    { ...ON, drafterCapability: capability({ supported: false, reason: "x" }) },
    // Run types without a drafter phase.
    { ...ON, drafterCapability: capability(), trainingMethod: "cpt" },
    { ...ON, drafterCapability: capability(), isEmbeddingModel: true },
  ];
  for (const patch of cases) {
    const payload = buildTrainingStartPayload(config(patch), null);
    if (patch.trainingMethod === "cpt" || patch.isEmbeddingModel) {
      assert.equal(Object.hasOwn(payload, "drafter"), false);
    } else {
      assert.deepEqual(payload, off);
    }
  }
  const mlx = buildTrainingStartPayload(
    config({ ...ON, drafterCapability: capability() }),
    null,
    { device_backend: "mlx" } as never,
  );
  assert.equal(Object.hasOwn(mlx, "drafter"), false);
  // Image / audio fine-tunes have no drafter phase; the same models on a text dataset do.
  for (const patch of [
    { isVisionModel: true, isDatasetImage: true },
    { isAudioModel: true, isDatasetAudio: true },
  ]) {
    const media = buildTrainingStartPayload(
      config({ ...ON, drafterCapability: capability(), ...patch }),
      null,
    );
    assert.equal(Object.hasOwn(media, "drafter"), false);
  }
  const text = buildTrainingStartPayload(
    config({
      ...ON,
      drafterCapability: capability(),
      isVisionModel: true,
      isDatasetImage: false,
    }),
    null,
  );
  assert.equal(Object.hasOwn(text, "drafter"), true);
});

test("an enabled drafter on a supported model adds only the drafter block", () => {
  const off = buildTrainingStartPayload(config(), null);
  const on = buildTrainingStartPayload(
    config({ ...ON, drafterCapability: capability() }),
    null,
  );
  const { drafter, ...rest } = on;
  assert.deepEqual(rest, off);
  assert.deepEqual(drafter, {
    enabled: true,
    strategy: "auto",
    data_source: "regenerate",
    max_examples: 1500,
    epochs: 2,
    learning_rate: 0.0002,
    draft_steps: 4,
  });
});

test("half-typed fields are clamped into the backend's bounds", () => {
  const sent = (patch: Partial<TrainingConfigState>) =>
    buildTrainingStartPayload(
      config({ ...ON, drafterCapability: capability(), ...patch }),
      null,
    ).drafter;
  assert.equal(sent({ drafterMaxExamples: 3 })?.max_examples, 16);
  assert.equal(sent({ drafterMaxExamples: 1e9 })?.max_examples, 100000);
  assert.equal(sent({ drafterEpochs: 0 })?.epochs, 0.1);
  assert.equal(sent({ drafterDraftSteps: 2.7 })?.draft_steps, 2);
  assert.equal(sent({ drafterDraftSteps: Number.NaN })?.draft_steps, 3);
  assert.equal(sent({ drafterLearningRate: -1 })?.learning_rate, null);
});

test("persisted state from before drafter training migrates to the feature off", () => {
  assert.equal(TRAINING_CONFIG_PERSISTENCE_VERSION, 23);
  const migrated = migrateTrainingConfig(
    { selectedModel: MODEL, trainingMethod: "lora" },
    22,
  ) as unknown as TrainingConfigState;
  assert.equal(migrated.drafterEnabled, false);
  assert.equal(migrated.drafterStrategy, "auto");
  assert.equal(migrated.drafterMaxExamples, 2000);
  assert.equal(migrated.drafterLearningRate, null);
  const malformed = migrateTrainingConfig(
    { drafterEnabled: "yes", drafterStrategy: "medusa", drafterDraftSteps: 99 },
    22,
  ) as unknown as TrainingConfigState;
  assert.equal(malformed.drafterEnabled, false);
  assert.equal(malformed.drafterStrategy, "auto");
  assert.equal(malformed.drafterDraftSteps, 8);
});

test("the capability answer is never persisted, the settings are", () => {
  const persisted = partializeTrainingConfig({
    ...config({ ...ON, drafterCapability: capability() }),
  } as never);
  assert.equal(Object.hasOwn(persisted, "drafterCapability"), false);
  assert.equal(persisted.drafterEnabled, true);
  assert.equal(persisted.drafterDataSource, "regenerate");
});

test("YAML round trip keeps drafter settings, and leaves files without it unchanged", () => {
  const offYaml = serializeConfigToYaml(config(), false);
  assert.equal(
    Object.hasOwn(parseYamlConfig(offYaml).training ?? {}, "drafter"),
    false,
  );
  const onYaml = serializeConfigToYaml(config({ ...ON }), false);
  const restored = drafterStateFrom(parseYamlConfig(onYaml).training?.drafter);
  assert.deepEqual(restored, { ...ON });
});

test("the drafter phase has a progress colour (labels are typed exhaustively)", () => {
  assert.ok(phaseColors.training_drafter);
});
