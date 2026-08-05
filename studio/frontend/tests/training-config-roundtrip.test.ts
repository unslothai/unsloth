// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Exercise the Save/Load path through the real store action.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { BackendModelConfig } from "../src/features/training/api/models-api.ts";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const yaml = await import("js-yaml");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { parseYamlConfig, serializeConfigToYaml } = await import(
  "../src/features/training/lib/yaml-config.ts"
);
const { mapBackendModelConfigToTrainingPatch } = await import(
  "../src/features/training/lib/model-defaults.ts"
);

// This shipped config uses a boolean gradient_checkpointing value.
const TUNED_MODEL_CONFIG = new URL(
  "../../backend/assets/configs/model_defaults/llama/unsloth_Llama-3.2-1B-Instruct.yaml",
  import.meta.url,
);

/** Seed the store as model selection does. */
function seedTunedModelDefaults(): void {
  const config = yaml.load(
    readFileSync(TUNED_MODEL_CONFIG, "utf8"),
  ) as BackendModelConfig;
  useTrainingConfigStore.setState(mapBackendModelConfigToTrainingPatch(config));
}

/**
 * Snapshot every non-action store value. userEditRevision is edit bookkeeping every
 * user edit bumps, not a config value, so it stays out of the comparison.
 */
function snapshot(): Record<string, unknown> {
  const state = useTrainingConfigStore.getState() as unknown as Record<
    string,
    unknown
  >;
  return Object.fromEntries(
    Object.entries(state).filter(
      ([key, value]) => typeof value !== "function" && key !== "userEditRevision",
    ),
  );
}

function keysChangedBy(action: () => void): string[] {
  const before = snapshot();
  action();
  const after = snapshot();
  return Object.keys(after).filter(
    (key) => !Object.is(after[key], before[key]),
  );
}

function importConfig(text: string): void {
  useTrainingConfigStore.getState().applyConfigPatch(parseYamlConfig(text));
}

test("a partial import patches only the keys the file names", () => {
  seedTunedModelDefaults();

  const changed = keysChangedBy(() =>
    importConfig("training:\n  max_seq_length: 4096\n"),
  );

  assert.deepEqual(
    changed,
    ["contextLength"],
    "a file naming one key must not reset the selected model's other tuned values",
  );
  assert.equal(useTrainingConfigStore.getState().contextLength, 4096);
});

test("a tuned model recipe survives an unrelated import", () => {
  seedTunedModelDefaults();
  const tuned = snapshot();

  // Ensure sparse import preserves unrelated seeded values.
  assert.equal(tuned.learningRate, 2e-5);
  assert.equal(tuned.batchSize, 1);
  assert.equal(tuned.optimizerType, "adamw_torch");
  assert.equal(tuned.lrSchedulerType, "cosine");
  assert.equal(tuned.trainOnCompletions, true);
  assert.equal(tuned.loraAlpha, 16);

  importConfig("lora:\n  lora_r: 64\n");

  const after = useTrainingConfigStore.getState();
  assert.equal(after.loraRank, 64, "the imported key applies");
  assert.equal(after.learningRate, 2e-5);
  assert.equal(after.batchSize, 1);
  assert.equal(after.optimizerType, "adamw_torch");
  assert.equal(after.lrSchedulerType, "cosine");
  assert.equal(after.trainOnCompletions, true);
  assert.equal(after.loraAlpha, 16);
});

test("gradient_checkpointing is read from a YAML boolean as well as a string", () => {
  seedTunedModelDefaults();
  assert.equal(
    useTrainingConfigStore.getState().gradientCheckpointing,
    "true",
    "the shipped config says gradient_checkpointing: true, unquoted",
  );

  importConfig("training:\n  gradient_checkpointing: false\n");
  assert.equal(useTrainingConfigStore.getState().gradientCheckpointing, "none");

  importConfig("training:\n  gradient_checkpointing: unsloth\n");
  assert.equal(
    useTrainingConfigStore.getState().gradientCheckpointing,
    "unsloth",
  );
});

test("a blank number is treated as absent, not as zero", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({ epochs: 5, warmupSteps: 7 });

  importConfig('training:\n  num_epochs: ""\n  warmup_steps: "   "\n');

  const after = useTrainingConfigStore.getState();
  assert.equal(after.epochs, 5);
  assert.equal(after.warmupSteps, 7);

  importConfig("training:\n  num_epochs: 0\n");
  assert.equal(
    useTrainingConfigStore.getState().epochs,
    0,
    "a real 0 still applies; only the blank is ignored",
  );
});

test("saving and reloading a config keeps the logging settings", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({
    enableWandb: true,
    wandbProject: "my-project",
    enableTensorboard: true,
    tensorboardDir: "my-runs",
    logFrequency: 25,
  });

  const saved = serializeConfigToYaml(useTrainingConfigStore.getState(), false);

  useTrainingConfigStore.setState({
    enableWandb: false,
    wandbProject: "",
    enableTensorboard: false,
    tensorboardDir: "",
    logFrequency: 1,
  });
  importConfig(saved);

  const after = useTrainingConfigStore.getState();
  assert.equal(after.enableWandb, true);
  assert.equal(after.wandbProject, "my-project");
  assert.equal(after.enableTensorboard, true);
  assert.equal(after.tensorboardDir, "my-runs");
  assert.equal(after.logFrequency, 25);
});

test("saving and reloading a config keeps the embedding learning rate", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({ embeddingLearningRate: 3e-5 });

  const saved = serializeConfigToYaml(useTrainingConfigStore.getState(), false);

  useTrainingConfigStore.setState({ embeddingLearningRate: null });
  importConfig(saved);
  assert.equal(useTrainingConfigStore.getState().embeddingLearningRate, 3e-5);

  // Preserve null as "derive it", distinct from an absent key.
  useTrainingConfigStore.setState({ embeddingLearningRate: null });
  const clearedSave = serializeConfigToYaml(
    useTrainingConfigStore.getState(),
    false,
  );
  useTrainingConfigStore.setState({ embeddingLearningRate: 9e-5 });
  importConfig(clearedSave);
  assert.equal(useTrainingConfigStore.getState().embeddingLearningRate, null);
});

test("a file with no embedding learning rate leaves the current one alone", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({ embeddingLearningRate: 4e-5 });

  importConfig("training:\n  max_seq_length: 4096\n");
  assert.equal(useTrainingConfigStore.getState().embeddingLearningRate, 4e-5);
});
