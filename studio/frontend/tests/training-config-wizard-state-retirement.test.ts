// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `currentStep` was persisted, so deleting the wizard without a migration leaves
// it in every existing install and partializeTrainingConfig keeps writing it back.
// These pin the retirement: the orphan goes, everything else survives, and a blob
// from a newer build still hydrates.

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

/** A v20 blob as an install that ran the onboarding wizard actually stored it. */
function wizardEraBlob(): Record<string, unknown> {
  return {
    currentStep: 2,
    projectName: "customer-support-lora",
    selectedModel: "unsloth/gemma-3-270m",
    trainingMethod: "qlora",
    maxSteps: 123,
    learningRate: 0.0002,
    datasetSource: "upload",
    uploadedFile: "/home/u/train.jsonl",
    modelType: "text",
    datasetStreaming: false,
  };
}

test("the retired wizard step is dropped from a pre-retirement blob", () => {
  const state = migrateTrainingConfig(wizardEraBlob(), 20) as unknown as Record<
    string,
    unknown
  >;

  assert.equal(
    Object.hasOwn(state, "currentStep"),
    false,
    "currentStep must not survive the migration",
  );
});

test("dropping the wizard step preserves every user-authored value", () => {
  const state = migrateTrainingConfig(wizardEraBlob(), 20) as unknown as Record<
    string,
    unknown
  >;

  assert.equal(state.projectName, "customer-support-lora");
  assert.equal(state.selectedModel, "unsloth/gemma-3-270m");
  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.maxSteps, 123);
  assert.equal(state.learningRate, 0.0002);
  assert.equal(state.uploadedFile, "/home/u/train.jsonl");
});

test("the version was bumped, so the migration actually runs for old installs", () => {
  assert.ok(
    TRAINING_CONFIG_PERSISTENCE_VERSION >= 21,
    "zustand skips migrate() when the stored version matches, so retiring a persisted key requires a bump",
  );
});

test("every historical version migrates without throwing", () => {
  for (let version = 0; version <= TRAINING_CONFIG_PERSISTENCE_VERSION; version++) {
    const state = migrateTrainingConfig(
      wizardEraBlob(),
      version,
    ) as unknown as Record<string, unknown>;
    assert.equal(
      Object.hasOwn(state, "currentStep"),
      version >= 21,
      `currentStep handling is wrong for version ${version}`,
    );
  }
});

test("a blob written by a newer build still hydrates (forwards compatible)", () => {
  const future: Record<string, unknown> = {
    ...wizardEraBlob(),
    someFieldFromTheFuture: { nested: true },
  };
  delete future.currentStep;

  const merged = mergeTrainingConfig(
    migrateTrainingConfig(future, TRAINING_CONFIG_PERSISTENCE_VERSION + 5),
    { trainingMethod: "lora", trainOnCompletions: false } as never,
  ) as unknown as Record<string, unknown>;

  assert.equal(merged.projectName, "customer-support-lora");
  assert.equal(merged.trainingMethod, "qlora");
});

test("a partial or empty blob does not throw", () => {
  for (const blob of [{}, { projectName: "only-this" }, { currentStep: 4 }]) {
    const migrated = migrateTrainingConfig(
      { ...blob },
      1,
    ) as unknown as Record<string, unknown>;
    assert.equal(Object.hasOwn(migrated, "currentStep"), false);
    mergeTrainingConfig(migrated, {
      trainingMethod: "lora",
      trainOnCompletions: false,
    } as never);
  }
});

test("the retired key is not re-persisted after a migrated load", () => {
  const migrated = migrateTrainingConfig(wizardEraBlob(), 20);
  const repersisted = partializeTrainingConfig(migrated as never) as Record<
    string,
    unknown
  >;

  assert.equal(Object.hasOwn(repersisted, "currentStep"), false);
});
