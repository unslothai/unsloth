// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "vite";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const server = await createServer({
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const { reconcileTrainingGpuSelection, validateRefreshedTrainingGpuSelection } =
  await server.ssrLoadModule(
    "/src/features/training/lib/training-gpu-selection.ts",
  );
const { validateTrainingConfig } = await server.ssrLoadModule(
  "/src/features/training/lib/validation.ts",
);
test.after(() => server.close());

const validTrainingConfig = {
  selectedModel: "org/model",
  modelKnownCached: false,
  modelLocalPath: null,
  modelFormat: null,
  learningRate: 0.0002,
  embeddingLearningRate: null,
  datasetSource: "huggingface",
  dataset: "org/dataset",
  datasetSplit: "train",
  manualDatasetOptionsValid: true,
  uploadedFile: null,
  s3Config: null,
  modelType: "text",
  isVisionModel: false,
  isEmbeddingModel: false,
  isAudioModel: false,
  isDatasetAudio: false,
  loraVariant: "rslora",
  trainingMethod: "qlora",
} as const;

test("discovery does not clear remembered training devices", () => {
  assert.deepEqual(reconcileTrainingGpuSelection("ddp", [2, 5], null), {
    parallelismMode: "ddp",
    selectedGpuIds: [2, 5],
  });
});

test("stale single and multi-device selections fall back to automatic", () => {
  for (const mode of ["single", "ddp", "model_parallel"] as const) {
    const ids = mode === "single" ? [5] : [2, 5];
    assert.deepEqual(reconcileTrainingGpuSelection(mode, ids, [2]), {
      parallelismMode: "auto",
      selectedGpuIds: null,
    });
  }
});

test("multi-device selection retains enough surviving physical devices", () => {
  for (const mode of ["ddp", "model_parallel"] as const) {
    assert.deepEqual(reconcileTrainingGpuSelection(mode, [2, 5, 8], [2, 8]), {
      parallelismMode: mode,
      selectedGpuIds: [2, 8],
    });
  }
});

test("settled empty inventory clears a remembered selection", () => {
  assert.deepEqual(reconcileTrainingGpuSelection("single", [2], []), {
    parallelismMode: "auto",
    selectedGpuIds: null,
  });
});

test("reconciliation does not interrupt an incomplete manual selection", () => {
  for (const ids of [[], [2]]) {
    assert.deepEqual(reconcileTrainingGpuSelection("ddp", ids, [2, 8]), {
      parallelismMode: "ddp",
      selectedGpuIds: ids,
    });
  }
});

test("reconciliation preserves a valid incomplete selection with duplicate IDs", () => {
  assert.deepEqual(reconcileTrainingGpuSelection("ddp", [2, 2], [2, 8]), {
    parallelismMode: "ddp",
    selectedGpuIds: [2, 2],
  });
});

test("automatic placement never retains explicit IDs", () => {
  assert.deepEqual(reconcileTrainingGpuSelection("auto", [2], [2]), {
    parallelismMode: "auto",
    selectedGpuIds: null,
  });
});

test("refreshed membership validates and returns the same current config snapshot", async () => {
  const config = { parallelismMode: "ddp", selectedGpuIds: [2, 5] };
  let validationConfig: typeof config | null = null;
  let validationGpuIds: number[] | null = null;
  const result = await validateRefreshedTrainingGpuSelection({
    refresh: async () => [2, 5],
    abortIfInputsChanged: () => false,
    getConfig: () => config,
    validate: (current: typeof config, available: number[] | null) => {
      validationConfig = current;
      validationGpuIds = available;
      return { ok: true };
    },
  });

  assert.deepEqual(result, { kind: "valid", availableGpuIds: [2, 5], config });
  assert.equal(validationConfig, config);
  assert.deepEqual(validationGpuIds, [2, 5]);
});

test("refreshed selection aborts before validation when inputs changed", async () => {
  let resolveRefresh!: (ids: number[] | null) => void;
  let inputsChanged = false;
  let validated = false;
  const pending = validateRefreshedTrainingGpuSelection({
    refresh: () =>
      new Promise<number[] | null>((resolve) => {
        resolveRefresh = resolve;
      }),
    abortIfInputsChanged: () => inputsChanged,
    getConfig: () => ({ selectedGpuIds: [2, 5] }),
    validate: () => {
      validated = true;
      return { ok: true };
    },
  });
  inputsChanged = true;
  resolveRefresh([2, 5]);
  const result = await pending;

  assert.deepEqual(result, { kind: "inputs-changed" });
  assert.equal(validated, false);
});

test("missing refreshed devices reject the current explicit selection", async () => {
  const result = await validateRefreshedTrainingGpuSelection({
    refresh: async () => [2],
    abortIfInputsChanged: () => false,
    getConfig: () => ({ parallelismMode: "ddp", selectedGpuIds: [2, 5] }),
    validate: (
      _config: { parallelismMode: string; selectedGpuIds: number[] },
      available: number[] | null,
    ) =>
      available?.includes(5)
        ? { ok: true }
        : { ok: false, errorKey: "gpuSelectionUnavailable" },
  });

  assert.deepEqual(result, {
    kind: "invalid",
    errorKey: "gpuSelectionUnavailable",
  });
});

test("an unresolved refresh keeps automatic placement valid but rejects explicit IDs", async () => {
  const automaticConfig = {
    ...validTrainingConfig,
    parallelismMode: "auto",
    selectedGpuIds: null,
  };
  const explicitConfig = {
    ...validTrainingConfig,
    parallelismMode: "ddp",
    selectedGpuIds: [2, 5],
  };
  const autoResult = await validateRefreshedTrainingGpuSelection({
    refresh: async () => null,
    abortIfInputsChanged: () => false,
    getConfig: () => automaticConfig,
    validate: (config: typeof automaticConfig, available: number[] | null) =>
      validateTrainingConfig(config, undefined, true, available),
  });
  const explicitResult = await validateRefreshedTrainingGpuSelection({
    refresh: async () => null,
    abortIfInputsChanged: () => false,
    getConfig: () => explicitConfig,
    validate: (config: typeof explicitConfig, available: number[] | null) =>
      validateTrainingConfig(config, undefined, true, available),
  });

  assert.equal(autoResult.kind, "valid");
  assert.deepEqual(explicitResult, {
    kind: "invalid",
    errorKey: "studio.training.validation.gpuSelectionUnavailable",
  });
});
