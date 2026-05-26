// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { cacheOptionsForTraining } from "../src/features/models/lib/training-cache-options.ts";
import {
  cachedInventoryPathMatchesSelection,
  cacheReferenceMatchesSelection,
} from "../src/features/training/lib/cache-reference.ts";
import {
  inferTrainingModelTypeFromCapabilityKeys,
  inferTrainingModelTypeFromFlags,
  inferTrainingModelTypeFromMetadata,
  resolvePickerInferredModelType,
} from "../src/features/training/lib/model-type-inference.ts";
import { isMissingLocalDatasetCacheError } from "../src/features/training/lib/local-cache-errors.ts";
import { migratePersistedResourceSelections } from "../src/features/training/lib/persisted-resource-selection.ts";
import {
  completeResourceSet,
  resolveTrainingResourceNotice,
} from "../src/features/training/lib/resource-availability.ts";
import { preserveTrainingDraftFromModelDefaults } from "../src/features/training/lib/training-draft-preservation.ts";
import {
  createPersistedStateKeys,
  pickPersistedTrainingConfigState,
  TRAINING_TRANSIENT_STATE_KEYS,
} from "../src/features/training/lib/training-config-persistence.ts";

const emptySet = new Set<string>();

function notice(
  overrides: Partial<Parameters<typeof resolveTrainingResourceNotice>[0]>,
) {
  return resolveTrainingResourceNotice({
    kind: "model",
    id: "Org/Model",
    isLocal: false,
    knownCached: false,
    localPath: null,
    completeSet: emptySet,
    partialSet: emptySet,
    ...overrides,
  });
}

test("resource notices stay empty without a selection", () => {
  assert.equal(notice({ id: null }), null);
});

test("resource notices flag remote Hugging Face resources", () => {
  assert.deepEqual(notice({})?.status, "download");
});

test("resource notices flag partial resources before remote downloads", () => {
  assert.deepEqual(
    notice({ partialSet: new Set(["org/model"]) })?.status,
    "partial",
  );
});

test("resource notices ignore complete cached resources", () => {
  assert.equal(
    notice({
      completeSet: completeResourceSet([{ repoId: "Org/Model" }]),
      partialSet: new Set(["org/model"]),
    }),
    null,
  );
});

test("resource notices ignore local paths and uploaded datasets", () => {
  assert.equal(notice({ isLocal: true }), null);
  assert.equal(notice({ knownCached: false, localPath: "/models/base" }), null);
  assert.equal(
    notice({
      kind: "dataset",
      id: "/tmp/upload.jsonl",
      isLocal: true,
    }),
    null,
  );
});

test("training cache options only preserve complete local references", () => {
  assert.deepEqual(
    cacheOptionsForTraining({ cacheState: "partial", localPath: "/cache" }),
    { knownCached: false, localPath: null },
  );
  assert.deepEqual(
    cacheOptionsForTraining({ cacheState: "remote", localPath: null }),
    { knownCached: false, localPath: null },
  );
  assert.deepEqual(
    cacheOptionsForTraining({ cacheState: "cached", localPath: "/cache" }),
    { knownCached: true, localPath: "/cache" },
  );
  assert.deepEqual(
    cacheOptionsForTraining({ cacheState: "local", localPath: "/models/base" }),
    { knownCached: false, localPath: "/models/base" },
  );
});

test("cache reference guard preserves non-cache and changed local selections", () => {
  assert.equal(
    cacheReferenceMatchesSelection({
      currentId: "Org/Model",
      expectedId: "Org/Model",
      knownCached: false,
      currentLocalPath: "/models/local",
      expectedLocalPath: "/cache/model",
    }),
    false,
  );
  assert.equal(
    cacheReferenceMatchesSelection({
      currentId: "Org/Model",
      expectedId: "Org/Model",
      knownCached: true,
      currentLocalPath: "/models/local",
      expectedLocalPath: "/cache/model",
    }),
    false,
  );
  assert.equal(
    cacheReferenceMatchesSelection({
      currentId: "Org/Model",
      expectedId: "Org/Model",
      knownCached: true,
      currentLocalPath: "/cache/model",
      expectedLocalPath: "/cache/model",
    }),
    true,
  );
  assert.equal(
    cacheReferenceMatchesSelection({
      currentId: "Org/Model",
      expectedId: "Org/Model",
      knownCached: true,
      currentLocalPath: null,
      expectedLocalPath: "/cache/model",
    }),
    false,
  );
  assert.equal(
    cacheReferenceMatchesSelection({
      currentId: "Org/Model",
      expectedId: "Org/Model",
      knownCached: true,
      currentLocalPath: "C:\\Users\\me\\cache\\model\\",
      expectedLocalPath: "c:/users/me/cache/model",
    }),
    true,
  );
});

test("cached inventory guard requires matching cache paths when known", () => {
  assert.equal(cachedInventoryPathMatchesSelection("/cache/a", null), true);
  assert.equal(cachedInventoryPathMatchesSelection(null, "/cache/a"), false);
  assert.equal(
    cachedInventoryPathMatchesSelection("/cache/a/", "/cache/a"),
    true,
  );
  assert.equal(
    cachedInventoryPathMatchesSelection("/cache/other", "/cache/a"),
    false,
  );
});

test("v15 resource migration preserves selections and backfills resource fields", () => {
  const state: Record<string, unknown> = {
    currentStep: 4,
    modelType: "vision",
    selectedModel: "Org/Model",
    datasetSource: "upload",
    datasetFormat: "sharegpt",
    datasetFormatBeforeCpt: "alpaca",
    datasetFormatAutoForcedByCpt: true,
    dataset: "Org/Data",
    datasetKnownCached: true,
    datasetLocalPath: "/cache/data",
    datasetSubset: "default",
    datasetSplit: "train",
    datasetEvalSplit: "validation",
    datasetManualMapping: { text: "prompt" },
    datasetSystemPrompt: "system",
    datasetLabelMapping: { label: { yes: "yes" } },
    datasetAdvisorNotification: "review",
    datasetSliceStart: "10",
    datasetSliceEnd: "20",
    uploadedFile: "/tmp/data.jsonl",
    uploadedEvalFile: "/tmp/eval.jsonl",
    trainingMethod: "qlora",
    learningRate: 0.0002,
  };

  migratePersistedResourceSelections(state);

  assert.equal(state.currentStep, 4);
  assert.equal(state.modelType, "vision");
  assert.equal(state.selectedModel, "Org/Model");
  assert.equal(state.modelKnownCached, false);
  assert.equal(state.modelLocalPath, null);
  assert.equal(state.modelFormat, null);
  assert.equal(state.datasetSource, "upload");
  assert.equal(state.datasetFormat, "sharegpt");
  assert.equal(state.datasetFormatBeforeCpt, "alpaca");
  assert.equal(state.datasetFormatAutoForcedByCpt, true);
  assert.equal(state.dataset, "Org/Data");
  assert.equal(state.datasetKnownCached, true);
  assert.equal(state.datasetLocalPath, "/cache/data");
  assert.equal(state.datasetSubset, "default");
  assert.equal(state.datasetSplit, "train");
  assert.equal(state.datasetEvalSplit, "validation");
  assert.deepEqual(state.datasetManualMapping, { text: "prompt" });
  assert.equal(state.datasetSystemPrompt, "system");
  assert.deepEqual(state.datasetLabelMapping, { label: { yes: "yes" } });
  assert.equal(state.datasetAdvisorNotification, "review");
  assert.equal(state.datasetSliceStart, "10");
  assert.equal(state.datasetSliceEnd, "20");
  assert.equal(state.uploadedFile, "/tmp/data.jsonl");
  assert.equal(state.uploadedEvalFile, "/tmp/eval.jsonl");
  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.learningRate, 0.0002);
});

test("training model type inference uses one shared precedence", () => {
  assert.equal(
    inferTrainingModelTypeFromCapabilityKeys(["vision", "audio", "embedding"]),
    "embeddings",
  );
  assert.equal(
    inferTrainingModelTypeFromCapabilityKeys(["vision", "audio"]),
    "audio",
  );
  assert.equal(inferTrainingModelTypeFromCapabilityKeys(["vision"]), "vision");
  assert.equal(inferTrainingModelTypeFromCapabilityKeys([]), "text");
  assert.equal(
    inferTrainingModelTypeFromFlags({
      isEmbedding: false,
      isAudio: true,
      isVision: true,
    }),
    "audio",
  );
  assert.equal(
    inferTrainingModelTypeFromMetadata({
      tags: ["image-text-to-text"],
      pipelineTag: null,
      identifiers: ["org/model"],
    }),
    "vision",
  );
  assert.equal(
    inferTrainingModelTypeFromMetadata({
      tags: [],
      pipelineTag: "text-generation",
      identifiers: ["org/qwen3-embedding"],
    }),
    "embeddings",
  );
  assert.equal(resolvePickerInferredModelType("vision", "text"), "vision");
  assert.equal(resolvePickerInferredModelType("audio", "text"), "audio");
  assert.equal(
    resolvePickerInferredModelType("embeddings", "text"),
    "embeddings",
  );
  assert.equal(resolvePickerInferredModelType("vision", "audio"), "audio");
  assert.equal(resolvePickerInferredModelType("text", "vision"), "vision");
});

test("missing local dataset cache errors are detected consistently", () => {
  assert.equal(
    isMissingLocalDatasetCacheError(
      new Error("Dataset is not available in the local cache"),
    ),
    true,
  );
  assert.equal(
    isMissingLocalDatasetCacheError("Local cache entry was not found"),
    true,
  );
  assert.equal(isMissingLocalDatasetCacheError("Network unavailable"), false);
});

test("Hub model handoff preservation keeps draft fields and refreshes trust remote code", () => {
  const patch = preserveTrainingDraftFromModelDefaults({
    contextLength: 32768,
    learningRate: 0.00001,
    loraRank: 64,
    trainOnCompletions: false,
    targetModules: ["q_proj"],
    trustRemoteCode: true,
  });

  assert.deepEqual(patch, { trustRemoteCode: true });
});

test("training persisted state derives from state shape and excludes transients", () => {
  const state = {
    selectedModel: "Org/Model",
    trainOnCompletions: true,
    trainOnCompletionsManuallySet: true,
    learningRateManuallySet: true,
    trainingMethodManuallySet: true,
    yamlLearningRate: 0.0002,
    datasetFormatBeforeCpt: "alpaca",
    datasetFormatAutoForcedByCpt: true,
    isCheckingVision: true,
    isDatasetImage: true,
    modelDefaultsAppliedKey: "key",
    maxPositionEmbeddings: 8192,
  } as const;
  const persistedKeys = createPersistedStateKeys(
    state,
    TRAINING_TRANSIENT_STATE_KEYS,
  );

  assert.equal(persistedKeys.includes("trainOnCompletions"), true);
  assert.equal(persistedKeys.includes("trainOnCompletionsManuallySet"), true);
  assert.equal(persistedKeys.includes("learningRateManuallySet"), true);
  assert.equal(persistedKeys.includes("trainingMethodManuallySet"), true);
  assert.equal(persistedKeys.includes("yamlLearningRate"), true);
  assert.equal(persistedKeys.includes("datasetFormatBeforeCpt"), true);
  assert.equal(persistedKeys.includes("datasetFormatAutoForcedByCpt"), true);
  assert.equal(persistedKeys.includes("isCheckingVision"), false);
  assert.equal(persistedKeys.includes("isDatasetImage"), false);
  assert.equal(persistedKeys.includes("modelDefaultsAppliedKey"), false);
  assert.equal(persistedKeys.includes("maxPositionEmbeddings"), false);

  assert.deepEqual(
    pickPersistedTrainingConfigState(state, persistedKeys),
    {
      selectedModel: "Org/Model",
      trainOnCompletions: true,
      trainOnCompletionsManuallySet: true,
      learningRateManuallySet: true,
      trainingMethodManuallySet: true,
      yamlLearningRate: 0.0002,
      datasetFormatBeforeCpt: "alpaca",
      datasetFormatAutoForcedByCpt: true,
    },
  );
});
