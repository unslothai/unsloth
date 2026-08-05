// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  createHfBrowseDatasetSelection,
  datasetSelectionStreamingPatch,
  datasetSourceInvariantPatch,
  resolveDeferredTrainOnCompletionsDefault,
} = await import("../src/features/training/stores/training-config-policy.ts");

test("selecting an on-device Hub dataset disables streaming", () => {
  const deviceOptions = {
    knownCached: true,
    localPath: "/cache/datasets--org--dataset",
    preferLocalCache: true,
  };
  const cachedSelection = createHfBrowseDatasetSelection(
    "org/dataset",
    deviceOptions,
  );

  assert.deepEqual(
    datasetSelectionStreamingPatch(cachedSelection, deviceOptions),
    { datasetStreaming: false },
  );
  assert.deepEqual(datasetSelectionStreamingPatch(cachedSelection), {});
  assert.deepEqual(
    datasetSelectionStreamingPatch(
      createHfBrowseDatasetSelection("org/remote-dataset"),
    ),
    {},
  );
});

test("streaming is constrained to Hugging Face dataset sources", () => {
  assert.deepEqual(
    datasetSourceInvariantPatch({
      datasetSource: "huggingface",
      datasetStreaming: true,
    }),
    {},
  );
  for (const datasetSource of ["upload", "s3"] as const) {
    assert.deepEqual(
      datasetSourceInvariantPatch({
        datasetSource,
        datasetStreaming: true,
      }),
      { datasetStreaming: false },
    );
    assert.deepEqual(
      datasetSourceInvariantPatch({
        datasetSource,
        datasetStreaming: false,
      }),
      {},
    );
  }
});

test("resolves deferred completion defaults without violating training constraints", () => {
  const base = {
    currentValue: false,
    datasetFormat: "chatml" as const,
    datasetStreaming: false,
    isEmbeddingModel: false,
    modelDefault: true,
    trainingMethod: "qlora" as const,
  };

  assert.equal(resolveDeferredTrainOnCompletionsDefault(base), true);
  assert.equal(
    resolveDeferredTrainOnCompletionsDefault({
      ...base,
      currentValue: true,
      modelDefault: false,
    }),
    false,
  );
  assert.equal(
    resolveDeferredTrainOnCompletionsDefault({
      ...base,
      currentValue: true,
      modelDefault: undefined,
    }),
    true,
  );
  assert.equal(
    resolveDeferredTrainOnCompletionsDefault({
      ...base,
      datasetStreaming: true,
    }),
    false,
  );
  assert.equal(
    resolveDeferredTrainOnCompletionsDefault({
      ...base,
      datasetFormat: "raw",
    }),
    false,
  );
  assert.equal(
    resolveDeferredTrainOnCompletionsDefault({
      ...base,
      trainingMethod: "cpt",
    }),
    false,
  );
  assert.equal(
    resolveDeferredTrainOnCompletionsDefault({
      ...base,
      isEmbeddingModel: true,
    }),
    false,
  );
});
