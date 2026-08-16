// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const { mergeTrainingConfig } = await import(
  "../src/features/training/stores/training-config-persistence.ts"
);
const { buildTrainingMethodPatch } = await import(
  "../src/features/training/stores/training-method-transition.ts"
);

test("rehydrated CPT provenance survives the next method switch", () => {
  const serialized = JSON.stringify({
    trainingMethod: "cpt",
    learningRate: 0.000031,
    datasetFormat: "raw",
    trainingMethodProvenance: {
      learningRateManuallySet: true,
      modelAdapterLearningRate: 0.00001,
      datasetFormatBeforeCpt: "sharegpt",
    },
  });
  const rehydrated = mergeTrainingConfig(
    JSON.parse(serialized),
    initialTrainingConfigState as never,
  );
  const state = {
    ...rehydrated,
    ...buildTrainingMethodPatch(rehydrated, "qlora"),
  };

  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.learningRate, 0.000031);
  assert.equal(state.datasetFormat, "sharegpt");
  assert.equal(state.trainingMethodProvenance.datasetFormatBeforeCpt, null);
});

test("rehydrated model learning rate is restored for adapter methods", () => {
  const serialized = JSON.stringify({
    trainingMethod: "full",
    learningRate: 0.00002,
    datasetFormat: "auto",
    trainingMethodProvenance: {
      learningRateManuallySet: false,
      modelAdapterLearningRate: 0.00001,
      datasetFormatBeforeCpt: null,
    },
  });
  const rehydrated = mergeTrainingConfig(
    JSON.parse(serialized),
    initialTrainingConfigState as never,
  );
  const state = {
    ...rehydrated,
    ...buildTrainingMethodPatch(rehydrated, "lora"),
  };

  assert.equal(state.learningRate, 0.00001);
});
