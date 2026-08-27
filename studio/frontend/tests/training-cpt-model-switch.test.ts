// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);

const LLAMA_TARGETS = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
];

async function waitForModelDefaults(model: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const state = useTrainingConfigStore.getState();
    if (
      !state.isLoadingModelDefaults &&
      state.modelDefaultsAppliedFor === model
    ) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("model defaults did not settle");
}

after(() => setAuthFetchHandler(null));

test("leaving CPT after a model switch restores the new model targets", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({
    selectedModel: "old/llama",
    modelDefaultsAppliedFor: "old/llama",
    trainingMethod: "cpt",
    targetModules: [...LLAMA_TARGETS, "embed_tokens", "lm_head"],
    trainingMethodProvenance: {
      learningRateManuallySet: false,
      modelAdapterLearningRate: null,
      datasetFormatBeforeCpt: "chatml",
      targetModulesBeforeCpt: [...LLAMA_TARGETS],
    },
  });

  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: "LiquidAI/LFM2-1.2B",
        config: { lora: { target_modules: ["all-linear"] } },
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: false,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
  ]);
});
