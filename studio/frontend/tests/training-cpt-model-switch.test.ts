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

function deferLfmDefaults(): () => void {
  let resolveModelConfig!: (response: Response) => void;
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        resolveModelConfig = resolve;
      }),
  );
  return () =>
    resolveModelConfig(
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
    );
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

test("model targets apply when CPT is selected during the defaults request", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
});

test("an explicit target edit still wins during the defaults request", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore
    .getState()
    .setTargetModules(["q_proj", "embed_tokens", "lm_head"]);
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "q_proj",
    "embed_tokens",
    "lm_head",
  ]);
});

test("a target edit before entering CPT still wins", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTargetModules(["q_proj"]);
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    ...LLAMA_TARGETS,
    "embed_tokens",
    "lm_head",
  ]);
  assert.deepEqual(
    useTrainingConfigStore.getState().trainingMethodProvenance
      .targetModulesBeforeCpt,
    ["q_proj"],
  );
  useTrainingConfigStore.getState().setTrainingMethod("qlora");
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "q_proj",
  ]);
});

test("an unrelated edit does not block the model targets", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().setBatchSize(3);
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.equal(useTrainingConfigStore.getState().batchSize, 3);
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
});

test("an unrelated edit does not block targets when CPT was already active", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setBatchSize(3);
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.equal(useTrainingConfigStore.getState().batchSize, 3);
  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "all-linear",
    "embed_tokens",
    "lm_head",
  ]);
});

test("targets imported during the defaults request still win", async () => {
  useTrainingConfigStore.getState().reset();
  const resolveModelConfig = deferLfmDefaults();

  useTrainingConfigStore
    .getState()
    .selectTrainingModel("LiquidAI/LFM2-1.2B", "text");
  useTrainingConfigStore.getState().setTrainingMethod("cpt");
  useTrainingConfigStore.getState().applyConfigPatch({
    lora: { target_modules: ["q_proj"] },
  });
  resolveModelConfig();
  await waitForModelDefaults("LiquidAI/LFM2-1.2B");

  assert.deepEqual(useTrainingConfigStore.getState().targetModules, [
    "q_proj",
  ]);
});
