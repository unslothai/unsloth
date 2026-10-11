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

const ADAPTER = "/outputs/qwen-vl-run/checkpoint-30";

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

function serveVisionAdapterDetails(): void {
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: ADAPTER,
        config: {},
        is_vision: true,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: true,
        model_type: "vision",
        model_size_bytes: null,
        max_position_embeddings: 32768,
      }),
    ),
  );
}

after(() => setAuthFetchHandler(null));

test("a vision adapter picked for LoRA keeps its vision capability", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "lora" });
  serveVisionAdapterDetails();

  useTrainingConfigStore.getState().selectTrainingModel(ADAPTER, "vision");
  await waitForModelDefaults(ADAPTER);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelFormat, "adapter");
  assert.equal(state.isVisionModel, true);
  assert.equal(state.modelType, "vision");
  assert.equal(state.maxPositionEmbeddings, 32768);
});

test("an adapter picked for full finetuning is still refused", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "full" });
  serveVisionAdapterDetails();

  useTrainingConfigStore.getState().selectTrainingModel(ADAPTER, "vision");
  await waitForModelDefaults(ADAPTER);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelFormat, "adapter");
  assert.equal(state.isVisionModel, false);
  assert.equal(state.modelType, null);
});
