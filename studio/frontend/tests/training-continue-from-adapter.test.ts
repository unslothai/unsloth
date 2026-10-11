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

function serveAdapterDetails(overrides: Record<string, unknown>): void {
  setAuthFetchHandler(() =>
    Promise.resolve(
      Response.json({
        id: ADAPTER,
        config: {},
        is_vision: false,
        is_embedding: false,
        is_audio: false,
        audio_type_known: true,
        is_lora: true,
        model_type: "text",
        model_size_bytes: null,
        max_position_embeddings: 32768,
        ...overrides,
      }),
    ),
  );
}

after(() => setAuthFetchHandler(null));

test("a vision adapter picked for LoRA keeps its vision capability", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "lora" });
  serveAdapterDetails({ is_vision: true, model_type: "vision" });

  useTrainingConfigStore.getState().selectTrainingModel(ADAPTER, "vision");
  await waitForModelDefaults(ADAPTER);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelFormat, "adapter");
  assert.equal(state.isVisionModel, true);
  assert.equal(state.modelType, "vision");
  assert.equal(state.maxPositionEmbeddings, 32768);
});

test("an adapter picked under full finetuning keeps its modality for a later LoRA switch", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "full" });
  serveAdapterDetails({ is_vision: true, model_type: "vision" });

  useTrainingConfigStore.getState().selectTrainingModel(ADAPTER, "vision");
  await waitForModelDefaults(ADAPTER);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelFormat, "adapter");
  assert.equal(state.isVisionModel, true);
  assert.equal(state.modelType, "vision");
  assert.equal(state.maxPositionEmbeddings, 32768);
});

test("a decision adapter is typed as a decision model, not continued", async () => {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingMethod: "qlora" });
  serveAdapterDetails({
    is_vision: false,
    is_decision: true,
    decision_layout: "clef",
    model_type: "decision",
  });

  useTrainingConfigStore.getState().selectTrainingModel(ADAPTER, "text");
  await waitForModelDefaults(ADAPTER);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.modelFormat, "adapter");
  assert.equal(state.modelType, "decision");
  assert.equal(state.decisionLayout, null);
});
