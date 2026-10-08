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
const { useTrainingConfigStore } =
  await import("../src/features/training/stores/training-config-store.ts");

const MODEL_LR = 3e-4;

function config(id: string, embedding: boolean) {
  return {
    id,
    config: { training: { learning_rate: MODEL_LR, max_steps: 60 } },
    is_vision: false,
    is_embedding: embedding,
    is_audio: false,
    audio_type_known: true,
    is_lora: false,
    model_type: embedding ? "embeddings" : "text",
    model_size_bytes: 0,
    max_position_embeddings: 32768,
  };
}

async function selectWithGrpoStored(id: string, embedding: boolean) {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({ trainingObjective: "grpo" });
  setAuthFetchHandler((input) =>
    Response.json(
      input.startsWith("/api/models/config/") ? config(id, embedding) : {},
    ),
  );
  useTrainingConfigStore.getState().selectTrainingModel(id, "text");
  for (let i = 0; i < 100; i += 1) {
    const s = useTrainingConfigStore.getState();
    if (!s.isLoadingModelDefaults && s.modelDefaultsAppliedFor === id) {
      return s;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("model defaults did not settle");
}

after(() => setAuthFetchHandler(null));

test("an embedding model keeps its own learning rate while GRPO is stored", async () => {
  const s = await selectWithGrpoStored(
    "sentence-transformers/all-MiniLM-L6-v2",
    true,
  );
  assert.equal(s.learningRate, MODEL_LR);
});

test("a text model with GRPO selected gets the GRPO learning rate", async () => {
  const s = await selectWithGrpoStored("unsloth/Qwen3-0.6B", false);
  assert.notEqual(s.learningRate, MODEL_LR);
});
