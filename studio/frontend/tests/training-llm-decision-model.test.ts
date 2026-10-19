// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test, { after } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const yaml = await import("js-yaml");
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);

type TrainingState = ReturnType<typeof useTrainingConfigStore.getState>;

const LLM = "unsloth/Qwen3.5-4B";
const LAYA = "convaiinnovations/laya";

const LLM_DECISION_YAML = yaml.load(
  readFileSync(
    new URL(
      "../../backend/assets/configs/model_defaults/decision/llm_decision_defaults.yaml",
      import.meta.url,
    ),
    "utf8",
  ),
) as { training: Record<string, number>; lora: Record<string, number> };

const BASE = {
  id: LLM,
  is_vision: true,
  is_embedding: false,
  is_audio: false,
  audio_type_known: true,
  is_lora: false,
  model_size_bytes: 9_000_000_000,
  max_position_embeddings: 262144,
};

// What /api/models/config answers for the LLM, with and without as_decision.
const LLM_CONFIG = { ...BASE, config: { training: {} }, model_type: "vision" };
const LLM_DECISION_CONFIG = {
  ...BASE,
  config: LLM_DECISION_YAML,
  is_decision: true,
  decision_layout: "llm",
  decision_checkpoints: null,
  model_type: "decision",
};
const LAYA_CONFIG = {
  ...BASE,
  id: LAYA,
  is_vision: false,
  config: { training: { learning_rate: 8e-4 }, lora: { lora_r: 64 } },
  is_decision: true,
  decision_checkpoints: [
    { name: "laya-multilingual", subfolder: "multilingual", description: "" },
  ],
  model_type: "decision",
};

const CLEF = "Cloudflare/clef-flash";
const CLEF_CONFIG = {
  ...BASE,
  id: CLEF,
  is_vision: false,
  config: { training: { learning_rate: 2e-4, max_seq_length: 4096 } },
  is_decision: true,
  decision_layout: "clef",
  decision_checkpoints: null,
  model_type: "decision",
};

function serve(asDecision: object = LLM_DECISION_CONFIG): string[] {
  const requested: string[] = [];
  setAuthFetchHandler((input) => {
    requested.push(input);
    if (!input.startsWith("/api/models/config/")) {
      return Response.json({});
    }
    if (input.includes("laya")) {
      return Response.json(LAYA_CONFIG);
    }
    if (input.includes("clef")) {
      return Response.json(CLEF_CONFIG);
    }
    return Response.json(
      input.includes("as_decision=true") ? asDecision : LLM_CONFIG,
    );
  });
  return requested;
}

async function waitFor(done: (state: TrainingState) => boolean): Promise<void> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const state = useTrainingConfigStore.getState();
    if (!state.isLoadingModelDefaults && done(state)) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("the training config did not settle");
}

async function selectLlm(): Promise<string[]> {
  const requested = serve();
  useTrainingConfigStore.getState().selectTrainingModel(LLM, "text");
  await waitFor((state) => state.modelDefaultsAppliedFor === LLM);
  return requested;
}

after(() => setAuthFetchHandler(null));

test("an LLM trained as a decision model gets the decision recipe and keeps QLoRA", async () => {
  useTrainingConfigStore.getState().reset();
  const requested = await selectLlm();
  assert.equal(useTrainingConfigStore.getState().modelType, "vision");
  useTrainingConfigStore.setState({ trainingMethod: "qlora" });

  useTrainingConfigStore.getState().setTrainAsDecision(true);
  await waitFor((state) => state.modelType === "decision");

  assert.ok(requested.some((url) => url.includes("as_decision=true")));
  const state = useTrainingConfigStore.getState();
  assert.equal(state.decisionLayout, "llm");
  assert.equal(state.decisionCheckpoints, null);
  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.learningRate, LLM_DECISION_YAML.training.learning_rate);
  assert.equal(state.epochs, LLM_DECISION_YAML.training.num_epochs);
  assert.equal(state.loraRank, LLM_DECISION_YAML.lora.lora_r);
  assert.equal(state.contextLength, LLM_DECISION_YAML.training.max_seq_length);
  assert.equal(state.datasetStreaming, false);

  const payload = buildTrainingStartPayload(state, null);
  assert.equal(payload.is_decision, true);
  assert.equal(payload.training_type, "LoRA/QLoRA");
  assert.equal(payload.load_in_4bit, true);
  assert.equal(
    payload.max_seq_length,
    LLM_DECISION_YAML.training.max_seq_length,
  );
  assert.equal(payload.model_subfolder, null);
  assert.equal(payload.use_dora, false);
  assert.equal(payload.train_on_completions, false);
});

test("switching back to a language model restores the method it had", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLlm();
  useTrainingConfigStore.setState({ trainingMethod: "full" });

  useTrainingConfigStore.getState().setTrainAsDecision(true);
  await waitFor((state) => state.modelType === "decision");
  assert.equal(useTrainingConfigStore.getState().trainingMethod, "qlora");

  useTrainingConfigStore.getState().setTrainAsDecision(false);
  await waitFor((state) => state.modelType === "vision");

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainingMethod, "full");
  assert.equal(state.decisionLayout, null);
  assert.equal(state.settingsBeforeDecision, null);
  assert.equal(buildTrainingStartPayload(state, null).is_decision, false);
});

test("a model the backend cannot train as a decision model turns the choice off", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLlm();
  serve({
    ...LLM_CONFIG,
    is_vision: false,
    is_audio: true,
    model_type: "audio",
  });

  useTrainingConfigStore.getState().setTrainAsDecision(true);
  await waitFor((state) => !state.trainAsDecision);

  assert.equal(useTrainingConfigStore.getState().modelType, "audio");
});

test("a failed decision config request turns the choice back off", async () => {
  useTrainingConfigStore.getState().reset();
  await selectLlm();
  setAuthFetchHandler((input) =>
    input.includes("as_decision=true")
      ? new Response("down", { status: 500 })
      : Response.json({}),
  );

  useTrainingConfigStore.getState().setTrainAsDecision(true);
  await waitFor((state) => state.modelDefaultsError !== null);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trainAsDecision, false);
  assert.equal(buildTrainingStartPayload(state, null).is_decision, false);
});

test("Laya stays a 16-bit LoRA run whatever the choice", async () => {
  useTrainingConfigStore.getState().reset();
  serve();
  useTrainingConfigStore.setState({ trainAsDecision: true });
  useTrainingConfigStore.getState().selectTrainingModel(LAYA, "text");
  await waitFor((state) => state.modelDefaultsAppliedFor === LAYA);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.decisionLayout, "laya");
  assert.equal(state.trainingMethod, "lora");
  const payload = buildTrainingStartPayload(state, null);
  assert.equal(payload.load_in_4bit, false);
  assert.equal(payload.max_seq_length, 1024);
  assert.equal(payload.model_subfolder, "multilingual");
});

test("Clef keeps its own layout and QLoRA whatever the choice", async () => {
  useTrainingConfigStore.getState().reset();
  serve();
  useTrainingConfigStore.setState({ trainAsDecision: true });
  useTrainingConfigStore.getState().selectTrainingModel(CLEF, "text");
  await waitFor((state) => state.modelDefaultsAppliedFor === CLEF);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.decisionLayout, "clef");
  assert.equal(state.trainingMethod, "qlora");
  const payload = buildTrainingStartPayload(state, null);
  assert.equal(payload.is_decision, true);
  assert.equal(payload.load_in_4bit, true);
  assert.equal(payload.max_seq_length, 4096);
});
