// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";
registerStoreStubResolver();
installLocalStorageFake();
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useTrainingConfigStore: store } =
  await import("../src/features/training/stores/training-config-store.ts");
const { buildTrainingStartPayload } =
  await import("../src/features/training/api/mappers.ts");

function modelResponse(decision = false) {
  return Response.json({
    id: "org/model",
    model_type: decision ? "decision" : "text",
    is_vision: false,
    is_audio: false,
    is_embedding: false,
    is_lora: false,
    decision_layout: decision ? "llm" : null,
    config: {
      training: { learning_rate: 0.9, batch_size: 99 },
      lora: { lora_r: 8, target_modules: ["q_proj"] },
    },
  });
}
async function settle() {
  for (let i = 0; i < 30 && store.getState().isLoadingModelDefaults; i++) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(store.getState().isLoadingModelDefaults, false);
}

test("duplicate restores settings and mappings without secrets or unrelated draft fields", async () => {
  setAuthFetchHandler(() => modelResponse());
  store.setState({
    uploadedEvalFile: "unrelated.json",
    wandbToken: "old-secret",
    tensorboardDir: "/old/output",
    datasetSystemPrompt: "old prompt",
    loraRank: 512,
  });
  const config = {
    model_name: "org/model",
    hf_dataset: "org/data",
    training_type: "LoRA/QLoRA",
    load_in_4bit: true,
    batch_size: 8,
    learning_rate: "0.0001",
    lora_r: 64,
    target_modules: ["q_proj", "v_proj"],
    subset: "main",
    train_split: "train",
    eval_split: "test",
    format_type: "chatml",
    custom_format_mapping: {
      question: "user",
      answer: "assistant",
      __system_prompt: "Be concise",
      __label_mapping: { label: { "0": "no" } },
    },
    dataset_slice_start: 0,
    dataset_slice_end: 100,
    trust_remote_code: true,
    approved_remote_code_fingerprint: "old-approval",
    resume_from_checkpoint: "/old/checkpoint",
    output_dir: "/old/output",
  };
  store.getState().restoreRunConfig(config);
  await settle();
  const state = store.getState();
  const payload = buildTrainingStartPayload(state, null);
  for (const key of [
    "model_name",
    "hf_dataset",
    "training_type",
    "load_in_4bit",
    "batch_size",
    "learning_rate",
    "lora_r",
    "target_modules",
    "subset",
    "train_split",
    "eval_split",
    "format_type",
    "custom_format_mapping",
    "dataset_slice_start",
    "dataset_slice_end",
  ] as const) {
    assert.deepEqual(payload[key], config[key], key);
  }
  assert.equal(state.uploadedEvalFile, null);
  assert.equal(state.wandbToken, "");
  assert.equal(state.tensorboardDir, "");
  assert.equal(payload.trust_remote_code, false);
  assert.equal(payload.approved_remote_code_fingerprint, null);
  assert.equal(payload.resume_from_checkpoint, undefined);
  assert.equal(payload.hf_token, null);
});

test("decision model details do not replace a duplicated recipe", async () => {
  setAuthFetchHandler(() => modelResponse(true));
  store.getState().restoreRunConfig({
    model_name: "org/model",
    hf_dataset: "org/data",
    is_decision: true,
    decision_layout: "llm",
    training_type: "LoRA/QLoRA",
    load_in_4bit: true,
    learning_rate: "0.0003",
    lora_r: 96,
    batch_size: 3,
  });
  await settle();
  const state = store.getState();
  assert.equal(state.learningRate, 0.0003);
  assert.equal(state.batchSize, 3);
  assert.equal(state.loraRank, 96);
  assert.equal(state.trainAsDecision, true);
  assert.equal(state.trainingMethod, "qlora");
  assert.equal(state.modelType, "decision");
});

test("a duplicated CPT run switches to LoRA with the model's defaults", async () => {
  setAuthFetchHandler(() => modelResponse());
  store.getState().restoreRunConfig({
    model_name: "org/model",
    hf_dataset: "org/data",
    training_type: "Continued Pretraining",
    learning_rate: "0.00005",
    lora_r: 128,
    target_modules: ["q_proj", "embed_tokens", "lm_head"],
    format_type: "raw",
  });
  await settle();
  assert.equal(store.getState().trainingMethod, "cpt");
  assert.equal(store.getState().loraRank, 128);
  store.getState().setTrainingMethod("lora");
  const state = store.getState();
  assert.equal(state.loraRank, 8);
  assert.equal(state.learningRate, 0.9);
  assert.deepEqual(state.targetModules, ["q_proj"]);
});

test("partial configs use clean defaults and malformed runs leave the draft intact", async () => {
  setAuthFetchHandler(() => modelResponse());
  store.getState().setLoraRank(512);
  store.getState().restoreRunConfig({
    model_name: "org/model",
    local_datasets: ["saved.jsonl"],
    local_eval_datasets: ["eval.jsonl"],
  });
  await settle();
  assert.notEqual(store.getState().loraRank, 512);
  assert.equal(store.getState().datasetSource, "upload");
  assert.equal(store.getState().uploadedFile, "saved.jsonl");
  assert.equal(store.getState().uploadedEvalFile, "eval.jsonl");
  const before = store.getState();
  assert.throws(() => before.restoreRunConfig({}), /no saved model/);
  assert.equal(store.getState(), before);
});

test("S3 restores only saved bucket settings", async () => {
  setAuthFetchHandler(() => modelResponse());
  store.getState().restoreRunConfig({
    model_name: "org/model",
    dataset_source: "s3",
    s3_dataset: {
      bucket: "training",
      region: "eu-west-1",
      prefix: "data/",
      use_iam_role: false,
    },
    s3_config: { accessKeyId: "secret" },
  });
  await settle();
  assert.deepEqual(store.getState().s3Config, {
    bucket: "training",
    region: "eu-west-1",
    prefix: "data/",
    useIamRole: false,
  });
});

test("late model details preserve settings captured by a new CPT transition", async () => {
  let release: (response: Response) => void = () => {};
  setAuthFetchHandler(
    () =>
      new Promise<Response>((resolve) => {
        release = resolve;
      }),
  );
  store.getState().restoreRunConfig({
    model_name: "org/model",
    training_type: "LoRA/QLoRA",
    load_in_4bit: false,
    lora_r: 64,
    target_modules: ["v_proj"],
    format_type: "chatml",
    train_on_completions: true,
  });
  store.getState().setTrainingMethod("cpt");
  release(modelResponse());
  await settle();
  store.getState().setTrainingMethod("lora");
  assert.equal(store.getState().loraRank, 64);
  assert.deepEqual(store.getState().targetModules, ["v_proj"]);
  assert.equal(store.getState().trainOnCompletions, true);
});

test("a duplicated GRPO run keeps its objective, RL settings, rewards and column roles", async () => {
  setAuthFetchHandler(() => modelResponse());
  store.getState().restoreRunConfig({
    model_name: "org/model",
    hf_dataset: "org/data",
    training_type: "LoRA/QLoRA",
    custom_format_mapping: { question: "prompt", solution: "answer" },
    objective: "grpo",
    rl_settings: {
      beta: 0.04,
      num_generations: 8,
      temperature: 0.7,
      variant: "gspo",
      system_prompt: null,
    },
    reward_specs: [
      { name: "exact-answer", weight: 2, rule: { type: "exact_match" } },
    ],
  });
  await settle();
  const state = store.getState();
  assert.equal(state.trainingObjective, "grpo");
  assert.deepEqual(state.rlRoleMapping, {
    question: "prompt",
    solution: "answer",
  });
  assert.deepEqual(state.datasetManualMapping, {});
  assert.equal(state.rlBeta, 0.04);
  assert.equal(state.grpoNumGenerations, 8);
  assert.equal(state.grpoVariant, "gspo");
  assert.deepEqual(state.grpoRewards, [{ name: "exact-answer", weight: 2 }]);
  const payload = buildTrainingStartPayload(state);
  assert.equal(payload.objective, "grpo");
  assert.deepEqual(payload.custom_format_mapping, {
    question: "prompt",
    solution: "answer",
  });
  assert.deepEqual(payload.grpo_rewards, [{ name: "exact-answer", weight: 2 }]);
});
