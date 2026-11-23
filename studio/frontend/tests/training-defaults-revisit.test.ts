// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after, beforeEach } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { useHfTokenStore } = await import(
  "../src/features/hub/stores/hf-token-store.ts"
);
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { deriveTrainingReadiness } = await import(
  "../src/features/training/hooks/use-training-readiness.ts"
);

type Store = typeof useTrainingConfigStore;

const MODEL = "unsloth/Qwen3-4B";
const DATASET = {
  datasetSource: "huggingface" as const,
  dataset: "yahma/alpaca-cleaned",
  datasetKnownCached: false,
  isDatasetImage: false,
  isCheckingDataset: false,
};

let requests: string[] = [];
let failNext = false;
let audioTypeKnown = true;

function details(): Record<string, unknown> {
  return {
    id: MODEL,
    config: {},
    is_vision: false,
    is_embedding: false,
    is_audio: false,
    audio_type_known: audioTypeKnown,
    is_lora: false,
    model_type: "text",
    model_size_bytes: null,
    max_position_embeddings: 40960,
  };
}

setAuthFetchHandler((input) => {
  if (!input.startsWith("/api/models/config/")) {
    return new Response("{}", { status: 404 });
  }
  requests.push(input);
  if (failNext) {
    failNext = false;
    return new Response("{}", { status: 500 });
  }
  return Response.json(details());
});

after(() => setAuthFetchHandler(null));

beforeEach(() => {
  requests = [];
  failNext = false;
  audioTypeKnown = true;
  useHfTokenStore.setState({ token: "" });
  useTrainingConfigStore.getState().reset();
});

async function settled(store: Store = useTrainingConfigStore): Promise<void> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const state = store.getState();
    if (!state.isLoadingModelDefaults && !state.isCheckingVision) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("model defaults did not settle");
}

async function pickModel(): Promise<void> {
  useTrainingConfigStore.getState().selectTrainingModel(MODEL, "text");
  await settled();
  useTrainingConfigStore.setState(DATASET);
}

async function visitTrain(store: Store = useTrainingConfigStore) {
  store.getState().ensureModelDefaultsLoaded();
  const duringMount = store.getState();
  await settled(store);
  return duringMount;
}

test("a second Train visit makes no request and leaves Start enabled", async () => {
  await pickModel();
  assert.equal(requests.length, 1);
  assert.equal(
    deriveTrainingReadiness(useTrainingConfigStore.getState(), "cuda", true)
      .isReady,
    true,
  );

  const duringMount = await visitTrain();
  assert.equal(requests.length, 1);
  assert.equal(duringMount.isLoadingModelDefaults, false);
  assert.equal(duringMount.isCheckingVision, false);
  const readiness = deriveTrainingReadiness(duringMount, "cuda", true);
  assert.equal(readiness.isLoadingModel, false);
  assert.equal(readiness.isReady, true);

  await visitTrain();
  assert.equal(requests.length, 1);
});

test("a changed Hugging Face token asks again", async () => {
  await pickModel();
  useHfTokenStore.setState({ token: "hf_new_token_for_test" });
  const duringMount = await visitTrain();
  assert.equal(duringMount.isLoadingModelDefaults, true);
  assert.equal(requests.length, 2);
  await visitTrain();
  assert.equal(requests.length, 2);
});

test("a changed cache reference or decision switch asks again", async () => {
  await pickModel();
  useTrainingConfigStore.setState({
    modelKnownCached: true,
    modelLocalPath: "/cache/models--unsloth--Qwen3-4B/snapshots/abc",
  });
  await visitTrain();
  assert.equal(requests.length, 2);
  assert.match(requests[1], /prefer_local_cache=true/);

  useTrainingConfigStore.setState({ trainAsDecision: true });
  await visitTrain();
  assert.equal(requests.length, 3);
  assert.match(requests[2], /as_decision=true/);
});

test("a failed load is asked again on the next visit", async () => {
  failNext = true;
  useTrainingConfigStore.getState().selectTrainingModel(MODEL, "text");
  await settled();
  assert.notEqual(useTrainingConfigStore.getState().modelDefaultsError, null);
  assert.equal(requests.length, 1);

  await visitTrain();
  assert.equal(requests.length, 2);
  assert.equal(useTrainingConfigStore.getState().modelDefaultsError, null);

  await visitTrain();
  assert.equal(requests.length, 2);
});

test("a failure after a success clears the marker, so the same request is retried", async () => {
  await pickModel();
  useHfTokenStore.setState({ token: "hf_other_token_for_test" });
  failNext = true;
  await visitTrain();
  assert.equal(requests.length, 2);
  assert.notEqual(useTrainingConfigStore.getState().modelDefaultsError, null);

  // Back to the token the first answer was for: still asked, the failure reset the marker.
  useHfTokenStore.setState({ token: "" });
  await visitTrain();
  assert.equal(requests.length, 3);
});

test("an answer whose probe could not read the repo is asked again", async () => {
  audioTypeKnown = false;
  await pickModel();
  await visitTrain();
  assert.equal(requests.length, 2);
});

test("reset forgets the applied defaults", async () => {
  await pickModel();
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState({
    selectedModel: MODEL,
    modelDefaultsAppliedFor: MODEL,
  });
  await visitTrain();
  assert.equal(requests.length, 2);
});

test("the first visit after a page reload still asks", async () => {
  await pickModel();
  await visitTrain();
  assert.equal(requests.length, 1);
  assert.equal(useTrainingConfigStore.getState().modelDefaultsAppliedFor, MODEL);

  // A fresh module instance is a reload: it rehydrates from storage with no in-memory marker.
  // The query keeps the module cache from handing back the instance already loaded.
  const { useTrainingConfigStore: reloaded } = (await import(
    new URL(
      "../src/features/training/stores/training-config-store.ts?reload",
      import.meta.url,
    ).href
  )) as { useTrainingConfigStore: Store };
  assert.notEqual(reloaded, useTrainingConfigStore);
  await reloaded.persist.rehydrate();
  assert.equal(reloaded.getState().selectedModel, MODEL);
  assert.equal(reloaded.getState().modelDefaultsAppliedFor, MODEL);
  await visitTrain(reloaded);
  assert.equal(requests.length, 2);
  await visitTrain(reloaded);
  assert.equal(requests.length, 2);
});
