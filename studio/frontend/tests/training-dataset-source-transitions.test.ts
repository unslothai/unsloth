// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";

import { createServer } from "vite";

const values = new Map<string, string>();
const storage = {
  getItem: (key: string) => values.get(key) ?? null,
  setItem: (key: string, value: string) => values.set(key, value),
  removeItem: (key: string) => values.delete(key),
};
values.set(
  "unsloth_training_config_v1",
  JSON.stringify({
    state: {
      browseDatasetSelection: {
        source: "upload",
        uploadedFile: "/datasets/uploads/persisted.jsonl",
      },
      datasetSource: "upload",
      datasetStreaming: true,
      evalSteps: 0.1,
      uploadedFile: "/datasets/uploads/persisted.jsonl",
    },
    version: 20,
  }),
);
const location = { protocol: "http:" };
const windowTarget = {
  addEventListener: () => undefined,
  localStorage: storage,
  location,
  removeEventListener: () => undefined,
};
const head = { appendChild: () => undefined };
const documentTarget = {
  addEventListener: () => undefined,
  createElement: () => ({ appendChild: () => undefined }),
  createTextNode: () => ({}),
  getElementsByTagName: () => [head],
  head,
  removeEventListener: () => undefined,
};

let fetchCalls = 0;
Object.assign(globalThis, {
  document: documentTarget,
  fetch: () => {
    fetchCalls += 1;
    return Promise.resolve(
      new Response(
        '{"columns":["text"],"detected_format":"raw","is_audio":false,"is_image":false,"requires_manual_mapping":false}',
        { headers: { "Content-Type": "application/json" }, status: 200 },
      ),
    );
  },
  localStorage: storage,
  location,
  window: windowTarget,
});

const server = await createServer({
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const { useTrainingConfigStore } = await server.ssrLoadModule(
  "/src/features/training/stores/training-config-store.ts",
);
const { buildTrainingStartPayload } = await server.ssrLoadModule(
  "/src/features/training/api/mappers.ts",
);
const hydratedState = useTrainingConfigStore.getState();
const hydratedDatasetState = {
  browseDatasetSelection: hydratedState.browseDatasetSelection,
  datasetSource: hydratedState.datasetSource,
  datasetStreaming: hydratedState.datasetStreaming,
  evalSteps: hydratedState.evalSteps,
  uploadedFile: hydratedState.uploadedFile,
};

after(() => server.close());

function resetState(overrides: Record<string, unknown>): void {
  useTrainingConfigStore.getState().reset();
  useTrainingConfigStore.setState(overrides);
}

test("hydration repairs persisted upload streaming without dropping evaluation", () => {
  assert.deepEqual(hydratedDatasetState, {
    browseDatasetSelection: {
      source: "upload",
      uploadedFile: "/datasets/uploads/persisted.jsonl",
    },
    datasetSource: "upload",
    datasetStreaming: false,
    evalSteps: 0.1,
    uploadedFile: "/datasets/uploads/persisted.jsonl",
  });
});

test("upload selection clears Hub streaming and preserves uploaded evaluation", () => {
  resetState({
    browseDatasetSelection: {
      dataset: "org/streamed",
      knownCached: false,
      localPath: null,
      source: "huggingface",
    },
    dataset: "org/streamed",
    datasetSource: "huggingface",
    datasetStreaming: true,
    evalSteps: 0,
  });

  useTrainingConfigStore.getState().selectLocalDataset(null);
  useTrainingConfigStore
    .getState()
    .setUploadedFile("/datasets/uploads/train.jsonl");
  useTrainingConfigStore
    .getState()
    .setUploadedEvalFile("/datasets/uploads/eval.jsonl");

  const state = useTrainingConfigStore.getState();
  assert.equal(state.datasetSource, "upload");
  assert.equal(state.datasetStreaming, false);
  assert.equal(state.evalSteps, 0.1);
  assert.deepEqual(state.browseDatasetSelection, {
    source: "upload",
    uploadedFile: "/datasets/uploads/train.jsonl",
  });

  const payload = buildTrainingStartPayload(state, null);
  assert.equal(payload.hf_dataset, null);
  assert.equal(payload.dataset_streaming, false);
  assert.deepEqual(payload.local_datasets, ["/datasets/uploads/train.jsonl"]);
  assert.deepEqual(payload.local_eval_datasets, [
    "/datasets/uploads/eval.jsonl",
  ]);
  assert.equal(payload.eval_steps, 0.1);
  assert.equal(payload.s3_config, null);
});

test("cached Hub selection waits for a resolved split before checking format", async () => {
  resetState({ datasetSource: "huggingface" });
  const beforeSelection = fetchCalls;

  useTrainingConfigStore.getState().selectHfDataset("org/validation-only", {
    knownCached: true,
    localPath: "/cache/datasets--org--validation-only",
    preferLocalCache: true,
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fetchCalls, beforeSelection);

  useTrainingConfigStore.getState().setDatasetSplit(null);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fetchCalls, beforeSelection);

  useTrainingConfigStore.getState().ensureDatasetChecked();
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fetchCalls, beforeSelection);

  useTrainingConfigStore.getState().setDatasetSplit("validation");
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fetchCalls, beforeSelection + 1);
});

test("remote Hub selection preserves its immediate default split check", async () => {
  resetState({ datasetSource: "huggingface" });
  const beforeSelection = fetchCalls;

  useTrainingConfigStore.getState().selectHfDataset("org/remote");
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fetchCalls, beforeSelection + 1);

  useTrainingConfigStore.getState().setDatasetSplit("train");
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fetchCalls, beforeSelection + 2);
});

test("streaming cached Hub selection preserves its immediate split check", async () => {
  resetState({ datasetSource: "huggingface", datasetStreaming: true });
  const beforeSelection = fetchCalls;

  useTrainingConfigStore.getState().selectHfDataset("org/cached-stream", {
    knownCached: true,
    localPath: "/cache/datasets--org--cached-stream",
  });
  await new Promise<void>((resolve) => setImmediate(resolve));

  assert.equal(useTrainingConfigStore.getState().datasetStreaming, true);
  assert.equal(fetchCalls, beforeSelection + 1);
});

test("S3 selection clears streaming and restores the prior Hub selection", async () => {
  resetState({
    browseDatasetSelection: {
      dataset: "org/cached",
      knownCached: true,
      localPath: "/cache/datasets--org--cached",
      source: "huggingface",
    },
    dataset: "org/cached",
    datasetKnownCached: true,
    datasetLocalPath: "/cache/datasets--org--cached",
    datasetSource: "huggingface",
    datasetStreaming: true,
  });

  useTrainingConfigStore.getState().selectS3Source();
  useTrainingConfigStore.getState().setS3Config({
    accessKeyId: "key",
    bucket: "training-data",
    prefix: "datasets/train",
    region: "eu-north-1",
    secretAccessKey: "secret",
  });

  const s3State = useTrainingConfigStore.getState();
  assert.equal(s3State.datasetSource, "s3");
  assert.equal(s3State.datasetStreaming, false);
  assert.deepEqual(s3State.browseDatasetSelection, {
    dataset: "org/cached",
    knownCached: true,
    localPath: "/cache/datasets--org--cached",
    source: "huggingface",
  });

  const payload = buildTrainingStartPayload(s3State, null);
  assert.equal(payload.hf_dataset, null);
  assert.equal(payload.dataset_streaming, false);
  assert.deepEqual(payload.local_datasets, []);
  assert.deepEqual(payload.local_eval_datasets, []);
  assert.deepEqual(payload.s3_config, {
    accessKeyId: "key",
    bucket: "training-data",
    prefix: "datasets/train",
    region: "eu-north-1",
    secretAccessKey: "secret",
  });

  useTrainingConfigStore.getState().restoreBrowseDatasetSource();
  await new Promise<void>((resolve) => setImmediate(resolve));

  const restored = useTrainingConfigStore.getState();
  assert.equal(restored.datasetSource, "huggingface");
  assert.equal(restored.dataset, "org/cached");
  assert.equal(restored.datasetKnownCached, true);
  assert.equal(restored.datasetLocalPath, "/cache/datasets--org--cached");
  assert.equal(restored.datasetStreaming, false);
});

test("S3 preserves and restores a prior uploaded selection", () => {
  resetState({
    browseDatasetSelection: {
      source: "upload",
      uploadedFile: String.raw`C:\datasets\train.JSONL`,
    },
    datasetSource: "upload",
    datasetStreaming: true,
    evalSteps: 0.1,
    uploadedEvalFile: String.raw`C:\datasets\eval.JSONL`,
    uploadedFile: String.raw`C:\datasets\train.JSONL`,
  });

  useTrainingConfigStore.getState().selectS3Source();
  assert.equal(useTrainingConfigStore.getState().datasetStreaming, false);
  assert.deepEqual(useTrainingConfigStore.getState().browseDatasetSelection, {
    source: "upload",
    uploadedFile: String.raw`C:\datasets\train.JSONL`,
  });

  useTrainingConfigStore.getState().restoreBrowseDatasetSource();
  const restored = useTrainingConfigStore.getState();
  assert.equal(restored.datasetSource, "upload");
  assert.equal(restored.uploadedFile, String.raw`C:\datasets\train.JSONL`);
  assert.equal(restored.datasetStreaming, false);
});

test("reselecting a non-Hub source repairs stale streaming state", () => {
  for (const datasetSource of ["upload", "s3"] as const) {
    resetState({ datasetSource, datasetStreaming: true, evalSteps: 0.1 });
    useTrainingConfigStore.getState().setDatasetSource(datasetSource);
    const state = useTrainingConfigStore.getState();
    assert.equal(state.datasetStreaming, false);
    assert.equal(state.evalSteps, 0.1);
    state.setDatasetStreaming(true);
    assert.equal(useTrainingConfigStore.getState().datasetStreaming, false);
    assert.equal(useTrainingConfigStore.getState().evalSteps, 0.1);
  }
});

test("every manual dataset draft edit advances the user edit revision", () => {
  resetState({ manualDatasetOptionsValid: true, userEditRevision: 41 });

  useTrainingConfigStore.getState().markManualDatasetOptionsEdited(true);
  assert.equal(useTrainingConfigStore.getState().userEditRevision, 42);
  assert.equal(useTrainingConfigStore.getState().manualDatasetOptionsValid, true);

  useTrainingConfigStore.getState().markManualDatasetOptionsEdited(false);
  assert.equal(useTrainingConfigStore.getState().userEditRevision, 43);
  assert.equal(
    useTrainingConfigStore.getState().manualDatasetOptionsValid,
    false,
  );
});
