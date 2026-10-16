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
const { useTrainingConfigStore: store } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { createActivationAccumulator } = await import(
  "../src/features/training/hooks/use-activation-data.ts"
);

test("an audio model starts with capture off even if the stored pick says on", () => {
  store.setState({ enableActivationCapture: true, isAudioModel: true });
  assert.equal(buildTrainingStartPayload(store.getState(), null).enable_activation_capture, false);
  store.setState({ isAudioModel: false });
  assert.equal(buildTrainingStartPayload(store.getState(), null).enable_activation_capture, true);
});

function record(step: number) {
  return { step, loss: 1, layers: {}, grad_norms: null, lora_norms: null } as never;
}

test("a response for the previous job is dropped after a job change", async () => {
  const pending: Array<(v: unknown) => void> = [];
  const acc = createActivationAccumulator(
    (() => new Promise((resolve) => pending.push(resolve))) as never,
  );
  const oldJob = acc.fetch("old");
  acc.reset();
  const newJob = acc.fetch("new");
  pending[0]({ metadata: null, records: [record(5), record(6)] });
  pending[1]({ metadata: null, records: [record(1)] });
  assert.equal(await oldJob, null);
  const data = await newJob;
  assert.deepEqual(data?.records.map((r) => r.step), [1]);
});

test("the next fetch for the same job asks from the last step it saw", async () => {
  const since: Array<number | undefined> = [];
  const acc = createActivationAccumulator((async (_j: unknown, _o: unknown, after: number | undefined) => {
    since.push(after);
    return { metadata: null, records: [record(since.length * 10)] };
  }) as never);
  await acc.fetch("job");
  const data = await acc.fetch("job");
  assert.deepEqual(since, [undefined, 10]);
  assert.deepEqual(data?.records.map((r) => r.step), [10, 20]);
});

test("two overlapping fetches for the same job merge each step once", async () => {
  const pending: Array<(v: unknown) => void> = [];
  const acc = createActivationAccumulator(
    (() => new Promise((resolve) => pending.push(resolve))) as never,
  );
  const first = acc.fetch("job");
  const second = acc.fetch("job");
  pending[0]({ metadata: null, records: [record(1), record(2)] });
  pending[1]({ metadata: null, records: [record(1), record(2), record(3)] });
  await first;
  const data = await second;
  assert.deepEqual(data?.records.map((r) => r.step), [1, 2, 3]);
});

test("embedding and decision runs start with capture off", () => {
  store.setState({ enableActivationCapture: true, isAudioModel: false });
  for (const patch of [
    { isEmbeddingModel: true, modelType: "text" },
    { isEmbeddingModel: false, modelType: "embeddings" },
    { isEmbeddingModel: false, modelType: "decision" },
  ] as const) {
    store.setState(patch as never);
    assert.equal(
      buildTrainingStartPayload(store.getState(), null).enable_activation_capture,
      false,
      JSON.stringify(patch),
    );
  }
  store.setState({ isEmbeddingModel: false, modelType: "text" } as never);
  assert.equal(buildTrainingStartPayload(store.getState(), null).enable_activation_capture, true);
});
