// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { fetchModelSize } = await import(
  "../src/features/hub/lib/dataset-size.ts"
);

function stubSiblings(sizes: Record<string, number>): void {
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({
        siblings: Object.entries(sizes).map(([rfilename, size]) => ({
          rfilename,
          size,
        })),
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    )) as typeof fetch;
}

test("a gpt-oss shaped repo is sized by its root safetensors only", async () => {
  stubSiblings({
    "config.json": 2,
    "model.safetensors.index.json": 36,
    "model-00000-of-00002.safetensors": 4_792,
    "model-00001-of-00002.safetensors": 4_798,
    "model-00002-of-00002.safetensors": 4_170,
    "original/dtypes.json": 13,
    "original/model.safetensors": 13_761,
    "metal/model.bin": 13_750,
  });
  assert.deepEqual(await fetchModelSize("acme/gpt-oss-shaped"), {
    totalBytes: 2 + 36 + 4_792 + 4_798 + 4_170,
    weightsBytes: 4_792 + 4_798 + 4_170,
  });
});

test("a whisper shaped repo is sized by its safetensors copy only", async () => {
  stubSiblings({
    "config.json": 2,
    "training_args.bin": 3,
    "model.safetensors": 967,
    "pytorch_model.bin": 967,
    "tf_model.h5": 968,
    "flax_model.msgpack": 967,
    "2_Dense/pytorch_model.bin": 9,
  });
  assert.deepEqual(await fetchModelSize("acme/whisper-shaped"), {
    totalBytes: 2 + 3 + 967 + 9,
    weightsBytes: 3 + 967 + 9,
  });
});

test("sibling formats still count when there is no root safetensors", async () => {
  stubSiblings({
    "config.json": 2,
    "pytorch_model.bin": 967,
    "flax_model.msgpack": 967,
    "original/model.safetensors": 900,
    "metal/model.bin": 900,
  });
  assert.deepEqual(await fetchModelSize("acme/bin-only"), {
    totalBytes: 2 + 967 + 967 + 900 + 900,
    weightsBytes: 967 + 967 + 900 + 900,
  });
});

test("underscore-sharded root safetensors still skip the bin copy", async () => {
  stubSiblings({
    "config.json": 2,
    "model_00001-of-00072.safetensors": 4_900,
    "model_00002-of-00072.safetensors": 4_900,
    "model.safetensors.index.json": 1,
    "pytorch_model_00001-of-00072.bin": 4_900,
    "pytorch_model_00002-of-00072.bin": 4_900,
    "pytorch_model.bin.index.json": 1,
  });
  assert.deepEqual(await fetchModelSize("acme/bloom-shaped"), {
    totalBytes: 2 + 4_900 + 4_900 + 1,
    weightsBytes: 4_900 + 4_900,
  });
});

test("indexless shards do not open the gate", async () => {
  // Numbered shards are resolved through model.safetensors.index.json; with no index the
  // bin copy is the only loadable checkpoint, so it must still be counted.
  stubSiblings({
    "config.json": 2,
    "model-00001-of-00002.safetensors": 500,
    "model-00002-of-00002.safetensors": 500,
    "pytorch_model.bin": 990,
  });
  assert.deepEqual(await fetchModelSize("acme/indexless-shards"), {
    totalBytes: 2 + 500 + 500 + 990,
    weightsBytes: 500 + 500 + 990,
  });
});

test("the same repo with an index does skip the bin copy", async () => {
  stubSiblings({
    "config.json": 2,
    "model-00001-of-00002.safetensors": 500,
    "model-00002-of-00002.safetensors": 500,
    "model.safetensors.index.json": 1,
    "pytorch_model.bin": 990,
  });
  assert.deepEqual(await fetchModelSize("acme/indexed-shards"), {
    totalBytes: 2 + 500 + 500 + 1,
    weightsBytes: 500 + 500,
  });
});

test("a non-ascii shard number does not open the gate", async () => {
  // The backend's ROOT_SAFETENSORS_RE is the authority on what gets downloaded. Python's
  // \d matches these digits and JavaScript's does not, so spelling either side \d would
  // size this repo here while the backend dropped its .bin copy. Both use [0-9]; neither
  // treats this as a root checkpoint, so nothing is skipped.
  stubSiblings({
    "config.json": 2,
    "model-٠١-of-٠٢.safetensors": 500,
    "pytorch_model.bin": 500,
  });
  assert.deepEqual(await fetchModelSize("acme/non-ascii-shards"), {
    totalBytes: 2 + 500 + 500,
    weightsBytes: 500 + 500,
  });
});
