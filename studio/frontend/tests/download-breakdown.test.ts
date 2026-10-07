// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  breakdownOfPersisted,
  downloadParts,
  withCachedCheckpoint,
} = await import("../src/features/hub/download-manager/download-breakdown.ts");
const { diffusionStagingEntries } = await import(
  "../src/lib/diffusion-pipeline-load-target.ts"
);

const GB = 1e9;
// Qwen-Image-2.1 Q2_K on the diffusers route: the GGUF is cached, Run fetches the rest.
const qwen = {
  fileBytes: {
    "text_encoder/model-00001-of-00004.safetensors": 5 * GB,
    "text_encoder/model-00002-of-00004.safetensors": 5 * GB,
    "text_encoder/model-00003-of-00004.safetensors": 5 * GB,
    "text_encoder/model-00004-of-00004.safetensors": 2.5 * GB,
    "vae/diffusion_pytorch_model.safetensors": 1.4 * GB,
  },
  cachedCheckpointBytes: 2.5 * GB,
};

test("a companion download splits into the cached model, text encoder and VAE", () => {
  const parts = downloadParts(qwen, 0);
  assert.deepEqual(
    parts?.map((p) => [p.kind, p.bytes, p.doneBytes]),
    [
      ["model", 2.5 * GB, 2.5 * GB],
      ["encoder", 17.5 * GB, 0],
      ["vae", 1.4 * GB, 0],
    ],
  );
});

test("progress fills the files in the order they download", () => {
  const parts = downloadParts(qwen, 18 * GB);
  assert.equal(parts?.find((p) => p.kind === "encoder")?.doneBytes, 17.5 * GB);
  assert.equal(parts?.find((p) => p.kind === "vae")?.doneBytes, 0.5 * GB);
  // The cached model is not part of the job's byte count.
  assert.equal(parts?.find((p) => p.kind === "model")?.doneBytes, 2.5 * GB);
});

test("one kind of file keeps the single bar", () => {
  assert.equal(
    downloadParts({ fileBytes: { "vae/diffusion_pytorch_model.safetensors": GB } }, 0),
    null,
  );
  assert.equal(downloadParts(undefined, 0), null);
});

test("a malformed persisted breakdown is dropped", () => {
  assert.deepEqual(breakdownOfPersisted({ fileBytes: { a: "1" } }), {});
  assert.deepEqual(breakdownOfPersisted({ fileBytes: [1] }), {});
  assert.deepEqual(breakdownOfPersisted(null), {});
  assert.deepEqual(breakdownOfPersisted(qwen), { breakdown: qwen });
});

test("the cached checkpoint lands on the first companion only when nothing stages it", () => {
  const companions = [
    { repoId: "a", checkpoint: false },
    { repoId: "b", checkpoint: false },
  ];
  assert.deepEqual(
    withCachedCheckpoint(companions, 2.5 * GB).map((e) => e.cachedCheckpointBytes),
    [2.5 * GB, undefined],
  );
  const withModel = [{ repoId: "m", checkpoint: true }, ...companions];
  assert.equal(withCachedCheckpoint(withModel, 2.5 * GB), withModel);
});

test("staging carries the plan's per-file sizes and the cached checkpoint", () => {
  const entries = diffusionStagingEntries(
    [
      {
        repo_id: "unsloth/Qwen-Image-2.1",
        files: Object.keys(qwen.fileBytes),
        bytes: 18.9 * GB,
        file_bytes: qwen.fileBytes,
        gguf_filename: null,
        checkpoint: false,
      },
    ],
    "unsloth/Qwen-Image-2.1-GGUF",
    { filename: "qwen-image-2.1-Q2_K.gguf", checkpointBytes: 2.5 * GB },
  );
  assert.equal(entries.length, 1);
  assert.deepEqual(entries[0].fileBytes, qwen.fileBytes);
  assert.equal(entries[0].cachedCheckpointBytes, 2.5 * GB);
});
