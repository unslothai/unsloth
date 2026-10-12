// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  breakdownOfPersisted,
  downloadParts,
  withPlanBreakdown,
} = await import("../src/features/hub/download-manager/download-breakdown.ts");
const { diffusionStagingEntries } = await import(
  "../src/lib/diffusion-pipeline-load-target.ts"
);
const { checkpointFirst, selectDownloadEntries } = await import(
  "../src/features/hub/download-manager/required-assets.ts"
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

test("a resumed download's files already on disk don't fill the segments early", () => {
  // The job counts 19 GB; 4.6 GB of it was on disk before this run, so it isn't in fileBytes.
  const parts = downloadParts(qwen, 4.6 * GB + 10 * GB, 18.9 * GB + 4.6 * GB);
  assert.equal(parts?.find((p) => p.kind === "encoder")?.doneBytes, 10 * GB);
  assert.equal(parts?.find((p) => p.kind === "vae")?.doneBytes, 0);
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

test("a cached checkpoint lands on the first companion", () => {
  const companions = [
    { repoId: "a", checkpoint: false },
    { repoId: "b", checkpoint: false },
  ];
  assert.deepEqual(
    withPlanBreakdown(companions, 2.5 * GB).map((e) => e.cachedCheckpointBytes),
    [2.5 * GB, undefined],
  );
});

test("every job of a plan shows all three parts from the start", () => {
  // FLUX.2-klein-9B: the GGUF downloads as its own job before the companions.
  const [gguf, companion] = withPlanBreakdown(
    [
      { checkpoint: true, fileBytes: { "flux-2-klein-9b-Q4_K_M.gguf": 5.9 * GB } },
      {
        checkpoint: false,
        fileBytes: {
          "text_encoder/model.safetensors": 16 * GB,
          "vae/diffusion_pytorch_model.safetensors": 168e6,
        },
      },
    ],
    5.9 * GB,
  );
  const halfway = downloadParts(
    { fileBytes: gguf.fileBytes ?? {}, laterBytes: gguf.laterBytes },
    3 * GB,
  );
  assert.deepEqual(
    halfway?.map((p) => [p.kind, p.bytes, p.doneBytes]),
    [
      ["model", 5.9 * GB, 3 * GB],
      ["encoder", 16 * GB, 0],
      ["vae", 168e6, 0],
    ],
  );
  // The GGUF counts once: as the companion's earlier job, not also as a cached checkpoint.
  assert.equal(companion.cachedCheckpointBytes, undefined);
  const next = downloadParts(
    { fileBytes: companion.fileBytes ?? {}, earlierBytes: companion.earlierBytes },
    0,
  );
  assert.deepEqual(next?.[0], { kind: "model", bytes: 5.9 * GB, doneBytes: 5.9 * GB });
});

test("tokenizer and scheduler files count as the text encoder, so the bar has three parts", () => {
  const parts = downloadParts(
    {
      fileBytes: {
        "model_index.json": 1e3,
        "scheduler/scheduler_config.json": 1e3,
        "text_encoder/model-00001-of-00002.safetensors": 16 * GB,
        "tokenizer/tokenizer.json": 16e6,
        "vae/diffusion_pytorch_model.safetensors": 168e6,
      },
      cachedCheckpointBytes: 5.9 * GB,
    },
    0,
  );
  assert.deepEqual(
    parts?.map((p) => p.kind),
    ["model", "encoder", "vae"],
  );
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

test("a hosted prequant transformer at the repo root is the model, its encoder twin is not", () => {
  const parts = downloadParts(
    {
      fileBytes: {
        "Qwen-Image-2.1-FP8.safetensors": 12 * GB,
        "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors": 9 * GB,
        "vae/diffusion_pytorch_model.safetensors": 1.4 * GB,
      },
    },
    0,
  );
  assert.deepEqual(
    parts?.map((p) => [p.kind, p.bytes]),
    [
      ["model", 12 * GB],
      ["encoder", 9 * GB],
      ["vae", 1.4 * GB],
    ],
  );
});

test("LTX-2's vocoder is part of the VAE, and files fill in name order, not plan order", () => {
  const parts = downloadParts(
    {
      fileBytes: {
        "text_encoder/model-00001-of-00011.safetensors": 9 * GB,
        "vocoder/diffusion_pytorch_model.safetensors": 0.1 * GB,
        "audio_vae/diffusion_pytorch_model.safetensors": 0.1 * GB,
      },
    },
    0.1 * GB,
  );
  assert.deepEqual(
    parts?.map((p) => [p.kind, p.bytes, p.doneBytes]),
    [
      ["encoder", 9 * GB, 0],
      ["vae", 0.2 * GB, 0.1 * GB],
    ],
  );
});

test("the Hub queue sizes earlier and later parts in the order it runs the jobs", () => {
  // The plan lists the encoder repo before the GGUF; the queue fetches the checkpoint first.
  const plan: { repoId: string; bytes: number; checkpoint: boolean; fileBytes: Record<string, number> }[] = [
    {
      repoId: "unsloth/Qwen-Image-2.1-FP8",
      bytes: 9 * GB,
      checkpoint: false,
      fileBytes: { "Qwen-Image-2.1-text_encoder-FP8.safetensors": 9 * GB },
    },
    {
      repoId: "unsloth/Qwen-Image-2.1-GGUF",
      bytes: 12 * GB,
      checkpoint: true,
      fileBytes: { "qwen-image-2.1-Q4_K_M.gguf": 12 * GB },
    },
  ];
  const [first, second] = withPlanBreakdown(checkpointFirst(plan), 0);
  assert.equal(first.repoId, "unsloth/Qwen-Image-2.1-GGUF");
  assert.deepEqual(
    downloadParts(first, 0.1 * GB)?.map((p) => [p.kind, p.doneBytes]),
    [
      ["model", 0.1 * GB],
      ["encoder", 0],
    ],
  );
  assert.deepEqual(
    downloadParts(second, 0)?.map((p) => [p.kind, p.doneBytes]),
    [
      ["model", 12 * GB],
      ["encoder", 0],
    ],
  );
  // Assets left unticked: the checkpoint alone keeps the single bar.
  const [only] = withPlanBreakdown(checkpointFirst(selectDownloadEntries(plan, false)), 0);
  assert.equal(downloadParts(only, 6 * GB), null);
  // The Hub hook sizes the breakdown after the include choice and the reorder.
  assert.match(
    readSrc("features/hub/catalog/use-required-assets-download.tsx"),
    /withPlanBreakdown\(checkpointFirst\(entries\)/,
  );
});

test("a checkpoint the plan stages without bytes still shows as the cached model", () => {
  // An older snapshot holds the GGUF: the plan stages it to link the file but counts nothing.
  const [, companion] = withPlanBreakdown(
    [
      { checkpoint: true, fileBytes: {} },
      { checkpoint: false, fileBytes: { "text_encoder/model.safetensors": 9 * GB, "vae/diffusion_pytorch_model.safetensors": GB } },
    ],
    7 * GB,
  );
  assert.equal(companion.cachedCheckpointBytes, 7 * GB);
});

test("FLUX.1's root-level ae.safetensors counts as the VAE", () => {
  const parts = downloadParts(
    { fileBytes: { "flux1-dev-Q4_K_M.gguf": 6.8 * GB, "ae.safetensors": 0.335 * GB } },
    0,
  );
  assert.deepEqual(parts?.map((p) => p.kind), ["model", "vae"]);
});
