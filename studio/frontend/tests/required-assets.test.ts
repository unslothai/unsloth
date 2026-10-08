// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  checkpointFirst,
  assetLabel,
  additionalAssetDownloads,
  selectDownloadEntries,
  downloadBytes,
  formatDownloadBytes,
} from "../src/features/hub/download-manager/required-assets.ts";
const model = {
  repoId: "m",
  files: ["model.gguf"],
  checkpoint: true,
  bytes: 6.3e9,
};
const assets = {
  repoId: "a",
  files: ["encoder.bin"],
  checkpoint: false,
  bytes: 10.8e9,
};
test("model-only preserves checkpoints and excludes explicitly identified assets", () => {
  assert.deepEqual(selectDownloadEntries([model, assets], false), [model]);
  assert.deepEqual(selectDownloadEntries([model, assets], true), [
    model,
    assets,
  ]);
  assert.equal(formatDownloadBytes(downloadBytes([model, assets])), "17.1 GB");
});
test("unknown-size assets still need disclosure; an absent marker is not an asset", () => {
  const unsized = { ...assets, bytes: 0 };
  assert.deepEqual(
    additionalAssetDownloads([model, unsized, { repoId: "legacy", bytes: 2 }]),
    [unsized],
  );
  assert.equal(formatDownloadBytes(0), "Size unknown");
});
test("cache-aware plans with only checkpoints do not request an asset download", () => {
  assert.deepEqual(additionalAssetDownloads([model]), []);
  assert.deepEqual(additionalAssetDownloads([]), []);
});

test("encoder configuration does not mislabel a decoder asset group", () => {
  assert.equal(
    assetLabel({
      repoId: "unsloth/Qwen-Image-2.1",
      bytes: 1.4e9,
      files: [
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors",
        "processor/tokenizer.json",
      ],
    }),
    "Decoder & configuration",
  );
  assert.equal(
    assetLabel({
      repoId: "unsloth/Qwen-Image-2.1-FP8",
      bytes: 9.4e9,
      files: ["Qwen-Image-2.1-text_encoder-FP8.safetensors"],
    }),
    "Text encoder",
  );
});

test("encoder-first plans download the checkpoint first without mutating the plan", () => {
  const decoder = { ...assets, repoId: "decoder" };
  const original = [assets, model, decoder];
  assert.deepEqual(checkpointFirst(original), [model, assets, decoder]);
  assert.deepEqual(original, [assets, model, decoder]);
  assert.deepEqual(checkpointFirst([assets, decoder]), [assets, decoder]);
});
