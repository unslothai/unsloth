// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A media GGUF's second download (text encoder, VAE, configs) must say what it is, and a quant on disk
// whose assets are still missing must not read as ready to run.

import assert from "node:assert/strict";
import test from "node:test";

import {
  assetLabel,
  requiredAssetKind,
} from "../src/features/hub/download-manager/required-assets.ts";
import { readSrc } from "./helpers/kit.ts";

test("companion files are named by what they hold", () => {
  assert.equal(
    requiredAssetKind(["Qwen-Image-2.1-text_encoder-FP8.safetensors"]),
    "Text encoder",
  );
  assert.equal(
    requiredAssetKind([
      "text_encoder/config.json",
      "vae/diffusion_pytorch_model.safetensors",
      "processor/tokenizer.json",
    ]),
    "Decoder & configuration",
  );
  assert.equal(
    requiredAssetKind([
      "text_encoder/model-00001-of-00004.safetensors",
      "vae/diffusion_pytorch_model.safetensors",
    ]),
    "Encoder & decoder",
  );
  // Configs alone name nothing; the caller keeps its own fallback.
  assert.equal(requiredAssetKind(["scheduler/scheduler_config.json"]), null);
  assert.equal(requiredAssetKind(undefined), null);
  assert.equal(
    assetLabel({
      repoId: "org/companion",
      bytes: 1,
      files: ["model_index.json"],
    }),
    "companion",
  );
});

test("the downloads panel names companion rows by component, running and queued", () => {
  const panel = readSrc(
    "features/hub/download-manager/download-manager-panel.tsx",
  );
  assert.match(panel, /requiredAssetKind\(files\) \?\? "Required assets"/);
  assert.match(panel, /requiredAssetSuffix\(job\.scopedFiles\)/);
  assert.match(panel, /requiredAssetSuffix\(entry\.files\)/);
  assert.match(panel, /Downloaded once and shared by every quant/);
  // The bare label is only the fallback now, never the row's whole description.
  assert.doesNotMatch(panel, /"Model file" : "Required assets"/);
});

test("a downloaded quant with missing assets shows what Run still fetches", () => {
  const card = readSrc("features/hub/catalog/gguf-download-card.tsx");
  assert.match(card, /more to run/);
  assert.match(card, /downloaded=\{item\.downloaded\}/);
  assert.match(card, /downloaded=\{Boolean\(selected\.downloaded\)\}/);
  assert.match(card, /data-model-needs-required-assets/);
});
