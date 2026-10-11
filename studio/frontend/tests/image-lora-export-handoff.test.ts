// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { savedLabel } from "../src/features/export/constants.ts";
import { catalogId } from "../src/features/images/train/diffusion-train-deploy.ts";

import { readSrcAsync } from "./helpers/kit.ts";

test("a published adapter's catalog id is its file stem on every OS", () => {
  assert.equal(catalogId("/home/u/.unsloth/studio/loras/diffusion/sksdog.safetensors"), "sksdog");
  assert.equal(catalogId("C:\\Users\\u\\loras\\diffusion\\my.style.gguf"), "my.style");
});

test("the export result names the exports folder unless the path is absolute", () => {
  assert.equal(savedLabel("image-loras/sksdog.safetensors"), "exports/image-loras/sksdog.safetensors");
  assert.equal(savedLabel("/data/loras/sksdog.safetensors"), "/data/loras/sksdog.safetensors");
  assert.equal(savedLabel("D:\\loras\\sksdog.safetensors"), "D:\\loras\\sksdog.safetensors");
});

test("the finished training run hands its LoRA to the Export page", async () => {
  const panel = await readSrcAsync("features/images/train/diffusion-train-panel.tsx");
  assert.match(panel, /to: "\/export",\s*search: \{ imageLora: catalogId\(/);
  const route = await readSrcAsync("app/routes/export.tsx");
  assert.match(route, /imageLora: typeof search\.imageLora === "string"/);
  const page = await readSrcAsync("features/export/export-page.tsx");
  assert.match(page, /<ImageLoraExportCard preselect=\{preselectImageLora\} \/>/);
});
