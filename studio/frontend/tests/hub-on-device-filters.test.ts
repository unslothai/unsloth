// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// On Device takes Discover's filter panel for capabilities, keeps format and sort as
// dropdowns (with MLX and Studio's own trained and exported models), and icon-only folder and cleanup buttons.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { matchesInventoryCapabilities } = await import(
  "../src/features/hub/lib/model-type-filter.ts"
);
const { matchesFormat } = await import("../src/features/hub/lib/format-filters.ts");

function cachedRow(id: string, pipelineTag: string | null, tags: string[] = []) {
  return {
    kind: "cached",
    id,
    repoId: id,
    tags,
    pipelineTag,
    capabilities: { supportsVision: false },
  } as never;
}

test("no picks keeps every on-device row", () => {
  assert.equal(matchesInventoryCapabilities(cachedRow("a/b", "text-generation"), []), true);
});

test("on-device rows match Discover's capability picks", () => {
  const video = cachedRow("Wan-AI/Wan2.2-T2V-GGUF", "text-to-video", ["diffusers"]);
  const image = cachedRow("black-forest-labs/FLUX.1-dev-GGUF", "text-to-image", ["diffusers"]);
  const text = cachedRow("unsloth/Qwen3-8B-GGUF", "text-generation");
  assert.equal(matchesInventoryCapabilities(video, ["video-gen"]), true);
  assert.equal(matchesInventoryCapabilities(video, ["image-gen"]), false);
  assert.equal(matchesInventoryCapabilities(image, ["image-gen"]), true);
  assert.equal(matchesInventoryCapabilities(text, ["text"]), true);
  assert.equal(matchesInventoryCapabilities(text, ["image-gen", "video-gen"]), false);
});

test("a row's own vision flag counts as Vision", () => {
  const row = {
    ...(cachedRow("someone/plain", null) as object),
    capabilities: { supportsVision: true },
  } as never;
  assert.equal(matchesInventoryCapabilities(row, ["vision"]), true);
});

test("Add folder and Free up space are icon buttons with labels", () => {
  const toolbar = readSrc("features/hub/catalog/models-toolbar.tsx");
  for (const label of ["Add folder", "Free up space"]) {
    assert.match(toolbar, new RegExp(`aria-label="${label}"`));
    assert.doesNotMatch(toolbar, new RegExp(`/>\\s*${label}\\s*</button>`));
  }
});

test("On Device keeps format and sort as dropdowns, with MLX and Trained & exported", () => {
  const toolbar = readSrc("features/hub/catalog/models-toolbar.tsx");
  assert.match(toolbar, /\{!isDataset && \(\s*<HubOptionMenu<FormatMenuValue>/);
  assert.match(toolbar, /tab === "downloaded" && \(\s*<HubOptionMenu<InventorySort>/);
  assert.match(toolbar, /option\.value === "unsloth"\s*\? tab === "downloaded"\s*: option\.value !== "recommended" \|\| tab === "discover"/);
  const panel = readSrc("features/hub/catalog/model-search-filters.tsx");
  assert.doesNotMatch(panel, /label="Format"|label="Sort by"/);
  const page = readSrc("features/hub/hub-page.tsx");
  assert.doesNotMatch(page, /InventorySortControl|InventoryTypeFilterControl/);
  assert.match(page, /deferredFormatFilter === "unsloth"\s*\? row\.source === "unsloth"/);
  assert.match(page, /includeUnslothModels: true/);
});

test("Trained & exported is where a model came from, never a file format", () => {
  assert.equal(matchesFormat("gguf", "unsloth"), false);
  assert.equal(matchesFormat("safetensors", "unsloth"), false);
  assert.equal(matchesFormat("mlx", "mlx"), true);
});
