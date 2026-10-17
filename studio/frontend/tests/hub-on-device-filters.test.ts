// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// On Device takes Discover's filter panel: capabilities, format and sort live in the
// popover, and Add folder and Free up space are icon buttons.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { matchesInventoryCapabilities } = await import(
  "../src/features/hub/lib/model-type-filter.ts"
);

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

test("On Device moves format and sort into the filter panel", () => {
  const toolbar = readSrc("features/hub/catalog/models-toolbar.tsx");
  assert.match(toolbar, /isDiscover && !isDataset && \(\s*<HubOptionMenu<FormatMenuValue>/);
  assert.match(toolbar, /onDevice=\{onDeviceFilters\}/);
  const panel = readSrc("features/hub/catalog/model-search-filters.tsx");
  assert.match(panel, /label="Format"/);
  assert.match(panel, /label="Sort by"/);
  const page = readSrc("features/hub/hub-page.tsx");
  assert.doesNotMatch(page, /InventorySortControl|InventoryTypeFilterControl/);
});
