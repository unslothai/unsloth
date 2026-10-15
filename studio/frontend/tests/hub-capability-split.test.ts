// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Hub filters image and video generation separately, though both stay one "diffusion"
// capability for loading and the Images and Video pages.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { detectCapabilities, generationKinds, HUB_CAPABILITY_FILTERS } = await import(
  "../src/features/hub/lib/model-capabilities.ts"
);

test("video models are video gen, image models image gen", () => {
  assert.deepEqual(generationKinds(["diffusers"], "text-to-video", "Wan-AI/Wan2.2-T2V"), { image: false, video: true });
  assert.deepEqual(generationKinds(["diffusers"], "text-to-image", "black-forest-labs/FLUX.1-dev"), { image: true, video: false });
  // A video family named without a video tag still counts as video.
  assert.equal(generationKinds(["diffusers"], undefined, "tencent/HunyuanVideo").video, true);
  // A generator with no signal either way is taken for an image model.
  assert.deepEqual(generationKinds(["diffusers"], undefined, "someone/custom-gen"), { image: true, video: false });
});

test("the shared capability stays diffusion, labelled for what the model makes", () => {
  const video = detectCapabilities(["diffusers"], "text-to-video", "Wan-AI/Wan2.2-T2V");
  assert.deepEqual(video.find((c) => c.key === "diffusion"), { key: "diffusion", label: "Video gen" });
  const image = detectCapabilities(["diffusers"], "text-to-image", "black-forest-labs/FLUX.1-dev");
  assert.deepEqual(image.find((c) => c.key === "diffusion"), { key: "diffusion", label: "Image gen" });
});

test("the Hub lists Image gen and Video gen as two picks", () => {
  assert.ok(HUB_CAPABILITY_FILTERS.includes("image-gen"));
  assert.ok(HUB_CAPABILITY_FILTERS.includes("video-gen"));
  assert.ok(!(HUB_CAPABILITY_FILTERS as readonly string[]).includes("diffusion"));
  const options = readSrc("features/hub/lib/view-models.ts");
  assert.match(options, /\{ value: "image-gen", label: "Image gen" \}/);
  assert.match(options, /\{ value: "video-gen", label: "Video gen" \}/);
});

test("the Images and Video pages' Search Hub links open their own pick", () => {
  assert.match(readSrc("features/images/images-page.tsx"), /hubCapability="image-gen"/);
  assert.match(readSrc("features/video/video-page.tsx"), /hubCapability="video-gen"/);
  // Older links carry the single "diffusion" filter.
  assert.match(readSrc("app/routes/hub.tsx"), /search\.capability === "diffusion" \? "image-gen"/);
});

test("capabilities are picked in the filter popover, which sits by the Models / Datasets toggle", () => {
  const toolbar = readSrc("features/hub/catalog/models-toolbar.tsx");
  assert.doesNotMatch(toolbar, /ariaLabel="Capability filter"/);
  const filter = toolbar.indexOf("<ModelSearchFiltersControl");
  const toggle = toolbar.indexOf('aria-label="Resource type"');
  assert.ok(filter > toolbar.indexOf('ariaLabel="Sort models"') && filter < toggle);
  assert.match(toolbar, /capabilities=\{capabilityFilters\}/);
  const popover = readSrc("features/hub/catalog/model-search-filters.tsx");
  assert.match(popover, /CAPABILITY_FILTER_OPTIONS\.map\(\(option\) => \{/);
  assert.match(popover, /role="checkbox"\s+aria-checked=\{checked\}/);
  assert.match(popover, /\{showModelFilters && \(/);
});

test("Text only lists the models no other box covers, so every box ticked lists every model", () => {
  assert.equal(HUB_CAPABILITY_FILTERS[0], "text");
  const viewModels = readSrc("features/hub/lib/view-models.ts");
  assert.match(viewModels, /\{ value: "text", label: "Text only" \}/);
  assert.match(viewModels, /if \(key === "text"\) \{\s*return !HUB_CAPABILITY_FILTERS\.some\(\s*\(other\) => other !== "text" && matchesHubCapability\(row, other\),/);
});

test("every capability starts ticked, and the last tick can't be removed", () => {
  const popover = readSrc("features/hub/catalog/model-search-filters.tsx");
  assert.match(popover, /capabilities\.length === 0 \|\| capabilities\.includes\(key\)/);
  assert.match(popover, /const locked = next\.length === 0;/);
  assert.match(popover, /icon=\{FilterMailIcon\}/);
});
