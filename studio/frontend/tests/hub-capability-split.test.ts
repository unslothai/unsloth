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
  // MiniMax-H3's pipeline, which the Video page runs.
  assert.deepEqual(generationKinds(["diffusers"], "image-text-to-video", "MiniMaxAI/MiniMax-H3"), { image: false, video: true });
  assert.deepEqual(
    detectCapabilities([], "image-text-to-video", "MiniMaxAI/MiniMax-H3").find((c) => c.key === "diffusion"),
    { key: "diffusion", label: "Video gen" },
  );
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
  assert.match(popover, /locked=\{next\.length === 0\}/);
  assert.match(popover, /icon=\{FilterMailIcon\}/);
});

test("the first pick from the default keeps just that capability", () => {
  const popover = readSrc("features/hub/catalog/model-search-filters.tsx");
  assert.match(popover, /capabilities\.length === 0 && isolateFirstCapability\s*\? \[option\.value\]/);
  // Armed once, until the first change from the defaults; the resets arm it again.
  const page = readSrc("features/hub/hub-page.tsx");
  assert.match(page, /setFirstPickArmed\(false\);\s*setCapabilityFilters\(next\);/);
  assert.match(page, /isolateFirstCapability=\{firstPickArmed\}/);
  assert.doesNotMatch(popover, />\s*Only\s*</);
});

test("size and context are stepped ranges whose ends mean no limit", () => {
  const popover = readSrc("features/hub/catalog/model-search-filters.tsx");
  assert.match(popover, /labels: \["<1B", null, "6B", "12B", "32B", null, "128B", ">500B"\]/);
  assert.match(popover, /"1M\+"\]/);
  assert.match(popover, /\[spec\.min\]: next\[0\] === 0 \? undefined : spec\.stops\[next\[0\]\]/);
  assert.match(popover, /\[spec\.max\]: next\[1\] === last \? undefined : spec\.stops\[next\[1\]\]/);
  assert.doesNotMatch(popover, /Apply (filters|limits)/);
});

test("Uploader is a two-way toggle between Unsloth and every uploader", () => {
  const popover = readSrc("features/hub/catalog/model-search-filters.tsx");
  assert.match(popover, /role="radiogroup"\s+aria-label="Uploader"/);
  assert.match(popover, /\["Unsloth", true\],\s*\["All uploaders", false\]/);
  assert.match(popover, /unslothOption \? \["unsloth"\] : \[\.\.\.HUB_UPLOADERS\]/);
});
