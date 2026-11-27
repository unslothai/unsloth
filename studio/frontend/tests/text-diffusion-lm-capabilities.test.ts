// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { detectCapabilities } = await import(
  "../src/features/hub/lib/model-capabilities.ts"
);

const keys = (tags: string[], pipeline: string | undefined, id: string) =>
  detectCapabilities(tags, pipeline, id).map((c) => c.key);

test("text-diffusion LMs with a bare diffusion tag are not image/video generators", () => {
  for (const [id, tags] of [
    ["GSAI-ML/LLaDA-8B-Instruct", ["llada", "diffusion"]],
    ["Dream-org/Dream-v0-Instruct-7B", ["diffusion"]],
    ["JetLM/SDAR-1.7B-Chat", ["sdar", "diffusion"]],
    ["google/diffusiongemma-26B-A4B-it", ["diffusion"]],
    ["nvidia/Nemotron-Labs-Diffusion-3B", ["diffusion"]],
  ] as const) {
    assert.ok(!keys([...tags], "text-generation", id).includes("diffusion"), id);
  }
});

test("image and video diffusion models keep the image/video badge", () => {
  assert.ok(keys(["diffusion"], undefined, "someone/my-model").includes("diffusion"));
  assert.ok(keys(["diffusers"], "text-to-image", "black-forest-labs/FLUX.1-dev").includes("diffusion"));
  assert.ok(keys([], undefined, "Lykon/dreamshaper-8").includes("diffusion"));
  // A text-diffusion family name does not hide a real image-generation tag.
  assert.ok(keys(["diffusers", "llada"], "text-to-image", "x/llada-image").includes("diffusion"));
});
