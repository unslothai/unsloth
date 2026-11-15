// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MINIMAX_H3_IMAGE_PRESETS,
  familyPresetsCopy,
  familyPresetsWithin,
  inputImageReference,
  inputImageWord,
  maxAdditionalImages,
  presetsWithin,
} from "../src/features/images/edit-conditioning.ts";
import {
  defaultsFor,
  defaultsKeyFor,
  fixedStepsFor,
} from "../src/features/images/image-generation-defaults.ts";
import { sizeLimitsFrom } from "../src/features/images/image-size.ts";
import { familyOverrideOptions } from "../src/features/model-picker/components/model-selector/family-override.ts";
import {
  classifyHost,
  curatedArtifactIsOfferable,
} from "../src/features/model-picker/components/model-selector/host-artifact-policy.ts";
import { detectCapabilities } from "../src/features/model-picker/components/model-selector/model-capabilities.ts";
import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  artifactForRepoId,
  catalogToModelOptions,
  curatedArtifactFit,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { generationKinds, isVideoFamilyName } = await import(
  "../src/features/hub/lib/model-capabilities.ts"
);

const REPO = "MiniMaxAI/MiniMax-H3-Image";

// What the backend reports for a loaded MiniMax-H3-Image (diffusion_conditioning for the family).
const H3_IMAGE_CONDITIONING = {
  max_condition_images: 16,
  alpha: false,
  dimension_multiple: 32,
  max_output_side: 6272,
  max_output_pixels: 20 * 1024 * 1024,
  reference_resolutions: [],
  unified_edit: true,
  multi_image_edit: false,
  localized_edit_modes: [],
};

test("MiniMax-H3-Image is an image model, while MiniMax-H3 stays a video model", () => {
  for (const id of [
    REPO,
    "unsloth/MiniMax-H3-Image-FP8",
    "/models/minimax_h3_image",
  ]) {
    const caps = detectCapabilities({ id });
    assert.equal(caps.imageGen, true, id);
    assert.equal(caps.videoGen, false, id);
    assert.equal(isVideoFamilyName(id), false, id);
    assert.deepEqual(
      generationKinds([], "text-to-image", id),
      { image: true, video: false },
      id,
    );
  }
  for (const id of [
    "MiniMaxAI/MiniMax-H3",
    "unsloth/MiniMax-H3-GGUF",
    "Comfy-Org/MiniMax-H3",
  ]) {
    assert.equal(detectCapabilities({ id }).videoGen, true, id);
    assert.equal(isVideoFamilyName(id), true, id);
  }
});

test("the family selector names MiniMax-H3-Image", () => {
  const options = familyOverrideOptions(["minimax-h3", "minimax-h3-image"]);
  assert.deepEqual(options.slice(1), [
    ["minimax-h3", "MiniMax-H3"],
    ["minimax-h3-image", "MiniMax-H3-Image"],
  ]);
});

test("the image catalog lists MiniMax-H3-Image as a pipeline, not the video catalog", () => {
  const hit = artifactForRepoId(REPO, IMAGE_CATALOG);
  assert.ok(hit);
  assert.equal(hit.group.scope, "image");
  assert.equal(hit.artifact.loadKind, "pipeline");
  assert.equal(artifactForRepoId(REPO, VIDEO_CATALOG), null);
});

test("it fits the measured L4 and A100 40GB tiers only where a dense quant runs", () => {
  const dense = ["int8"];
  const fit = (
    gpuGb: number,
    systemRamGb: number,
    denseQuantSchemes: string[],
  ) =>
    curatedArtifactFit(REPO, IMAGE_CATALOG, {
      gpuGb,
      systemRamGb,
      denseQuantSchemes,
    })?.fits;
  assert.equal(fit(22, 52, dense), true);
  assert.equal(fit(39.5, 83, dense), true);
  assert.equal(fit(22, 52, []), false);
  assert.equal(fit(16, 64, dense), false);
  assert.equal(fit(22, 32, dense), false);
});

test("Apple Silicon and CPU hosts keep the row but it never reads as fitting (the backend refuses the load)", () => {
  // A Mac must not open the picker with a family missing, so the row stays and the load answers with the backend's
  // "needs an NVIDIA or AMD GPU" refusal (diffusion_minimax_h3_image.h3_image_device_refusal).
  const mac = classifyHost({
    deviceType: "mac",
    deviceBackend: "mlx",
    budgetKnown: true,
  });
  const cpu = classifyHost({ deviceBackend: "cpu", budgetKnown: true });
  for (const host of [mac, cpu]) {
    assert.equal(curatedArtifactIsOfferable(REPO, host), true);
    assert.equal(
      catalogToModelOptions(IMAGE_CATALOG, host).some((o) => o.id === REPO),
      true,
    );
  }
  // Unified memory with no dense quant (MPS / CPU): no tier is met.
  assert.equal(
    curatedArtifactFit(REPO, IMAGE_CATALOG, {
      gpuGb: 128,
      systemRamGb: 128,
      denseQuantSchemes: [],
    })?.fits,
    false,
  );
  for (const deviceBackend of ["cuda", "rocm"]) {
    const host = classifyHost({
      deviceBackend,
      budgetKnown: true,
      denseQuantSupported: true,
    });
    assert.equal(
      catalogToModelOptions(IMAGE_CATALOG, host).some((o) => o.id === REPO),
      true,
      deviceBackend,
    );
  }
});

test("the page opens on 6 steps at guidance 1, and the step count is locked", () => {
  assert.deepEqual(defaultsFor(REPO), { steps: 6, guidance: 1 });
  assert.deepEqual(defaultsFor("/models/minimax_h3_image"), {
    steps: 6,
    guidance: 1,
  });
  assert.deepEqual(
    defaultsFor(defaultsKeyFor("/models/opaque", "minimax-h3-image")),
    { steps: 6, guidance: 1 },
  );
  assert.equal(fixedStepsFor("minimax-h3-image"), 6);
  assert.equal(fixedStepsFor("flux.3-klein"), null);
  assert.equal(fixedStepsFor(null), null);
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /value=\{fixedSteps \?\? steps\}/);
  assert.match(page, /disabled=\{fixedSteps !== null\}/);
});

test("its size presets are the 4 MP and 16 MP tables, all inside the backend's limits", () => {
  const limits = sizeLimitsFrom(H3_IMAGE_CONDITIONING);
  const presets = familyPresetsWithin("minimax-h3-image", limits);
  assert.deepEqual(presets, MINIMAX_H3_IMAGE_PRESETS);
  assert.equal(presets.length, 15);
  for (const p of presets) {
    assert.equal(p.width % 32, 0, p.label);
    assert.equal(p.height % 32, 0, p.label);
    assert.ok(p.width * p.height >= 2048 * 2048 * 0.98, p.label);
  }
  assert.equal(presets.filter((p) => p.label.endsWith("16 MP")).length, 8);
  assert.match(familyPresetsCopy("minimax-h3-image").hint, /at least 4 MP/);
  assert.deepEqual(
    familyPresetsWithin("flux.3-klein", limits),
    presetsWithin(limits),
  );
  assert.equal(familyPresetsCopy("flux.3-klein").label, "2K presets");
});

test("edits take up to 16 pictures, named the way the prompt refers to them", () => {
  assert.equal(maxAdditionalImages(H3_IMAGE_CONDITIONING, null), 15);
  assert.equal(inputImageWord("minimax-h3-image"), "Picture");
  assert.equal(inputImageReference("minimax-h3-image", 2), "<Picture 2>");
  assert.equal(inputImageWord("qwen-image-edit"), "Image");
  const page = readSrc("features/images/images-page.tsx");
  // The inputs reuse the page's crop editor (ImageDropzone's cropSubject) under the family's word.
  assert.match(page, /cropSubject=\{`\$\{imageWord\} \$\{numberOf\(i\)\}`\}/);
  assert.match(page, /label=\{`\$\{imageWord\} 1 \(source\)`\}/);
});
