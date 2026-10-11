// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_RESOLUTION,
  resolutionFor,
} from "../src/features/images/image-generation-defaults.ts";

const QWEN21 = "Qwen/Qwen-Image-2.1";

test("a quantised Qwen-Image-2.1 pipeline defaults to 512", () => {
  // Measured: about 26 GB at 1024 against about 19 GB at 512. Quantising the denoiser leaves the
  // activations alone, so on this family the canvas is what decides whether the load fits.
  for (const scheme of ["fp8", "int8", "fp8_dynamic", "nvfp4"]) {
    const size = resolutionFor(QWEN21, {
      modelKind: "pipeline",
      transformerQuant: scheme,
    });
    assert.deepEqual(size, { width: 512, height: 512 }, scheme);
  }
});

test("the single-file FP8 route gets it too", () => {
  // The hosted FP8 checkpoint loads as single_file, not pipeline. Only GGUF is excluded.
  assert.deepEqual(
    resolutionFor("unsloth/Qwen-Image-2.1-FP8", {
      modelKind: "single_file",
      transformerQuant: "fp8",
    }),
    { width: 512, height: 512 },
  );
});

test("a Qwen-Image-2.1 GGUF keeps 1024", () => {
  // The GGUF route streams the denoiser off disk, so its footprint does not turn on the canvas.
  // base_repo still carries the family substring, so the KIND is what has to exclude it.
  assert.deepEqual(
    resolutionFor(QWEN21, { modelKind: "gguf", transformerQuant: "int8" }),
    DEFAULT_RESOLUTION,
  );
});

test("a dense Qwen-Image-2.1 keeps 1024", () => {
  // Nobody running bf16 is short of VRAM, and shrinking its canvas would cost quality for nothing.
  // "off" is how the engaged record spells no quant; null is an older backend that never said.
  for (const scheme of ["off", "none", "", null, undefined]) {
    assert.deepEqual(
      resolutionFor(QWEN21, {
        modelKind: "pipeline",
        transformerQuant: scheme,
      }),
      DEFAULT_RESOLUTION,
      String(scheme),
    );
  }
});

test("other families keep 1024 even when quantised", () => {
  // The control. This is a per-family measurement, not a rule about quantisation, so a quantised
  // FLUX or Qwen-Image (2.0) must be untouched.
  for (const repo of [
    "black-forest-labs/FLUX.1-dev",
    "Qwen/Qwen-Image",
    "Tongyi-MAI/Z-Image-Turbo",
  ]) {
    assert.deepEqual(
      resolutionFor(repo, { modelKind: "pipeline", transformerQuant: "int8" }),
      DEFAULT_RESOLUTION,
      repo,
    );
  }
});

test("an unloaded or unknown model keeps 1024", () => {
  assert.deepEqual(
    resolutionFor("", { modelKind: null, transformerQuant: null }),
    DEFAULT_RESOLUTION,
  );
});
