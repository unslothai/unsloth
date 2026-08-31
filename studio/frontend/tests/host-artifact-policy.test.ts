// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyHost,
  curatedArtifactIsOfferable,
  h3PerfSuffix,
} from "../src/features/model-picker/components/model-selector/host-artifact-policy.ts";

test("the backends that can place a diffusion pipeline are accelerated", () => {
  for (const deviceBackend of ["cuda", "rocm", "xpu"]) {
    assert.equal(
      classifyHost({ deviceBackend, budgetKnown: true }),
      "accelerated",
      deviceBackend,
    );
  }
});

test("the backends that only run the native engine are gguf-only", () => {
  for (const deviceBackend of ["mlx", "cpu"]) {
    assert.equal(
      classifyHost({ deviceBackend, budgetKnown: true }),
      "gguf-only",
      deviceBackend,
    );
  }
});

test("a Mac is gguf-only whatever backend it reports", () => {
  // Apple GPUs report as available and the backend string varies with what torch found, but no
  // Mac can place the Modular Diffusers workflow: video.py refuses the load outright.
  for (const deviceBackend of ["mlx", "cpu", "cuda", null]) {
    assert.equal(
      classifyHost({ deviceType: "mac", deviceBackend, budgetKnown: true }),
      "gguf-only",
      String(deviceBackend),
    );
  }
});

test("a host still being discovered is unknown, not CPU-only", () => {
  // The anti-flicker guarantee. The GPU hook's opening state is available:false,
  // budgetKnown:false, and reading that as CPU-only would blink every non-GGUF row out and back
  // on a real GPU host.
  assert.equal(classifyHost({ budgetKnown: false }), "unknown");
  assert.equal(
    classifyHost({ deviceBackend: "cuda", budgetKnown: false }),
    "unknown",
  );
  assert.equal(
    classifyHost({ deviceBackend: "", budgetKnown: true }),
    "unknown",
  );
});

test("an unrecognised backend is treated as new hardware, not as a CPU", () => {
  assert.equal(
    classifyHost({ deviceBackend: "tpu", budgetKnown: true }),
    "unknown",
  );
});

test("a gguf-only host drops the one pipeline the backend refuses, and nothing else", () => {
  assert.equal(
    curatedArtifactIsOfferable("MiniMaxAI/MiniMax-H3", "gguf-only"),
    false,
  );
  for (const host of ["unknown", "accelerated"] as const) {
    assert.equal(
      curatedArtifactIsOfferable("MiniMaxAI/MiniMax-H3", host),
      true,
      host,
    );
  }
});

test("a gguf-only host keeps every non-GGUF row the backend can still load", () => {
  // Each of these is a load Unsloth supports on Apple Silicon or CPU today: the diffusion
  // pipelines are device-neutral (video_capability() certifies Apple Silicon, and
  // diffusion_device.py picks MPS bfloat16 for exactly these), and the STT rows run through
  // the whisper.cpp sidecar, whose format label in the catalog is informational only.
  for (const id of [
    "unsloth/whisper-large-v3-turbo",
    "unsloth/whisper-tiny",
    "unsloth/csm-1b",
    "stabilityai/sdxl-turbo",
    "Tongyi-MAI/Z-Image-Turbo",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "Lightricks/LTX-2",
    "unsloth/MiniMax-H3-GGUF",
  ]) {
    assert.equal(curatedArtifactIsOfferable(id, "gguf-only"), true, id);
  }
});

test("the speed suffixes name the two H3 rows on an accelerated host", () => {
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "accelerated"), "Fast FP8");
  assert.equal(h3PerfSuffix("unsloth/MiniMax-H3-GGUF", "accelerated"), "Slow");
});

test("no other model claims a speed it was never measured at", () => {
  for (const id of [
    "Lightricks/LTX-2.3",
    "unsloth/LTX-2.3-GGUF",
    "MiniMaxAI/MiniMax-H3-Other",
  ]) {
    assert.equal(h3PerfSuffix(id, "accelerated"), null, id);
  }
});

test("a gguf-only or undiscovered host gets no suffix at all", () => {
  for (const host of ["gguf-only", "unknown"] as const) {
    assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", host), null, host);
    assert.equal(h3PerfSuffix("unsloth/MiniMax-H3-GGUF", host), null, host);
  }
});
