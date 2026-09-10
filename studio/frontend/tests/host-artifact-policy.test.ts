// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DENSE_QUANT_PRECISION_CHIP,
  classifyHost,
  curatedArtifactIsOfferable,
  denseQuantPrecisionChip,
  h3PerfSuffix,
  hostIsAccelerated,
  hostRunsDenseQuant,
} from "../src/features/model-picker/components/model-selector/host-artifact-policy.ts";

test("the backends that can place a diffusion pipeline are accelerated", () => {
  for (const deviceBackend of ["cuda", "rocm", "xpu"]) {
    assert.equal(
      hostIsAccelerated(classifyHost({ deviceBackend, budgetKnown: true })),
      true,
      deviceBackend,
    );
  }
});

// Dense quant follows backend capability rather than the backend name.
test("the dense-quant class follows the backend's capability answer, not its name", () => {
  assert.equal(
    classifyHost({ deviceBackend: "cuda", budgetKnown: true, denseQuantSupported: true }),
    "dense-quant",
  );
  assert.equal(hostRunsDenseQuant("dense-quant"), true);
  const preAmpere = classifyHost({
    deviceBackend: "cuda",
    budgetKnown: true,
    denseQuantSupported: false,
  });
  assert.equal(preAmpere, "accelerated");
  assert.equal(hostIsAccelerated(preAmpere), true);
  assert.equal(hostRunsDenseQuant(preAmpere), false);
  // Missing capability is conservative for older or unresolved backends.
  assert.equal(classifyHost({ deviceBackend: "cuda", budgetKnown: true }), "accelerated");
  for (const deviceBackend of ["rocm", "xpu"]) {
    const host = classifyHost({ deviceBackend, budgetKnown: true });
    assert.equal(host, "accelerated", deviceBackend);
    assert.equal(hostRunsDenseQuant(host), false, deviceBackend);
  }
  for (const host of ["gguf-only", "unknown"] as const) {
    assert.equal(hostRunsDenseQuant(host), false, host);
  }
  assert.equal(
    classifyHost({
      deviceType: "mac",
      deviceBackend: "cuda",
      budgetKnown: true,
      denseQuantSupported: true,
    }),
    "gguf-only",
  );
});

// The chip has to answer for the request, not only for the host.
test("the precision chip names what the request will run", () => {
  // Auto is the default and the only case where the ladder picks between the two.
  for (const auto of [undefined, null, "auto", "", "  AUTO  "]) {
    assert.equal(denseQuantPrecisionChip(auto), DENSE_QUANT_PRECISION_CHIP, String(auto));
  }
  // Precision=Off runs the checkpoint as-is, so there is no runtime precision to name.
  for (const off of ["none", "off", "None", " OFF "]) {
    assert.equal(denseQuantPrecisionChip(off), null, off);
  }
  // An explicit scheme names itself; "FP8 / INT8" would be wrong for all four.
  for (const [scheme, chip] of [
    ["fp8", "FP8"],
    ["int8", "INT8"],
    ["nvfp4", "NVFP4"],
    ["mxfp8", "MXFP8"],
  ] as const) {
    assert.equal(denseQuantPrecisionChip(scheme), chip, scheme);
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
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "accelerated"), "Fast");
  assert.equal(h3PerfSuffix("unsloth/MiniMax-H3-GGUF", "accelerated"), "Slow");
});

// The qualifier promises ordering without predicting a precision.
test("the fast row promises an ordering, not a precision", () => {
  const suffix = h3PerfSuffix("MiniMaxAI/MiniMax-H3", "accelerated");
  assert.ok(suffix, "the pipeline row still earns a qualifier");
  for (const scheme of ["fp8", "int8", "bf16", "nvfp4", "mxfp8"]) {
    assert.equal(suffix.toLowerCase().includes(scheme), false, scheme);
  }
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
