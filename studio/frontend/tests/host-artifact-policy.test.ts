// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyHost,
  curatedArtifactIsOfferable,
  densePerfSuffix,
  h3PerfSuffix,
  hostIsAccelerated,
  hostRunsDenseQuant,
} from "../src/features/model-picker/components/model-selector/host-artifact-policy.ts";
import { normalizeDenseQuantSchemes } from "../src/lib/dense-quant-schemes.ts";

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

// The scheme comes from the backend, so the qualifier states the precision that will run on THIS
// host instead of a precision that happens to be true of some other one.
test("the suffix names the scheme the host reported", () => {
  assert.equal(densePerfSuffix(["fp8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["int8"]), "Fast INT8");
  // Reported best-first, so only the first entry is named.
  assert.equal(densePerfSuffix(["fp8", "int8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["int8", "fp8"]), "Fast INT8");
  // A scheme the frontend has never heard of is still the backend's answer, not ours to drop.
  assert.equal(densePerfSuffix(["nvfp4"]), "Fast NVFP4");
});

// A host that says it can run the path without naming a scheme is an older backend, which
// reports only `dense_quant_supported`. It still earns the ordering, and claims no precision.
test("a host that names no scheme keeps the bare qualifier", () => {
  assert.equal(densePerfSuffix([]), "Fast");
  assert.equal(densePerfSuffix(undefined), "Fast");
  assert.equal(densePerfSuffix(null), "Fast");
  // Blanks are not a scheme name.
  assert.equal(densePerfSuffix(["", "   "]), "Fast");
  assert.equal(densePerfSuffix(["  fp8  "]), "Fast FP8");
});

// H3's pipeline row is the one this restores: it used to read "Fast FP8", was flattened to
// "Fast", and now names whichever scheme the host actually runs.
test("the H3 pipeline row names its precision from the same scheme list", () => {
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "dense-quant", ["fp8"]), "Fast FP8");
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "dense-quant", ["int8"]), "Fast INT8");
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "accelerated", ["fp8"]), "Fast FP8");
  // The slow row is a GGUF pick and names no precision either way.
  for (const schemes of [["fp8"], ["int8"], []]) {
    assert.equal(
      h3PerfSuffix("unsloth/MiniMax-H3-GGUF", "dense-quant", schemes),
      "Slow",
      schemes.join(",") || "none",
    );
  }
  // An absent field is the old backend's answer and must not become a guessed FP8.
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "dense-quant"), "Fast");
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "dense-quant", []), "Fast");
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

// /api/system gained `dense_quant_schemes` beside the existing `dense_quant_supported`, so every
// reader has to survive a backend that predates it.
test("an absent or malformed scheme list reads as no schemes, never as a guess", () => {
  assert.deepEqual(normalizeDenseQuantSchemes(undefined), []);
  assert.deepEqual(normalizeDenseQuantSchemes(null), []);
  // Not an array: an older backend answering the shape wrong is still not an fp8 host.
  assert.deepEqual(normalizeDenseQuantSchemes("fp8" as unknown as string[]), []);
  assert.deepEqual(normalizeDenseQuantSchemes([]), []);
  // Case and padding come from the wire; the comparison downstream is lower-case.
  assert.deepEqual(normalizeDenseQuantSchemes([" FP8 ", "INT8"]), ["fp8", "int8"]);
  // Order is the backend's preference and is preserved, since only the first entry is read.
  assert.deepEqual(normalizeDenseQuantSchemes(["int8", "fp8"]), ["int8", "fp8"]);
  assert.deepEqual(normalizeDenseQuantSchemes(["", "  ", 8, null, "fp8"]), ["fp8"]);
});

test("a gguf-only or undiscovered host gets no suffix at all", () => {
  for (const host of ["gguf-only", "unknown"] as const) {
    assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", host), null, host);
    assert.equal(h3PerfSuffix("unsloth/MiniMax-H3-GGUF", host), null, host);
    // A scheme list cannot promote a host that cannot place the pipeline at all.
    assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", host, ["fp8"]), null, host);
  }
});
