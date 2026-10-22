// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyHost,
  curatedArtifactIsOfferable,
  densePerfSuffix,
  ggufPerfSuffix,
  h3PerfSuffix,
  hostIsAccelerated,
  hostOffersDensePrecision,
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
  assert.equal(classifyHost({ deviceBackend: "cuda", budgetKnown: true }), "accelerated");
  for (const deviceBackend of ["rocm", "xpu"]) {
    const host = classifyHost({ deviceBackend, budgetKnown: true });
    assert.equal(host, "accelerated", deviceBackend);
    assert.equal(hostRunsDenseQuant(host), false, deviceBackend);
  }
  for (const host of ["gguf-only", "unknown"] as const) {
    assert.equal(hostRunsDenseQuant(host), false, host);
  }
  // mac + cuda means a browser-derived Mac on a remote host.
  for (const deviceBackend of ["mlx", "cpu"]) {
    assert.equal(
      classifyHost({
        deviceType: "mac",
        deviceBackend,
        budgetKnown: true,
        denseQuantSupported: true,
      }),
      "gguf-only",
      deviceBackend,
    );
  }
  assert.equal(
    classifyHost({
      deviceType: "mac",
      deviceBackend: "cuda",
      budgetKnown: true,
      denseQuantSupported: true,
    }),
    "dense-quant",
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

test("a resolved accelerated backend outranks a browser-derived Mac", () => {
  for (const deviceBackend of ["cuda", "rocm", "xpu"]) {
    assert.equal(
      classifyHost({ deviceType: "mac", deviceBackend, budgetKnown: true }),
      "accelerated",
      deviceBackend,
    );
  }
});

test("a Mac is gguf-only whatever backend it reports", () => {
  // No Mac can place the Modular Diffusers workflow: video.py refuses the load.
  for (const deviceBackend of ["mlx", "cpu", null]) {
    assert.equal(
      classifyHost({ deviceType: "mac", deviceBackend, budgetKnown: true }),
      "gguf-only",
      String(deviceBackend),
    );
  }
});

test("a host still being discovered is unknown, not CPU-only", () => {
  // The GPU hook opens with available:false; reading that as CPU-only flickers rows on GPU hosts.
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
  // Diffusion pipelines are device-neutral and STT rows run through the whisper.cpp sidecar.
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

test("every dense 8-bit scheme shares the one FP8 label", () => {
  assert.equal(densePerfSuffix(["fp8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["int8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["mxfp8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["fp8", "int8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["int8", "fp8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["nvfp4"]), "Fast NVFP4");
});

test("a host that names no scheme keeps the bare qualifier", () => {
  assert.equal(densePerfSuffix([]), "Fast");
  assert.equal(densePerfSuffix(undefined), "Fast");
  assert.equal(densePerfSuffix(null), "Fast");
  assert.equal(densePerfSuffix(["", "   "]), "Fast");
  assert.equal(densePerfSuffix(["  fp8  "]), "Fast FP8");
});

test("the H3 pipeline row names its precision from the same scheme list", () => {
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "dense-quant", ["fp8"]), "Fast FP8");
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "dense-quant", ["int8"]), "Fast FP8");
  assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", "accelerated", ["fp8"]), "Fast FP8");
  for (const schemes of [["fp8"], ["int8"], []]) {
    assert.equal(
      h3PerfSuffix("unsloth/MiniMax-H3-GGUF", "dense-quant", schemes),
      "Slow",
      schemes.join(",") || "none",
    );
  }
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

test("an absent or malformed scheme list reads as no schemes, never as a guess", () => {
  assert.deepEqual(normalizeDenseQuantSchemes(undefined), []);
  assert.deepEqual(normalizeDenseQuantSchemes(null), []);
  assert.deepEqual(normalizeDenseQuantSchemes("fp8" as unknown as string[]), []);
  assert.deepEqual(normalizeDenseQuantSchemes([]), []);
  assert.deepEqual(normalizeDenseQuantSchemes([" FP8 ", "INT8"]), ["fp8", "int8"]);
  assert.deepEqual(normalizeDenseQuantSchemes(["int8", "fp8"]), ["int8", "fp8"]);
  assert.deepEqual(normalizeDenseQuantSchemes(["", "  ", 8, null, "fp8"]), ["fp8"]);
});

test("a gguf-only or undiscovered host gets no suffix at all", () => {
  for (const host of ["gguf-only", "unknown"] as const) {
    assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", host), null, host);
    assert.equal(h3PerfSuffix("unsloth/MiniMax-H3-GGUF", host), null, host);
    assert.equal(h3PerfSuffix("MiniMaxAI/MiniMax-H3", host, ["fp8"]), null, host);
    assert.equal(ggufPerfSuffix(host), null, host);
  }
});

test("a GGUF row is the slow one wherever a dense row can run beside it", () => {
  assert.equal(ggufPerfSuffix("accelerated"), "Slow");
  assert.equal(ggufPerfSuffix("dense-quant"), "Slow");
});

test("the dense precisions are offered exactly where they can run", () => {
  assert.equal(hostOffersDensePrecision("dense-quant"), true);
  assert.equal(hostOffersDensePrecision("accelerated"), true);
  assert.equal(hostOffersDensePrecision("unknown"), true);
  assert.equal(hostOffersDensePrecision("gguf-only"), false);
});
