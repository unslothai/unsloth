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
  // A Mac stays gguf-only however its capability answers, tested through the only backends a
  // Mac can report: hardware.py yields mlx, or cpu when the MLX stack is absent. "cuda" no
  // longer reaches this branch, since mac + cuda now means a remote CUDA host.
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
  // deviceType is the BROWSER's platform until an authenticated reply carries device_type, so a
  // Mac driving a remote CUDA server must not classify that Linux host as gguf-only. Safe because
  // macOS never resolves to one of these: it reports mps, mlx or cpu.
  for (const deviceBackend of ["cuda", "rocm", "xpu"]) {
    assert.equal(
      classifyHost({ deviceType: "mac", deviceBackend, budgetKnown: true }),
      "accelerated",
      deviceBackend,
    );
  }
});

test("a Mac is gguf-only whatever backend it reports", () => {
  // Apple GPUs report as available and the backend string varies with what torch found, but no
  // Mac can place the Modular Diffusers workflow: video.py refuses the load outright.
  for (const deviceBackend of ["mlx", "cpu", null]) {
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

test("every dense 8-bit scheme shares the one FP8 label", () => {
  assert.equal(densePerfSuffix(["fp8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["int8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["mxfp8"]), "Fast FP8");
  // Reordering the ladder must not rename the row: the label is the tier, and the resolved record
  // names the scheme that actually loaded.
  assert.equal(densePerfSuffix(["fp8", "int8"]), "Fast FP8");
  assert.equal(densePerfSuffix(["int8", "fp8"]), "Fast FP8");
  // A 4-bit scheme is a different quality tier and keeps its own name.
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
  // An older backend reports nothing; keep the controls rather than remove ones it can honour.
  assert.equal(hostOffersDensePrecision("unknown"), true);
  assert.equal(hostOffersDensePrecision("gguf-only"), false);
});
