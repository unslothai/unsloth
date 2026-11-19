// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  catalogToModelOptions,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import {
  classifyHost,
  hostIsGgufOnly,
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

test("the backends that only run the native engine are gguf-only, a plain CPU cpu-only", () => {
  assert.equal(classifyHost({ deviceBackend: "mlx", budgetKnown: true }), "gguf-only");
  assert.equal(classifyHost({ deviceBackend: "cpu", budgetKnown: true }), "cpu-only");
  assert.equal(hostIsGgufOnly("cpu-only"), true);
  assert.equal(hostIsAccelerated("cpu-only"), false);
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
  assert.equal(hostOffersDensePrecision("cpu-only"), false);
});

test("a CPU-only host offers only the GGUF rows of a family that has a GGUF / native route", () => {
  const cpu = new Set(catalogToModelOptions(IMAGE_CATALOG, "cpu-only").map((o) => o.id));
  const gpu = new Set(catalogToModelOptions(IMAGE_CATALOG, "dense-quant", ["fp8"]).map((o) => o.id));
  const mac = new Set(catalogToModelOptions(IMAGE_CATALOG, "gguf-only").map((o) => o.id));
  // FLUX.3 Klein: the bf16 pipeline ("Fast FP8" on a GPU, 20.3 GB of bf16 on CPU) goes, the GGUF stays.
  assert.equal(gpu.has("black-forest-labs/flux-3-image-klein"), true);
  assert.equal(cpu.has("black-forest-labs/flux-3-image-klein"), false);
  assert.equal(cpu.has("unsloth/flux-3-image-klein-GGUF"), true);
  // Every image group with a GGUF row (or a native route, Krea 2) offers nothing else on CPU.
  for (const group of IMAGE_CATALOG) {
    const native = group.nativeGgufRoute || group.artifacts.some((a) => a.format === "gguf");
    for (const artifact of group.artifacts) {
      const want = !native || artifact.format === "gguf";
      assert.equal(cpu.has(artifact.repoId), want && artifact.repoId !== "MiniMaxAI/MiniMax-H3-Image", artifact.repoId);
    }
  }
  // Families with no GGUF route keep their rows on CPU.
  for (const id of [
    "unsloth/sdxl-turbo",
    "unsloth/stable-diffusion-xl-base-1.0",
    "unsloth/Lumina-Image-2.0",
    "unsloth/HiDream-I1-Full",
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    "ideogram-ai/ideogram-4-fp8",
  ]) {
    assert.equal(cpu.has(id), true, id);
  }
  assert.equal(cpu.has("unsloth/Krea-2-Turbo"), false);
  // A Mac (gguf-only, not cpu-only) and a GPU keep today's rows.
  assert.equal(mac.has("black-forest-labs/flux-3-image-klein"), true);
  assert.equal(mac.has("unsloth/Krea-2-Turbo"), true);
  assert.equal(gpu.has("unsloth/Krea-2-Turbo"), true);
});

test("a CPU-only host offers only the GGUF rows of a video family with a GGUF row", () => {
  const cpu = new Set(catalogToModelOptions(VIDEO_CATALOG, "cpu-only").map((o) => o.id));
  for (const id of ["Lightricks/LTX-2.5-Diffusers", "Lightricks/LTX-2.3", "MiniMaxAI/MiniMax-H3"]) {
    assert.equal(cpu.has(id), false, id);
  }
  for (const id of ["unsloth/LTX-2.5-GGUF", "unsloth/LTX-2.3-GGUF", "unsloth/MiniMax-H3-GGUF"]) {
    assert.equal(cpu.has(id), true, id);
  }
  // No GGUF row: the pipeline stays.
  assert.equal(cpu.has("Wan-AI/Wan2.2-TI2V-5B-Diffusers"), true);
});

test("the CPU-only rule needs the group: an id alone keeps today's answer", () => {
  assert.equal(curatedArtifactIsOfferable("black-forest-labs/flux-3-image-klein", "cpu-only"), true);
  assert.equal(curatedArtifactIsOfferable("MiniMaxAI/MiniMax-H3", "cpu-only"), false);
});
