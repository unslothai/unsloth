// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

// #7997: transformers writes quant_method "auto-round" (QuantizationMethod.AUTOROUND).
const AUTOROUND = {
  modelId: "Lorbus/Qwen3.6-27B-int4-AutoRound",
  pipelineTag: "image-text-to-text",
  tags: ["transformers", "safetensors", "qwen3_5", "autoround", "int4", "auto-round", "4-bit"],
  libraryName: "transformers",
  quantMethod: "auto-round",
};

test("AutoRound checkpoints are not offered to the Default engine", () => {
  for (const deviceType of ["cuda", "rocm", "mac", "cpu"]) {
    assert.deepEqual(classifyUnslothSupport({ ...AUTOROUND, deviceType }), {
      status: "unsupported",
      reason: "Detected AutoRound quantization.",
    });
  }
  for (const quantMethod of ["auto-round", "AUTO-ROUND", "auto_round", "autoround"]) {
    assert.equal(classifyUnslothSupport({ modelId: "owner/model", quantMethod }).status, "unsupported", quantMethod);
  }
});

test("with vLLM, AutoRound is marked as vLLM-runnable", () => {
  assert.deepEqual(classifyUnslothSupport({ ...AUTOROUND, deviceType: "cuda", vllmAvailable: true }), {
    status: "unsupported",
    reason: "Detected AutoRound quantization.",
    supportedIn: "vllm",
  });
  // vLLM recognises only the spelling transformers writes.
  assert.equal(
    classifyUnslothSupport({ modelId: "owner/model", quantMethod: "auto_round", vllmAvailable: true }).supportedIn,
    undefined,
  );
});

test("AutoRound GGUF exports still run through llama.cpp", () => {
  assert.equal(
    classifyUnslothSupport({ modelId: "owner/model-GGUF", tags: ["gguf"], quantMethod: "auto-round" }).status,
    "supported",
  );
});
