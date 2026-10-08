// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";
import type { EngineStatus } from "../src/features/model-picker/api/engines.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { vllmHostSupported } = await import("../src/features/model-picker/api/engines.ts");

// #11728: unsloth/Qwen3.8-27B-NVFP4 was labelled "May not be supported" on a host whose optional
// vLLM engine loads it. Status stays "unsupported" (the Default engine and training still cannot
// load it); supportedIn tells the hub to stop calling it unsupported.
const NVFP4 = {
  modelId: "unsloth/Qwen3.8-27B-NVFP4",
  tags: ["safetensors", "qwen3_5", "unsloth", "compressed-tensors"],
  quantMethod: "compressed-tensors",
  deviceType: "cuda",
};

test("without vLLM every caller sees exactly what it saw before", () => {
  for (const vllmAvailable of [undefined, false]) {
    assert.deepEqual(classifyUnslothSupport({ ...NVFP4, vllmAvailable }), {
      status: "unsupported",
      reason: "Detected compressed-tensors quantization.",
    });
    assert.deepEqual(
      classifyUnslothSupport({ modelId: "TheBloke/Llama-2-7B-AWQ", tags: ["awq"], deviceType: "cuda", vllmAvailable }),
      { status: "unsupported", reason: "Detected AWQ quantization." },
    );
  }
});

test("with vLLM, compressed-tensors, AWQ and GPTQ are marked as vLLM-runnable", () => {
  assert.deepEqual(classifyUnslothSupport({ ...NVFP4, vllmAvailable: true }), {
    status: "unsupported",
    reason: "Detected compressed-tensors quantization.",
    supportedIn: "vllm",
  });
  for (const quantMethod of ["awq", "GPTQ", " compressed-tensors "]) {
    const support = classifyUnslothSupport({ modelId: "owner/model", quantMethod, deviceType: "cuda", vllmAvailable: true });
    assert.equal(support.status, "unsupported");
    assert.equal(support.supportedIn, "vllm", quantMethod);
  }
  // Format found only by tag or by name, with no quantization_config to read.
  for (const input of [
    { modelId: "owner/model", tags: ["gptq"] },
    { modelId: "owner/model", tags: ["auto-gptq"] },
    { modelId: "TheBloke/Llama-2-7B-AWQ" },
  ]) {
    assert.equal(classifyUnslothSupport({ ...input, deviceType: "cuda", vllmAvailable: true }).supportedIn, "vllm");
  }
});

test("vLLM does not excuse any other reason a model cannot run in chat", () => {
  for (const quantMethod of ["hqq", "exl2", "aqlm", "quark"]) {
    assert.equal(
      classifyUnslothSupport({ modelId: "owner/model", quantMethod, deviceType: "cuda", vllmAvailable: true }).supportedIn,
      undefined,
      quantMethod,
    );
  }
  // Anything else that rejects the repo (a task, a library, another format) keeps today's answer.
  for (const extra of [
    { pipelineTag: "text-to-image" },
    { tags: ["onnx"] },
    { tags: ["diffusers"] },
    { modelId: "owner/model-exl2" },
  ]) {
    assert.deepEqual(
      classifyUnslothSupport({ ...NVFP4, ...extra, vllmAvailable: true }),
      classifyUnslothSupport({ ...NVFP4, ...extra }),
      JSON.stringify(extra),
    );
  }
  // GGUF and plain checkpoints are untouched.
  assert.deepEqual(
    classifyUnslothSupport({ modelId: "unsloth/Qwen3.8-27B-GGUF", tags: ["gguf"], quantMethod: "compressed-tensors", deviceType: "cuda", vllmAvailable: true }),
    { status: "supported", reason: null },
  );
  assert.deepEqual(
    classifyUnslothSupport({ modelId: "unsloth/Qwen3.8-27B", tags: ["safetensors"], deviceType: "cuda", vllmAvailable: true }),
    { status: "supported", reason: null },
  );
});

function engine(overrides: Partial<EngineStatus>): EngineStatus {
  return {
    engine: "vllm",
    version: "0.30.0",
    installed_version: null,
    installed: false,
    in_use: false,
    current: false,
    can_rollback: false,
    unsupported_reason: null,
    job: { state: "idle", phase: null, message: "" },
    ...overrides,
  };
}

test("a host counts once the backend says vLLM can run there, installed or not", () => {
  assert.equal(vllmHostSupported([engine({})]), true);
  assert.equal(vllmHostSupported([engine({ installed: true, installed_version: "0.30.0", current: true })]), true);
  assert.equal(vllmHostSupported([engine({ unsupported_reason: "Checking for a supported NVIDIA GPU." })]), false);
  assert.equal(vllmHostSupported([engine({ unsupported_reason: "Managed engines currently require Linux x86_64 or Windows x64." })]), false);
  assert.equal(vllmHostSupported([engine({ engine: "sglang" })]), false);
  assert.equal(vllmHostSupported([]), false);
});
