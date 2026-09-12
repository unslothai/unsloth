// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

for (const deviceType of ["mac", "cuda"]) {
  for (const native of ["pytorch", "safetensors"]) {
    test(`${deviceType}: native ${native} weights coexist with exports`, () => {
      for (const tags of [["onnx", "openvino", native], [native, "openvino", "onnx"]]) {
        assert.deepEqual(classifyUnslothSupport({
          modelId: "sentence-transformers/all-MiniLM-L6-v2",
          pipelineTag: "sentence-similarity",
          libraryName: "sentence-transformers",
          tags, deviceType,
        }), { status: "supported", reason: null });
      }
    });
  }
  for (const format of ["onnx", "openvino"]) {
    test(`${deviceType}: ${format}-only exports remain rejected`, () => {
      assert.equal(classifyUnslothSupport({ tags: [format], deviceType }).status, "unsupported");
      assert.equal(classifyUnslothSupport({
        modelId: `owner/model-${format}`, tags: [format, "safetensors"], deviceType,
      }).status, "unsupported");
    });
  }
  for (const format of ["gptq", "awq", "exl2", "coreml", "tflite", "ctranslate2"]) {
    test(`${deviceType}: native tags do not bypass ${format}`, () => {
      const result = classifyUnslothSupport({ tags: ["onnx", "safetensors", format], deviceType });
      assert.equal(result.status, "unsupported");
      assert.doesNotMatch(result.reason!, /ONNX/);
    });
  }
  test(`${deviceType}: quantization config still takes precedence`, () => {
    assert.deepEqual(classifyUnslothSupport({
      tags: ["onnx", "pytorch", "safetensors"], quantMethod: "gptq", deviceType,
    }), { status: "unsupported", reason: "Detected GPTQ quantization." });
  });
}

test("native storage tags do not make MLX weights GPU compatible", () => {
  assert.equal(classifyUnslothSupport({
    modelId: "mlx-community/model", tags: ["onnx", "safetensors"], deviceType: "cuda",
  }).status, "unsupported");
});
