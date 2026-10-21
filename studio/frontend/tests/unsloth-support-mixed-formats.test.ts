// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

// Tag sets as the Hub API returns them for these repositories.
const GPT2 = ["pytorch", "tf", "jax", "tflite", "onnx", "safetensors"];
const MINILM = ["pytorch", "tf", "rust", "onnx", "safetensors", "openvino"];

for (const deviceType of ["mac", "cuda"]) {
  for (const native of ["pytorch", "safetensors"]) {
    test(`${deviceType}: native ${native} weights coexist with export tags`, () => {
      for (const exportTag of ["onnx", "openvino", "tflite", "coreml"]) {
        const tags = [...GPT2, native, exportTag];
        assert.deepEqual(
          classifyUnslothSupport({ modelId: "openai-community/gpt2", tags, deviceType }),
          { status: "supported", reason: null },
        );
        assert.deepEqual(
          classifyUnslothSupport({
            modelId: "sentence-transformers/all-MiniLM-L6-v2",
            pipelineTag: "sentence-similarity",
            tags: [...MINILM, exportTag],
            deviceType,
          }),
          { status: "supported", reason: null },
        );
      }
    });
  }

  for (const format of ["onnx", "openvino", "tflite", "coreml"]) {
    test(`${deviceType}: export-only ${format} stays rejected`, () => {
      assert.equal(classifyUnslothSupport({ tags: [format], deviceType }).status, "unsupported");
      assert.equal(
        classifyUnslothSupport({ modelId: `owner/model-${format}`, tags: [format], deviceType }).status,
        "unsupported",
      );
      // A name alone must not excuse a repo with no native weights.
      assert.equal(
        classifyUnslothSupport({ modelId: "owner/model-onnx", tags: ["onnx", "openvino"], deviceType }).status,
        "unsupported",
      );
    });
  }

  for (const format of ["gptq", "awq", "exl2"]) {
    test(`${deviceType}: native tags do not bypass ${format}`, () => {
      const result = classifyUnslothSupport({
        tags: [...GPT2, format],
        deviceType,
      });
      assert.equal(result.status, "unsupported");
      assert.doesNotMatch(result.reason!, /ONNX/);
    });
  }

  test(`${deviceType}: quantization config still takes precedence`, () => {
    assert.deepEqual(
      classifyUnslothSupport({ tags: ["onnx", "pytorch", "safetensors"], quantMethod: "gptq", deviceType }),
      { status: "unsupported", reason: "Detected GPTQ quantization." },
    );
  });

  test(`${deviceType}: supported quantization does not excuse export-only repos`, () => {
    for (const quantMethod of ["bitsandbytes", "bnb", "bnb_4bit", "bnb_8bit"]) {
      assert.equal(
        classifyUnslothSupport({ modelId: "owner/model-onnx", tags: ["onnx"], quantMethod, deviceType }).status,
        "unsupported",
      );
      assert.equal(
        classifyUnslothSupport({ tags: ["onnx", "safetensors"], quantMethod, deviceType }).status,
        "supported",
      );
    }
  });
}

test("native storage tags do not make MLX weights GPU compatible", () => {
  assert.equal(
    classifyUnslothSupport({
      modelId: "mlx-community/model", tags: ["onnx", "safetensors"], deviceType: "cuda",
    }).status,
    "unsupported",
  );
});

for (const format of ["gptq", "awq", "exl2"]) {
  test(`Mac-compatible MLX tags do not hide ${format}`, () => {
    for (const tags of [["safetensors", "onnx", "mlx", format], [format, "mlx", "safetensors", "onnx"]]) {
      assert.equal(classifyUnslothSupport({ tags, deviceType: "mac" }).status, "unsupported");
    }
    assert.equal(
      classifyUnslothSupport({ modelId: `owner/model-${format}`, tags: ["safetensors", "onnx", "mlx"], deviceType: "mac" }).status,
      "unsupported",
    );
  });
}

test("MLX remains compatible with Mac for a repo with native weights and exports", () => {
  for (const tags of [
    ["safetensors", "onnx", "mlx"],
    ["safetensors", "onnx", "mlx", "coreml", "tflite"],
  ]) {
    assert.equal(classifyUnslothSupport({ tags, deviceType: "mac" }).status, "supported");
  }
});
test("MLX remains compatible with Mac without an unsupported format", () => {
  assert.equal(classifyUnslothSupport({ tags: ["safetensors", "onnx", "mlx"], deviceType: "mac" }).status, "supported");
});
