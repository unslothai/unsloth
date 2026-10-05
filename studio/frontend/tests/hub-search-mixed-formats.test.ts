// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

// Execute the production mapper without loading React, the Hub client or the network hooks.
const source = readFileSync(
  new URL("../src/features/hub/hooks/use-hub-model-search.ts", import.meta.url),
  "utf8",
);
const start = source.indexOf("function makeMapModel(");
assert.notEqual(start, -1);
const end = source.indexOf("\n}\n", start) + 2;
const makeMapModel = new Function(
  "classifyUnslothSupport", "EMBEDDING_TAGS", "isGgufLike", "estimateSizeFromDtypes", "detectBaseModel",
  ts.transpileModule(source.slice(start, end), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText +
    "; return makeMapModel;",
)(
  classifyUnslothSupport,
  new Set(["feature-extraction", "sentence-transformers", "sentence-similarity", "text-embeddings-inference", "embeddings"]),
  (id: string) => id.endsWith("-GGUF"),
  () => undefined,
  () => null,
);

// Tag sets as the Hub API returns them for these repositories.
const GPT2 = ["pytorch", "tf", "jax", "tflite", "onnx", "safetensors"];
const MINILM = ["pytorch", "tf", "rust", "onnx", "safetensors", "openvino"];

for (const device of ["mac", "cuda"]) {
  const map = makeMapModel(false, false, "", device);

  test(`${device}: search keeps native models with optional exports`, () => {
    for (const raw of [
      { name: "openai-community/gpt2", tags: GPT2 },
      { name: "sentence-transformers/all-MiniLM-L6-v2", tags: MINILM },
      { name: "owner/model", tags: ["onnx", "openvino", "tflite", "coreml", "safetensors"] },
    ]) {
      assert.equal(map(raw)?.id, raw.name);
    }
  });

  test(`${device}: search still rejects export-only and quantized models`, () => {
    for (const format of ["onnx", "openvino", "tflite", "coreml", "gptq", "awq"]) {
      assert.equal(map({ name: "owner/model", tags: [format] }), null);
      assert.equal(map({ name: `owner/model-${format}`, tags: [format, "tf"] }), null);
    }
    assert.equal(
      map({ name: "owner/model", tags: ["onnx", "safetensors"], config: { quantization_config: { quant_method: "gptq" } } }),
      null,
    );
  });

  test(`${device}: discover and embeddings keep their existing exceptions`, () => {
    const discover = makeMapModel(false, true, "", device);
    assert.ok(discover({ name: "owner/model", tags: ["onnx"] }));
    assert.ok(map({ name: "owner/embedding", tags: ["sentence-transformers", "onnx"] }));
  });
}

for (const device of ["mac", "cuda"]) {
  test(`${device}: supported quantization does not bypass export rejection`, () => {
    const map = makeMapModel(false, false, "", device);
    for (const quant_method of ["bitsandbytes", "bnb", "bnb_4bit", "bnb_8bit"]) {
      for (const format of ["onnx", "openvino", "tflite", "coreml"]) {
        const config = { quantization_config: { quant_method } };
        assert.equal(map({ name: "owner/model", tags: [format], config }), null);
        assert.ok(map({ name: "openai-community/gpt2", tags: [...GPT2, format], config }));
      }
    }
  });
}
