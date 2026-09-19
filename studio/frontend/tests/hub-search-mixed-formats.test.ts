// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "typescript";
import test from "node:test";
import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

// Execute the production mapper without loading React or the network hooks.
const source = readFileSync(new URL("../src/features/hub/hooks/use-hub-model-search.ts", import.meta.url), "utf8");
const start = source.indexOf("function makeMapModel(");
assert.notEqual(start, -1);
const end = source.indexOf("\n}\n", start) + 2;
const makeMapModel = new Function("classifyUnslothSupport", "EMBEDDING_TAGS", "isGgufLike", "estimateSizeFromDtypes", "detectBaseModel",
  ts.transpileModule(source.slice(start, end), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText + "; return makeMapModel;",
)(classifyUnslothSupport, new Set(["sentence-transformers", "sentence-similarity"]), (id: string) => id.endsWith("-GGUF"), () => undefined, () => null);

for (const device of ["mac", "cuda"]) {
  const map = makeMapModel(false, false, "", device);
  test(`${device}: search keeps native models with optional exports`, () => {
    for (const native of ["pytorch", "safetensors"]) {
      for (const format of ["onnx", "openvino"]) {
        const raw = { name: "owner/text-model", task: "text-generation", tags: [native, format] };
        assert.equal(map(raw)?.id, raw.name);
      }
    }
  });
  test(`${device}: search still rejects export-only and quantized models`, () => {
    for (const format of ["onnx", "openvino", "gptq", "awq"]) {
      assert.equal(map({ name: "owner/model", tags: [format] }), null);
      assert.equal(map({ name: `owner/model-${format}`, tags: [format, "safetensors"] }), null);
    }
    assert.equal(map({ name: "owner/model", tags: ["onnx", "safetensors"], config: { quantization_config: { quant_method: "gptq" } } }), null);
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
      for (const format of ["onnx", "openvino"]) {
        const config = { quantization_config: { quant_method } };
        assert.equal(map({ name: `owner/model-${format}`, tags: [format], config }), null);
        assert.ok(map({ name: "owner/model", tags: [format, "safetensors"], config }));
      }
    }
  });
}
