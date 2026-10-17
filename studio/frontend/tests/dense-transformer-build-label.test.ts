// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  denseTextEncoderBuildLabel,
  denseTransformerBuildLabel,
  isNativeEngineStatus,
  isPrecisionRefusal,
  memoryRecipeValue,
} from "../src/lib/resolved-precision.ts";

test("a native sd.cpp load is not labelled BF16", () => {
  assert.equal(denseTransformerBuildLabel({ dtype: "gguf" }), "GGUF (as-is)");
  assert.equal(denseTransformerBuildLabel({ dtype: "gguf", model_kind: null }), "GGUF (as-is)");
});

test("the diffusers kinds keep their own labels", () => {
  assert.equal(denseTransformerBuildLabel({ model_kind: "gguf", dtype: "bfloat16" }), "GGUF (as-is)");
  assert.equal(
    denseTransformerBuildLabel({ model_kind: "single_file", dtype: "bfloat16" }),
    "BF16",
  );
  assert.equal(
    denseTransformerBuildLabel({ model_kind: "single_file", dtype: "float16" }),
    "FP16",
  );
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "bfloat16" }), "BF16");
});

test("a native text encoder is not labelled BF16 either", () => {
  assert.equal(denseTextEncoderBuildLabel({ dtype: "gguf" }), "As in checkpoint");
  assert.equal(denseTextEncoderBuildLabel({ dtype: "bfloat16" }), "BF16");
  assert.equal(denseTextEncoderBuildLabel({}), "BF16");
});

test("the dense label follows the dtype the pipeline actually loaded in", () => {
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "float32" }), "FP32");
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "float16" }), "FP16");
  assert.equal(
    denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "torch.bfloat16" }),
    "BF16",
  );
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline" }), "BF16");
  assert.equal(denseTextEncoderBuildLabel({ dtype: "float32" }), "FP32");
  assert.equal(denseTextEncoderBuildLabel({ dtype: "float16" }), "FP16");
  assert.equal(denseTextEncoderBuildLabel({ dtype: "gguf" }), "As in checkpoint");
});

test("the native engine is recognisable so its attention is not called SDPA", () => {
  assert.equal(isNativeEngineStatus({ dtype: "gguf" }), true);
  assert.equal(isNativeEngineStatus({ engine: "sd_cpp", dtype: "bfloat16" }), true);
  assert.equal(isNativeEngineStatus({ engine: "diffusers", dtype: "gguf" }), false);
  assert.equal(isNativeEngineStatus({ dtype: "bfloat16" }), false);
  assert.equal(isNativeEngineStatus({}), false);
});

test("the native precision refusal is classified like the diffusers one", () => {
  assert.equal(
    isPrecisionRefusal(
      "transformer_quant='fp8' could not be used: this pick runs on the native engine, which " +
        "loads a GGUF checkpoint as it is and has no torchao quantisation path.",
    ),
    true,
  );
  assert.equal(
    isPrecisionRefusal(
      "transformer_quant='fp8' and text_encoder_quant='int8' could not be used: ...",
    ),
    true,
  );
  assert.equal(isPrecisionRefusal("Failed to load model: out of memory"), false);
});

test("an absent memory mode does not become Auto", () => {
  assert.equal(memoryRecipeValue(null, "model"), "model offload");
  assert.equal(memoryRecipeValue(undefined, "sequential"), "sequential offload");
  assert.equal(memoryRecipeValue("balanced", "model"), "balanced (model offload)");
  assert.equal(memoryRecipeValue("balanced", "none"), "balanced");
  assert.equal(memoryRecipeValue(null, "none"), "");
});
