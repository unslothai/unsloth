// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  denseTextEncoderBuildLabel,
  denseTransformerBuildLabel,
  isNativeEngineStatus,
  isPrecisionRefusal,
} from "../src/lib/resolved-precision.ts";

test("a native sd.cpp load is not labelled BF16", () => {
  // The native engine reports dtype "gguf" and no model_kind, so the kind-only rule fell through
  // to the BF16 arm and mislabelled every native GGUF checkpoint -- the default CPU path.
  assert.equal(denseTransformerBuildLabel({ dtype: "gguf" }), "GGUF (as-is)");
  assert.equal(denseTransformerBuildLabel({ dtype: "gguf", model_kind: null }), "GGUF (as-is)");
});

test("the diffusers kinds keep their own labels", () => {
  assert.equal(denseTransformerBuildLabel({ model_kind: "gguf", dtype: "bfloat16" }), "GGUF (as-is)");
  assert.equal(
    denseTransformerBuildLabel({ model_kind: "single_file", dtype: "bfloat16" }),
    "As in checkpoint",
  );
  // Only a full diffusers repo is genuinely bf16.
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "bfloat16" }), "BF16");
});

test("a native text encoder is not labelled BF16 either", () => {
  // The native engine has no runtime TE quant, so its status always reports null -- and several
  // families' native companion bundles are not bf16 (FLUX.1 loads t5xxl_fp16.safetensors).
  assert.equal(denseTextEncoderBuildLabel({ dtype: "gguf" }), "As in checkpoint");
  // The diffusers path really does load the base repo's dense bf16 encoder.
  assert.equal(denseTextEncoderBuildLabel({ dtype: "bfloat16" }), "BF16");
  assert.equal(denseTextEncoderBuildLabel({}), "BF16");
});

test("the dense label follows the dtype the pipeline actually loaded in", () => {
  // BF16 is the happy path, not the only one: a CPU diffusers load reports float32, an older
  // accelerator resolves to float16, and the video loader promotes an fp16-incompatible family
  // to float32 -- all three were labelled BF16 by the panel whose job is to say what loaded.
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "float32" }), "FP32");
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "float16" }), "FP16");
  assert.equal(
    denseTransformerBuildLabel({ model_kind: "pipeline", dtype: "torch.bfloat16" }),
    "BF16",
  );
  // Unknown stays BF16: a diffusers load that reports nothing is one.
  assert.equal(denseTransformerBuildLabel({ model_kind: "pipeline" }), "BF16");
  // The text encoder reads the same dtype, and its gguf arm still wins.
  assert.equal(denseTextEncoderBuildLabel({ dtype: "float32" }), "FP32");
  assert.equal(denseTextEncoderBuildLabel({ dtype: "float16" }), "FP16");
  assert.equal(denseTextEncoderBuildLabel({ dtype: "gguf" }), "As in checkpoint");
});

test("the native engine is recognisable so its attention is not called SDPA", () => {
  // sd.cpp reports no attention backend because it has none of ours: its attention comes from
  // native flags, not the PyTorch dispatcher, so "Native SDPA" is wrong on the default CPU path.
  assert.equal(isNativeEngineStatus({ dtype: "gguf" }), true);
  assert.equal(isNativeEngineStatus({ engine: "sd_cpp", dtype: "bfloat16" }), true);
  assert.equal(isNativeEngineStatus({ engine: "diffusers", dtype: "gguf" }), false);
  assert.equal(isNativeEngineStatus({ dtype: "bfloat16" }), false);
  assert.equal(isNativeEngineStatus({}), false);
});

test("the native precision refusal is classified like the diffusers one", () => {
  // The native gate says the same "could not be used" now, so the long actionable sentence is
  // shown under the precision title instead of as a generic one-line error.
  assert.equal(
    isPrecisionRefusal(
      "transformer_quant='fp8' could not be used: this pick runs on the native engine, which " +
        "loads a GGUF checkpoint as it is and has no torchao quantisation path.",
    ),
    true,
  );
  // Both knobs refused at once puts an "and" between the clauses.
  assert.equal(
    isPrecisionRefusal(
      "transformer_quant='fp8' and text_encoder_quant='int8' could not be used: ...",
    ),
    true,
  );
  assert.equal(isPrecisionRefusal("Failed to load model: out of memory"), false);
});
