// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { ggufQuantLabel, splitQuantSuffix } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);

test("NVFP4 inside a community model name is not the file's quant", () => {
  // Mirrors utils.gguf_names: NVFP names the quant only last in the name.
  assert.equal(
    ggufQuantLabel("Qwen3.8-27B-NVFP4-MTP-HIGH.gguf"),
    "Qwen3.8-27B-NVFP4-MTP-HIGH",
  );
  assert.equal(
    ggufQuantLabel(
      "Qwen3.6-35B-A3B-uncensored-heretic-NVFP4-Experts-Only-Q8_0.gguf",
    ),
    "Q8_0",
  );
  assert.equal(ggufQuantLabel("Qwen3.8-Flash-Next-NVFP4-Q8_0.gguf"), "Q8_0");
});

test("a trailing NVFP or MXFP token is still the quant", () => {
  assert.equal(
    ggufQuantLabel("NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-00001-of-00003.gguf"),
    "NVFP4",
  );
  assert.equal(ggufQuantLabel("Model-NVFP4.gguf"), "NVFP4");
  assert.equal(ggufQuantLabel("Qwen3.6-35B-A3B-UD-MXFP4_MOE.gguf"), "UD-MXFP4_MOE");
  assert.equal(ggufQuantLabel("Some-Model-MXFP4-MTP-HIGH.gguf"), "MXFP4");
  assert.deepEqual(splitQuantSuffix("unsloth/Model-GGUF:NVFP4"), [
    "unsloth/Model-GGUF",
    "NVFP4",
  ]);
});
