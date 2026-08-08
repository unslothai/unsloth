// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { denseTransformerBuildLabel } from "../src/lib/resolved-precision.ts";

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
