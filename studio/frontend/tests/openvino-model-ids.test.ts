// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isRecommendableFormat } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

test("OpenVINO ids are recommendable on chat-only installs, other int4/int8 repos are not", () => {
  for (const id of ["OpenVINO/Qwen3-8B-int4-ov", "unsloth/ornith-35b-uncensored-int4-ov", "a/b-ov_int8", "x/Qwen3-0.6B-openvino"]) {
    assert.equal(isRecommendableFormat(id, false, false), true, id);
  }
  for (const id of ["Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "RedHatAI/Llama-3.1-8B-Instruct-quantized.w8a8-int8", "a/overture-7b"]) {
    assert.equal(isRecommendableFormat(id, false, false), false, id);
  }
});
