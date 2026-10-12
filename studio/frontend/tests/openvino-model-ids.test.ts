// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { intelIntRecommendations, isRecommendableFormat } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

const ids = [
  "org/llama",
  "org/llama-ov_int8",
  "org/llama-ov_int4",
  "org/qwen-ov_int8",
  "org/phi",
];

test("on xpu the INT variant is recommended when its source is also on disk, INT4 first", () => {
  assert.deepEqual(
    [...intelIntRecommendations(ids, "xpu")],
    [["org/llama-ov_int4", "org/llama"]],
  );
});

test("INT8 is recommended when it is the only conversion", () => {
  assert.deepEqual(
    [...intelIntRecommendations(["a/b", "a/b-ov_int8"], " XPU ")],
    [["a/b-ov_int8", "a/b"]],
  );
});

test("no recommendation off Intel or without the unconverted source", () => {
  for (const backend of ["cuda", "rocm", "mlx", "cpu", "", null, undefined]) {
    assert.equal(intelIntRecommendations(ids, backend).size, 0);
  }
  assert.equal(intelIntRecommendations(["org/qwen-ov_int8"], "xpu").size, 0);
});

test("OpenVINO ids are recommendable on chat-only installs, other int4/int8 repos are not", () => {
  for (const id of ["OpenVINO/Qwen3-8B-int4-ov", "unsloth/ornith-35b-uncensored-int4-ov", "a/b-ov_int8", "x/Qwen3-0.6B-openvino"]) {
    assert.equal(isRecommendableFormat(id, false, false), true, id);
  }
  for (const id of ["Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "RedHatAI/Llama-3.1-8B-Instruct-quantized.w8a8-int8", "a/overture-7b"]) {
    assert.equal(isRecommendableFormat(id, false, false), false, id);
  }
});
