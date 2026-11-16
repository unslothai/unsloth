// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  NPU_CATALOG_TAG,
  NPU_TAG_TEXT,
  type NpuCompat,
  npuFit,
} from "../src/features/npu/compat.ts";

// GET /api/npu/compat for FastFlowLM 1.0.12's model_list.json and the pinned Q4NX converters.
export const FLM_1_0_12 = JSON.parse(
  readFileSync(new URL("./fixtures/npu-compat-flm-1.0.12.json", import.meta.url), "utf8"),
) as NpuCompat;

const tag = (arch: string, name: string) => npuFit(FLM_1_0_12, arch, name).tag;

test("the four tags read the same everywhere", () => {
  assert.deepEqual(NPU_TAG_TEXT, {
    runs: "Runs on NPU",
    convertRuns: "Convert, then runs",
    convertOnly: "Converts, won't run",
    unsupported: "Not supported",
  });
  assert.equal(NPU_TAG_TEXT[NPU_CATALOG_TAG], "Runs on NPU");
});

test("a size FastFlowLM has kernels for and the converter takes converts, then runs", () => {
  for (const [arch, name] of [
    ["qwen3", "unsloth/Qwen3-4B-Instruct-2507-GGUF"],
    ["qwen35", "unsloth/Qwen3.5-9B-GGUF"],
    ["qwen35moe", "unsloth/Qwen3.6-35B-A3B-GGUF"],
    ["gemma4", "unsloth/gemma-4-E4B-it-GGUF"],
    ["gemma4", "unsloth/gemma-4-12b-it-GGUF"],
    ["llama", "unsloth/Llama-3.2-3B-Instruct-GGUF"],
    ["qwen2vl", "unsloth/Qwen2.5-VL-3B-Instruct-GGUF"],
    ["lfm2", "LiquidAI/LFM2-1.2B-GGUF"],
  ]) {
    assert.equal(tag(arch, name), "convertRuns", name);
  }
});

test("Qwen3 14B converts but FastFlowLM has no kernels for it", () => {
  const fit = npuFit(FLM_1_0_12, "qwen3", "unsloth/Qwen3-14B-GGUF");
  assert.equal(fit.tag, "convertOnly");
  assert.match(fit.note ?? "", /no NPU kernels for this size.*Qwen3-0\.6B/);
  assert.equal(tag("qwen35moe", "unsloth/Qwen3.5-122B-A10B-GGUF"), "convertOnly");
  assert.equal(tag("llama", "unsloth/Llama-3.3-70B-Instruct-GGUF"), "convertOnly");
});

test("LFM3 1.7B runs on the NPU through FastFlowLM's own build; the converter has no lfm3", () => {
  const fit = npuFit(FLM_1_0_12, "lfm3", "LiquidAI/LFM3-1.7B-GGUF");
  assert.equal(fit.tag, "runs");
  assert.match(fit.note ?? "", /FastFlowLM ships its own LFM3-1\.7B/);
  // So does Qwen3.8 27B, which the pinned converter refuses by name.
  assert.equal(tag("qwen35", "unsloth/Qwen3.8-27B-GGUF"), "runs");
});

test("neither side, or a size neither has, is not supported", () => {
  assert.equal(tag("qwen3moe", "unsloth/Qwen3-30B-A3B-GGUF"), "unsupported");
  assert.equal(tag("lfm3", "acme/LFM3-8B-GGUF"), "unsupported");
  assert.equal(tag("qwen35", "acme/Qwen3.5-72B-GGUF"), "unsupported");
});

test("a size must stand alone in the name: 14B is not 4B, A3B is not 3B", () => {
  assert.equal(tag("qwen3", "acme/Qwen3-14B-GGUF"), "convertOnly");
  assert.equal(tag("qwen3", "acme/Qwen3-4B-GGUF"), "convertRuns");
  assert.equal(tag("llama", "meta/Llama-3.2-3B-GGUF"), "convertRuns");
});

test("no size in the name converts and names the sizes that run", () => {
  const fit = npuFit(FLM_1_0_12, "qwen3", "acme/my-finetune-GGUF");
  assert.equal(fit.tag, "convertRuns");
  assert.match(fit.note ?? "", /Only these qwen3 sizes run on the NPU: Qwen3-0\.6B, Qwen3-1\.7B, Qwen3-4B and Qwen3-8B\./);
});

test("an unread FastFlowLM or converter, or no architecture, gives no tag", () => {
  assert.deepEqual(npuFit(null, "qwen3", "x"), { tag: null, note: null });
  assert.equal(npuFit({ ...FLM_1_0_12, fastflowlm_known: false }, "qwen3", "x").tag, null);
  assert.equal(npuFit({ ...FLM_1_0_12, converter_known: false }, "qwen3", "x").tag, null);
  assert.equal(npuFit(FLM_1_0_12, null, "x").tag, null);
});
