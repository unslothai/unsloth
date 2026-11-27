// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The desktop app ships its own frontend and may adopt an older backend, so these
// shapes are real; newer backend fields must not break it either.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SttStatusResponse,
  describeDiffusionStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  mergeLoadedModels,
  sttEngineStatus,
} from "../src/features/loaded-models/loaded-models-sources.ts";

// settled() collapses any failed read (404, 401, 500, timeout) to null.
const UNREACHABLE = null;

test("a backend with no video route at all still lists the other runtimes", () => {
  const rows = mergeLoadedModels([
    describeInferenceStatus({
      active_model: "unsloth/Qwen3-4B-GGUF",
      loaded: ["unsloth/Qwen3-4B-GGUF"],
      is_gguf: true,
      gguf_variant: "Q4_K_M",
    } as never),
    describeDiffusionStatus(UNREACHABLE),
    describeVideoStatus(UNREACHABLE),
    describeSttStatus(UNREACHABLE),
  ]);
  assert.equal(rows.length, 1, "one dead runtime must not blank the others");
  assert.equal(rows[0].name, "unsloth/Qwen3-4B-GGUF");
});

test("every runtime unreachable is an empty list, never a crash", () => {
  assert.deepEqual(
    mergeLoadedModels([
      describeInferenceStatus(UNREACHABLE),
      describeDiffusionStatus(UNREACHABLE),
      describeVideoStatus(UNREACHABLE),
      describeSttStatus(UNREACHABLE),
    ]),
    [],
  );
});

test("a pre-split dictation backend reports through the legacy fields", () => {
  const rows = describeSttStatus({
    loaded_model: "large-v3",
    device: "cuda",
  } as SttStatusResponse);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].name, "large-v3");
  assert.equal(rows[0].sttEngine, "transformers");
  assert.equal(rows[0].detail, "Transformers · cuda");
});

test("a current backend does not double the dictation row", () => {
  const rows = describeSttStatus({
    loaded_model: "large-v3",
    device: "cuda",
    transformers: { loaded_model: "large-v3", device: "cuda" },
  } as SttStatusResponse);
  assert.equal(rows.length, 1);
});

test("the legacy fallback is transformers-only", () => {
  // Top-level fields belong to the Transformers sidecar only.
  const status = { loaded_model: "large-v3", device: "cuda" } as SttStatusResponse;
  assert.equal(sttEngineStatus(status, "transformers")?.loaded_model, "large-v3");
  assert.equal(sttEngineStatus(status, "mtmd"), null);
  assert.equal(sttEngineStatus(status, "gguf"), null);
});

test("a dictation backend without the mtmd engine skips it", () => {
  const rows = describeSttStatus({
    transformers: { loaded_model: null, device: null },
    gguf: { loaded_model: "ggml-base.en", device: "whisper.cpp" },
  } as SttStatusResponse);
  assert.deepEqual(
    rows.map((row) => row.sttEngine),
    ["gguf"],
  );
});

test("an engine block explicitly nulled is skipped, not read as legacy", () => {
  const rows = describeSttStatus({
    loaded_model: "large-v3",
    device: "cuda",
    transformers: null,
  } as SttStatusResponse);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].name, "large-v3");
});

test("a chat payload missing every optional field still renders", () => {
  const rows = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
  } as never);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].detail, "Transformers", "the ladder needs no flags");
  assert.equal(rows[0].kind, "text");
});

test("a diffusion payload missing dtype, device and family still renders", () => {
  const rows = describeDiffusionStatus({
    loaded: true,
    repo_id: "black-forest-labs/FLUX.1-dev",
  } as never);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].detail, "", "no parts is an empty line, not a stray dot");
  assert.equal(rows[0].name, "black-forest-labs/FLUX.1-dev");
});

test("undefined and null are the same absence", () => {
  const withNulls = describeVideoStatus({
    loaded: true,
    repo_id: "Wan-AI/Wan2.2-T2V-A14B",
    family: null,
    device: null,
    dtype: null,
    transformer_quant: null,
  } as never);
  const withUndefined = describeVideoStatus({
    loaded: true,
    repo_id: "Wan-AI/Wan2.2-T2V-A14B",
  } as never);
  assert.deepEqual(withNulls, withUndefined);
});

test("empty strings are dropped rather than printed as separators", () => {
  const rows = describeDiffusionStatus({
    loaded: true,
    repo_id: "x/y",
    family: "",
    device: "cuda",
    dtype: "",
  } as never);
  assert.equal(rows[0].detail, "cuda");
});

test("fields a future backend adds are ignored, not rendered", () => {
  const rows = describeDiffusionStatus({
    loaded: true,
    repo_id: "x/y",
    family: "flux",
    device: "cuda",
    dtype: "bfloat16",
    some_future_field: "should not appear",
    nested: { also: "ignored" },
  } as never);
  assert.equal(rows[0].detail, "flux · BF16 · cuda");
});

test("a backend with no gguf_variant field still reports the compute dtype", () => {
  const rows = describeDiffusionStatus({
    loaded: true,
    repo_id: "unsloth/Z-Image-Turbo-GGUF",
    family: "z-image",
    model_kind: "gguf",
    dtype: "bfloat16",
    device: "cuda",
  } as never);
  assert.equal(rows[0].detail, "z-image · GGUF · BF16 · cuda");
});

test("an unrecognised precision is passed through rather than dropped", () => {
  const rows = describeVideoStatus({
    loaded: true,
    repo_id: "x/y",
    family: "wan",
    device: "cuda",
    transformer_quant: "nvfp4",
  } as never);
  assert.equal(rows[0].detail, "wan · NVFP4 · cuda");
});

test("a chat runtime caching past the active model marks the extras inactive", () => {
  const rows = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
    loaded: ["unsloth/Qwen3-4B", "unsloth/Llama-3.2-3B"],
  } as never);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].inactive, undefined);
  assert.equal(rows[1].inactive, true);
  assert.equal(rows[1].detail, "Still in memory");
});

test("a duplicate in the loaded list is not listed twice", () => {
  const rows = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
    loaded: ["unsloth/Qwen3-4B", "unsloth/Llama-3.2-3B", "unsloth/Llama-3.2-3B"],
  } as never);
  assert.equal(rows.length, 2);
});

test("the same row arriving from two sources is merged once", () => {
  const row = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
  } as never);
  assert.equal(mergeLoadedModels([row, row]).length, 1);
});
