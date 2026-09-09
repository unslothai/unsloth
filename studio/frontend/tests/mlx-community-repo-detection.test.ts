// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isMlxId,
  matchesFormatFilter,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

const COMMUNITY = "mlx-community/Qwen3-8B-4bit";

test("an mlx-community repo is MLX even without an -MLX suffix", () => {
  assert.equal(isMlxId(COMMUNITY), true);
  assert.equal(isMlxId("MLX-Community/gemma-3-27b-it-8bit"), true);
  assert.equal(isMlxId("mlx-community/Llama-3.2-3B-Instruct-4bit"), true);
});

test("an mlx token in the repo leaf is MLX wherever it sits", () => {
  assert.equal(isMlxId("unsloth/Qwen3-8B-MLX"), true);
  assert.equal(isMlxId("unsloth/Qwen3-MLX-8B"), true);
  assert.equal(isMlxId("unsloth/mlx_Qwen3-8B"), true);
  assert.equal(isMlxId("Qwen3-8B-4bit-mlx"), true);
});

test("the mlx token stays bounded, so a longer word is not MLX", () => {
  assert.equal(isMlxId("org/mlxray-7B"), false);
  assert.equal(isMlxId("org/Qwen-mlxtra"), false);
  assert.equal(isMlxId("mlxcommunity/Qwen3-8B-4bit"), false);
  assert.equal(isMlxId("unsloth/Qwen3-8B-GGUF"), false);
});

test("the format filter routes an mlx-community repo to MLX, not Safetensors", () => {
  assert.equal(matchesFormatFilter(COMMUNITY, false, "mlx"), true);
  assert.equal(matchesFormatFilter(COMMUNITY, false, "safetensors"), false);
  assert.equal(matchesFormatFilter(COMMUNITY, false, "gguf"), false);
  assert.equal(matchesFormatFilter(COMMUNITY, false, "all"), true);
});

// localModelIsMlx passes LocalModelInfo.id, a filesystem path: parent directories must not decide.
test("a local path is judged by its leaf on both separators", () => {
  assert.equal(isMlxId("/Users/me/models/Qwen3-8B-MLX"), true);
  assert.equal(isMlxId("C:\\Users\\me\\models\\Qwen3-8B-MLX"), true);
  assert.equal(isMlxId("/Users/me/Qwen-MLX-builds/Qwen3-8B"), false);
  assert.equal(isMlxId("C:\\Users\\me\\Qwen-MLX-builds\\Qwen3-8B"), false);
  assert.equal(isMlxId("/Users/me/Qwen3-MLX-4bit/model.gguf"), false);
  assert.equal(isMlxId("C:\\Users\\me\\Qwen3-MLX-4bit\\model.gguf"), false);
  assert.equal(isMlxId("mlx-community/Qwen3-8B-4bit"), true);
});

test("a plain safetensors repo still answers the safetensors filter", () => {
  const plain = "unsloth/Qwen3-8B";
  assert.equal(matchesFormatFilter(plain, false, "safetensors"), true);
  assert.equal(matchesFormatFilter(plain, false, "mlx"), false);
  assert.equal(
    matchesFormatFilter("org/mlxray-7B", false, "safetensors"),
    true,
  );
});
