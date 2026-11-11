// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { detectResultFormat } from "../src/features/hub/lib/format-filters.ts";
import {
  hasMlxCardTag,
  isMlxId,
  matchesFormatFilter,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";
import { readSrc } from "./helpers/kit.ts";

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

test("a model-card mlx tag or library marks a repo whose name does not", () => {
  const unnamed = "FluidVoice-Labs/Qwen3.5-9B-4bit";
  assert.equal(hasMlxCardTag({ tags: ["safetensors", "MLX"] }), true);
  assert.equal(hasMlxCardTag({ libraryName: "MLX" }), true);
  assert.equal(
    hasMlxCardTag({ tags: ["safetensors"], libraryName: "transformers" }),
    false,
  );
  assert.equal(matchesFormatFilter(unnamed, false, "mlx", true), true);
  assert.equal(matchesFormatFilter(unnamed, false, "safetensors", true), false);
  assert.equal(matchesFormatFilter(unnamed, false, "mlx", false), false);
  const taggedGguf = "org/Model-GGUF";
  assert.equal(matchesFormatFilter(taggedGguf, true, "mlx", true), false);
  assert.equal(matchesFormatFilter(taggedGguf, true, "gguf", true), true);
});

test("Hub search rows are MLX by name or by model-card tag", () => {
  const row = (id: string, tags: string[], libraryName?: string) =>
    detectResultFormat({ id, isGguf: false, tags, libraryName });
  assert.equal(
    row("unsloth/gemma-4-E2B-it-UD-MLX-4bit", ["safetensors"]),
    "mlx",
  );
  assert.equal(row("FluidVoice-Labs/Qwen3.5-9B-4bit", ["mlx"]), "mlx");
  assert.equal(row("org/Qwen3-8B", [], "mlx"), "mlx");
  assert.equal(
    row("Qwen/Qwen3-8B", ["safetensors"], "transformers"),
    "safetensors",
  );
  assert.equal(
    detectResultFormat({
      id: "org/Model-MLX-GGUF",
      isGguf: true,
      tags: ["mlx"],
    }),
    "gguf",
  );
});

test("the picker and Hub inspector pass model-card tags to the shared rule", () => {
  const picker = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(
    picker,
    /r\.isGguf,\s*formatFilter,\s*hasMlxCardTag\(r\),?\s*\)/,
  );
  assert.match(picker, /isMlxId\(r\.id, hasMlxCardTag\(r\)\)/);
  assert.match(picker, /isMlxId\(c\.repo_id, cachedRepoHasMlxTag\(c\)\)/);
  assert.match(picker, /formatFilter,\s*cachedRepoHasMlxTag\(c\),?\s*\)/);
  assert.match(picker, /isMlxId\(id, taggedMlxIds\.has\(key\)\)/);
  const idOnlyFilters = picker.match(
    /isKnownGgufRepo\(id\),\s*formatFilter,\s*taggedMlxIds\.has\(id\.toLowerCase\(\)\),/g,
  );
  assert.equal(idOnlyFilters?.length, 2);
  const inspector = readSrc("features/hub/catalog/model-inspector.tsx");
  assert.match(
    inspector,
    /isMlxId\(model\.hubRepoId \?\? "", hasMlxCardTag\(model\)\)/,
  );
});
