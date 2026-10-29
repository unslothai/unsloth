// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat reads the shared device inventory; this pins its mapping against each endpoint.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { chatLocalModelOptions } =
  await import("../src/features/chat/local-model-options.ts");

const { localGgufKindFor } =
  await import("../src/features/model-picker/components/model-selector/local-gguf-policy.ts");

function row(over: Record<string, unknown> = {}) {
  return {
    id: "/models/demo",
    // biome-ignore lint/style/useNamingConvention: API schema
    display_name: "demo",
    path: "/models/demo",
    source: "models_dir",
    // biome-ignore lint/style/useNamingConvention: API schema
    updated_at: 1,
    ...over,
  } as never;
}

test("Ollama rows are offered under their own label", () => {
  // `ollama-manifest:` ids are loadable as-is via materialize_ollama_model_ref.
  const options = chatLocalModelOptions([
    row({
      id: "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fllama3",
      display_name: "llama3-GGUF-q4",
      source: "ollama",
      model_format: "gguf",
    }),
  ]);
  assert.equal(options.length, 1);
  const option = options[0];
  assert.ok(option);
  assert.equal(option.name, "llama3-GGUF-q4");
  assert.equal(option.baseModel, "Ollama");
  assert.equal(option.isGguf, true);
  assert.equal(option.isDirectGguf, true);
  assert.equal(localGgufKindFor(option, true), "direct");
});

test("Hermes downloads are offered under their own label", () => {
  const options = chatLocalModelOptions([
    row({
      id: "/home/u/.hermes/models/Qwen3.8-27B-UD-Q4_K_M.gguf",
      path: "/home/u/.hermes/models/Qwen3.8-27B-UD-Q4_K_M.gguf",
      display_name: "Qwen3.8-27B-UD-Q4_K_M",
      source: "hermes",
      model_format: "gguf",
    }),
  ]);
  assert.equal(options.length, 1);
  const option = options[0];
  assert.ok(option);
  assert.equal(option.name, "Qwen3.8-27B-UD-Q4_K_M");
  assert.equal(option.baseModel, "Hermes");
  assert.equal(option.isGguf, true);
  assert.equal(option.isDirectGguf, true);
});

test("a directory with two weight formats yields one option per load id", () => {
  // The inventory keys on (format, path), so one id can arrive as two rows.
  const options = chatLocalModelOptions([
    row({ model_format: "gguf", inventory_id: "models_dir:gguf:/models/demo" }),
    row({
      model_format: "safetensors",
      inventory_id: "models_dir:safetensors:/models/demo",
    }),
  ]);
  assert.equal(options.length, 1);
  const option = options[0];
  assert.ok(option);
  assert.equal(option.id, "/models/demo");
  assert.equal(option.isDirectGguf, undefined);
  assert.equal(localGgufKindFor(option, false), null);
  assert.equal(localGgufKindFor(option, true), "variants");
});

test("hf_cache rows stay out of the local list", () => {
  assert.deepEqual(chatLocalModelOptions([row({ source: "hf_cache" })]), []);
});

test("LM Studio rows use the model name without the publisher folder", () => {
  const modelPath = "N:\\AI Models\\Qwen\\Qwen3.6-40B-Deck-Opus";
  const options = chatLocalModelOptions([
    row({
      id: modelPath,
      source: "lmstudio",
      model_id: "Qwen/Qwen3.6-40B-Deck-Opus",
      display_name: "Qwen3.6-40B-Deck-Opus",
    }),
  ]);
  assert.equal(options[0]?.id, modelPath);
  assert.equal(options[0]?.name, "Qwen3.6-40B-Deck-Opus");
  assert.equal(options[0]?.baseModel, "LM Studio");
});

test("custom and models_dir rows keep the labels the picker groups on", () => {
  const options = chatLocalModelOptions([
    row({ id: "/a", source: "custom" }),
    row({ id: "/b", source: "models_dir" }),
  ]);
  assert.deepEqual(
    options.map((o) => o.baseModel),
    ["Custom Folders", "Local models"],
  );
});

test("an oMLX row is offered and labelled oMLX", () => {
  const options = chatLocalModelOptions([
    row({
      id: "/Users/u/.omlx/models/mlx-community/Qwen3.8-27B-4bit",
      path: "/Users/u/.omlx/models/mlx-community/Qwen3.8-27B-4bit",
      display_name: "Qwen3.8-27B-4bit",
      source: "omlx",
      model_format: "safetensors",
    }),
  ]);
  assert.equal(options.length, 1);
  assert.equal(options[0].baseModel, "oMLX");
});
