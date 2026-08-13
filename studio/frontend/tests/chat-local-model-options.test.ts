// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat reads local models from the shared device inventory rather than the compat
// /api/models/local endpoint. The two disagree in ways that silently drop models, so the
// mapping is pinned here against what each endpoint actually returns.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { chatLocalModelOptions } = await import(
  "../src/features/chat/local-model-options.ts"
);

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

test("Ollama rows are withheld until their refs are loadable", () => {
  // /api/hub/local scans read-only and returns an opaque `ollama-manifest:` id that only
  // materialize_ollama_model_ref can turn into a .gguf path, and nothing calls that yet.
  // The picker withholds these rows for the same reason (PICKER_LOCAL_SOURCES), so Chat
  // matches it: an entry that fails on click is worse than one that is not offered.
  assert.deepEqual(
    chatLocalModelOptions([
      row({
        id: "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fllama3",
        display_name: "llama3",
        source: "ollama",
      }),
    ]),
    [],
  );
});

test("a directory with two weight formats yields one option per load id", () => {
  // The shared inventory keys a row on (format, path), so this arrives as two rows with the
  // same `id`. The selector keys on `id`, so both would share a React key and read as
  // selected together.
  const options = chatLocalModelOptions([
    row({ model_format: "gguf", inventory_id: "models_dir:gguf:/models/demo" }),
    row({
      model_format: "safetensors",
      inventory_id: "models_dir:safetensors:/models/demo",
    }),
  ]);
  assert.equal(options.length, 1);
  assert.equal(options[0]?.id, "/models/demo");
});

test("hf_cache rows stay out of the local list", () => {
  assert.deepEqual(chatLocalModelOptions([row({ source: "hf_cache" })]), []);
});

test("LM Studio rows prefer the model id and keep their label", () => {
  const options = chatLocalModelOptions([
    row({
      id: "/lm/x",
      source: "lmstudio",
      model_id: "publisher/model",
      display_name: "x",
    }),
  ]);
  assert.equal(options[0]?.name, "publisher/model");
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
