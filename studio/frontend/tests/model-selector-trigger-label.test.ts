// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The selector trigger names a lora by the segment before its first slash, so the
// training run shows rather than "<run>/<model>". Local inventory names carry real
// slashes, so the same strip left an Ollama row reading "hf.co". Pinned end to end:
// what Chat feeds the selector, and what the trigger then renders.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { chatLocalModelOptions } = await import(
  "../src/features/chat/local-model-options.ts"
);

const { loraOptionLabel } = await import(
  "../src/features/model-picker/components/model-selector/row-meta.ts"
);

function localRow(over: Record<string, unknown> = {}) {
  return {
    id: "/models/demo",
    // biome-ignore lint/style/useNamingConvention: API schema
    display_name: "demo",
    path: "/models/demo",
    source: "models_dir",
    ...over,
  } as never;
}

test("a training or exported run is still named by its run prefix", () => {
  assert.equal(
    loraOptionLabel({ name: "foo_123/foo", source: "training" }),
    "foo_123",
  );
  assert.equal(
    loraOptionLabel({ name: "foo_123/foo", source: "exported" }),
    "foo_123",
  );
  assert.equal(
    loraOptionLabel({ name: "foo_123", source: "training" }),
    "foo_123",
  );
});

test("an LM Studio pick shows the model folder, not the publisher", () => {
  const options = chatLocalModelOptions([
    localRow({
      id: "N:\\AI Models\\Qwen\\Qwen3.6-40B-Deck-Opus",
      source: "lmstudio",
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: "Qwen/Qwen3.6-40B-Deck-Opus",
      // biome-ignore lint/style/useNamingConvention: API schema
      display_name: "Qwen3.6-40B-Deck-Opus",
    }),
  ]);
  const option = options[0];
  assert.ok(option);
  assert.equal(loraOptionLabel(option), "Qwen3.6-40B-Deck-Opus");
});

test("an Ollama pick keeps the slashes in its repo name", () => {
  // ollama pull hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M; repo_name keeps its host and
  // namespace for anything outside registry.ollama.ai/library.
  const options = chatLocalModelOptions([
    localRow({
      id: "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fhf.co%2Funsloth",
      source: "ollama",
      // biome-ignore lint/style/useNamingConvention: API schema
      model_format: "gguf",
      // biome-ignore lint/style/useNamingConvention: API schema
      display_name: "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M (qwen3 Q4_K_M)",
    }),
  ]);
  const option = options[0];
  assert.ok(option);
  assert.equal(
    loraOptionLabel(option),
    "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M (qwen3 Q4_K_M)",
  );

  const namespaced = chatLocalModelOptions([
    localRow({
      id: "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fsomeuser",
      source: "ollama",
      // biome-ignore lint/style/useNamingConvention: API schema
      model_format: "gguf",
      // biome-ignore lint/style/useNamingConvention: API schema
      display_name: "someuser/mymodel:latest",
    }),
  ])[0];
  assert.ok(namespaced);
  assert.equal(loraOptionLabel(namespaced), "someuser/mymodel:latest");
});
