// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { chatModelsFromCatalog } from "../src/features/settings/api/openai-model-catalog.ts";

test("non-chat entries never reach the chat example picker", () => {
  const models = chatModelsFromCatalog({
    data: [
      { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M" },
      { id: "unsloth/Z-Image-Turbo-GGUF", loaded: true, quant: "Q4_K_S", task: "text-to-image" },
      { id: "Lightricks/LTX-2", loaded: false, task: "text-to-video" },
      { id: "tiny", loaded: false, task: "automatic-speech-recognition" },
      { id: "unsloth/orpheus-3b-0.1-ft-GGUF", loaded: false, quant: "Q8_0", task: "text-to-speech" },
      { id: "unsloth/Llama-3.2-1B-Instruct-GGUF", loaded: false, task: "text-generation" },
    ],
  });
  assert.deepEqual(models, [
    { id: "unsloth/Qwen3-8B-GGUF", loaded: true, quant: "Q4_K_M", quants: undefined },
    {
      id: "unsloth/Llama-3.2-1B-Instruct-GGUF",
      loaded: false,
      quant: undefined,
      quants: undefined,
    },
  ]);
});

test("older servers without a task field list every entry as before", () => {
  const models = chatModelsFromCatalog({ data: [{ id: "a", loaded: true }, { id: "b" }] });
  assert.deepEqual(models, [
    { id: "a", loaded: true, quant: undefined, quants: undefined },
    { id: "b", loaded: false, quant: undefined, quants: undefined },
  ]);
});

test("every on-disk quant is kept beside the default pin", () => {
  const models = chatModelsFromCatalog({
    data: [
      { id: "a", loaded: true, quant: "Q8_0", quants: ["Q8_0", "UD-Q4_K_XL", 7] },
      { id: "b", quant: "Q4_K_M", quants: "Q4_K_M" },
    ],
  });
  assert.deepEqual(models, [
    { id: "a", loaded: true, quant: "Q8_0", quants: ["Q8_0", "UD-Q4_K_XL"] },
    { id: "b", loaded: false, quant: "Q4_K_M", quants: undefined },
  ]);
});

test("malformed bodies yield no models", () => {
  assert.deepEqual(chatModelsFromCatalog(null), []);
  assert.deepEqual(chatModelsFromCatalog({}), []);
  assert.deepEqual(chatModelsFromCatalog({ data: [{ id: 3 }, { id: "" }, {}] }), []);
});
