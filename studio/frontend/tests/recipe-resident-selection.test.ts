// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A stored spelling, the tag a load records, two tags on one blob: only the server relates them.

import assert from "node:assert/strict";
import test from "node:test";
import { isOllamaModelId } from "../src/features/hub/lib/model-identity.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Selection = { target: string; ggufVariant: string; aliases: string[] };

const chat = {
  status: {} as Record<string, unknown>,
  validated: [] as string[],
  resident: false,
  getInferenceStatus: () => Promise.resolve(chat.status),
  // biome-ignore lint/style/useNamingConvention: api schema
  validateModel: (request: { model_path: string }) => {
    chat.validated.push(request.model_path);
    return Promise.resolve({ valid: true, resident: chat.resident });
  },
};

// The gate is what this drives, so the hook's other dependencies are empty shells: reaching
// one throws instead of passing quietly.
const { isLocalModelAlreadyLoaded } = loadWithStubs<{
  isLocalModelAlreadyLoaded: (selection: Selection) => Promise<boolean>;
}>(
  new URL(
    "../src/features/recipe-studio/hooks/use-recipe-executions.ts",
    import.meta.url,
  ),
  {
    "@/features/chat": chat,
    "@/features/hub/lib/model-identity": { isOllamaModelId },
    "@/features/chat/presets/preset-policy": {},
    "@/config/env": {},
    "@/features/model-picker": {},
    "@/lib/toast": {},
    "@/shared/toast": {},
    react: {},
    "zustand/react/shallow": {},
    "../api": {},
    "../data/executions-db": {},
    "../executions/execution-helpers": {},
    "../executions/hydration": {},
    "../executions/run-settings": {},
    "../executions/runtime": {},
    "../stores/recipe-executions": {},
    "../executions/tracker": {},
  },
);

const LINK = "/home/u/.ollama/.studio_links/ab12cd34ef/llama3-latest.gguf";
const REF = "ollama-manifest:%2Fh%2F.ollama%2Fmanifests%2Fllama3%2Flatest";

async function loaded(target: string, resident: boolean): Promise<boolean> {
  // biome-ignore lint/style/useNamingConvention: api schema
  chat.status = { model_identifier: REF, gguf_variant: "" };
  chat.validated = [];
  chat.resident = resident;
  return await isLocalModelAlreadyLoaded({
    target,
    ggufVariant: "",
    aliases: [],
  });
}

test("a stored spelling is loaded when the server says its tag is resident", async () => {
  for (const stored of [
    LINK,
    LINK.replace("llama3-latest.gguf", "llama3-latest-Q4_K_M.gguf"),
    LINK.replace("llama3-latest", "llama3-8b"),
  ]) {
    assert.equal(await loaded(stored, true), true);
    assert.deepEqual(chat.validated, [stored]);
    assert.equal(await loaded(stored, false), false);
  }
});

test("only a name that disagrees and is an Ollama one costs a request", async () => {
  assert.equal(await loaded(REF, false), true);
  assert.deepEqual(chat.validated, []);
  assert.equal(await loaded("/models/mistral.gguf", true), false);
  assert.deepEqual(chat.validated, []);
});
