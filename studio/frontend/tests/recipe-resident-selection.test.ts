// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A stored spelling, the tag a load records, two tags on one blob: only the server relates them.

import assert from "node:assert/strict";
import test from "node:test";
import * as modelIdentity from "../src/features/hub/lib/model-identity.ts";
import type * as ResidencyModule from "../src/features/recipe-studio/lib/local-model-residency.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// chat-api reaches .tsx dialogs, which bare node cannot parse.
const chatApi = {
  status: {} as Record<string, unknown>,
  validated: [] as string[],
  resident: false,
  getInferenceStatus: () => Promise.resolve(chatApi.status),
  // biome-ignore lint/style/useNamingConvention: api schema
  validateModel: (request: { model_path: string }) => {
    chatApi.validated.push(request.model_path);
    return Promise.resolve({ valid: true, resident: chatApi.resident });
  },
};

const { isLocalModelAlreadyLoaded } = loadWithStubs<typeof ResidencyModule>(
  new URL(
    "../src/features/recipe-studio/lib/local-model-residency.ts",
    import.meta.url,
  ),
  {
    "@/features/chat/api/chat-api": chatApi,
    "@/features/hub/lib/model-identity": modelIdentity,
  },
);

const LINK = "/home/u/.ollama/.studio_links/ab12cd34ef/llama3-latest.gguf";
const REF = "ollama-manifest:%2Fh%2F.ollama%2Fmanifests%2Fllama3%2Flatest";

async function loaded(target: string, resident: boolean): Promise<boolean> {
  // biome-ignore lint/style/useNamingConvention: api schema
  chatApi.status = { model_identifier: REF, gguf_variant: "" };
  chatApi.validated = [];
  chatApi.resident = resident;
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
    assert.deepEqual(chatApi.validated, [stored]);
    assert.equal(await loaded(stored, false), false);
  }
});

test("only a name that disagrees and is an Ollama one costs a request", async () => {
  assert.equal(await loaded(REF, false), true);
  assert.deepEqual(chatApi.validated, []);
  assert.equal(await loaded("/models/mistral.gguf", true), false);
  assert.deepEqual(chatApi.validated, []);
});
