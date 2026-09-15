// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A stored spelling, the tag a load records, two tags on one blob: only the server relates them.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
register("./helpers/chat-api-stub.mjs", import.meta.url);

const { chatApiStub } = await import("./helpers/chat-api-stub.mjs");
const { isLocalModelAlreadyLoaded } = await import(
  "../src/features/recipe-studio/lib/local-model-residency.ts"
);

const LINK = "/home/u/.ollama/.studio_links/ab12cd34ef/llama3-latest.gguf";
const REF = "ollama-manifest:%2Fh%2F.ollama%2Fmanifests%2Fllama3%2Flatest";

async function loaded(target: string, resident: boolean): Promise<boolean> {
  // biome-ignore lint/style/useNamingConvention: api schema
  chatApiStub.status = { model_identifier: REF, gguf_variant: "" };
  chatApiStub.validated = [];
  chatApiStub.resident = resident;
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
    assert.deepEqual(chatApiStub.validated, [stored]);
    assert.equal(await loaded(stored, false), false);
  }
});

test("only a name that disagrees and is an Ollama one costs a request", async () => {
  assert.equal(await loaded(REF, false), true);
  assert.deepEqual(chatApiStub.validated, []);
  assert.equal(await loaded("/models/mistral.gguf", true), false);
  assert.deepEqual(chatApiStub.validated, []);
});
