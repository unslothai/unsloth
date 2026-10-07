// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A load served from disk must not prepare a Hub credential: validation can open a blocking dialog.
// Ollama ids are `ollama-manifest:` references, neither a path nor a repo id.

import assert from "node:assert/strict";
import test from "node:test";

import { isLocalModelPath } from "../src/features/chat/utils/model-download-staging.ts";
import { isOllamaModelId } from "../src/features/hub/lib/model-identity.ts";

function mayReachHub(modelId: string, nativePathToken: string | null): boolean {
  const servedFromDisk = isLocalModelPath(modelId) || isOllamaModelId(modelId);
  return !servedFromDisk && nativePathToken == null;
}

test("an opaque Ollama manifest reference prepares no Hub token", () => {
  assert.equal(isLocalModelPath("ollama-manifest:llama3:8b"), false);
  assert.equal(mayReachHub("ollama-manifest:llama3:8b", null), false);
});

test("an Ollama link path prepares no Hub token", () => {
  assert.equal(
    mayReachHub("/home/u/.unsloth/ollama_links/model.gguf", null),
    false,
  );
});

test("a local path or a native lease prepares no Hub token", () => {
  assert.equal(mayReachHub("/models/llama.gguf", null), false);
  assert.equal(mayReachHub("C:\\models\\llama.gguf", null), false);
  assert.equal(mayReachHub("unsloth/Llama-3.2-1B-Instruct", "lease-abc"), false);
});

test("a plain Hub repo id still prepares its token", () => {
  assert.equal(mayReachHub("unsloth/Llama-3.2-1B-Instruct", null), true);
  assert.equal(mayReachHub("unsloth/gemma-3-4b-it-GGUF", null), true);
});
