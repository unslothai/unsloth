// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which backend serves the active model. Four call sites used to answer this by asking
// whether a context length was present, which was true only while llama.cpp was the sole
// backend that reported one.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isServedByLlamaCpp } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("an MLX load is not a GGUF, however much context it reports", () => {
  // The store shape after an MLX model loads: a context window, a native window and a
  // memory-fitted ceiling, and no GGUF source of any kind.
  assert.equal(
    isServedByLlamaCpp({
      loadedIsGguf: false,
      activeGgufVariant: null,
      activeNativePathToken: null,
      checkpoint: "mlx-community/Qwen3.5-4B-MLX-8bit",
    }),
    false,
  );
});

test("the backend's own answer makes a GGUF a GGUF with nothing else to go on", () => {
  // An adopted resident GGUF: Studio started against a running server, so no variant
  // was picked in this session and the checkpoint is a repo id, not a file.
  assert.equal(
    isServedByLlamaCpp({
      loadedIsGguf: true,
      activeGgufVariant: null,
      activeNativePathToken: null,
      checkpoint: "unsloth/Qwen3.5-4B-GGUF",
    }),
    true,
  );
});

test("each pre-load source identifies a GGUF on its own", () => {
  const pick = { loadedIsGguf: null, activeGgufVariant: null, activeNativePathToken: null };
  // An HF variant pick.
  assert.equal(
    isServedByLlamaCpp({ ...pick, activeGgufVariant: "Q4_K_M", checkpoint: "org/Repo-GGUF" }),
    true,
  );
  // A file picked off disk, which carries a lease token and no variant label. The
  // checkpoint deliberately lacks the suffix, so only the token can answer.
  assert.equal(
    isServedByLlamaCpp({ ...pick, activeNativePathToken: "tok", checkpoint: "/m/custom-build" }),
    true,
  );
  // A direct path with neither.
  assert.equal(isServedByLlamaCpp({ ...pick, checkpoint: "/m/Model.GGUF" }), true);
  // None of them.
  assert.equal(isServedByLlamaCpp({ ...pick, checkpoint: "org/Repo" }), false);
});

test("a stale GGUF source outranks a false flag, so a pick is not lost mid-switch", () => {
  // Switching away from an MLX model leaves loadedIsGguf false until the next load
  // responds. The GGUF just picked must still read as one in that window.
  assert.equal(
    isServedByLlamaCpp({
      loadedIsGguf: false,
      activeGgufVariant: "Q4_K_M",
      activeNativePathToken: null,
      checkpoint: "org/Repo-GGUF",
    }),
    true,
  );
});

test("an external provider's model is served by neither backend", () => {
  // Local OpenAI-compatible servers name their models after the file, so the suffix
  // survives into the external id; loadedIsGguf still holds whatever loaded here last.
  assert.equal(
    isServedByLlamaCpp({
      loadedIsGguf: true,
      activeGgufVariant: null,
      activeNativePathToken: null,
      checkpoint: "external::lmstudio::qwen3.5-4b-q4_k_m.gguf",
    }),
    false,
  );
});

test("nothing loaded and nothing picked is not a GGUF", () => {
  assert.equal(isServedByLlamaCpp({}), false);
});
