// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ownsRestore,
  shouldPutBack,
} from "../src/features/benchmarks/llama-bench/llama-bench-restore.ts";

const chat = { model: "unsloth/Qwen3-4B-GGUF", variant: "Q4_K_M" };
const picked = { model: "unsloth/gemma-3-1b-it-GGUF", variant: "Q8_0" };
const none = { model: null, variant: null };

test("a persisted restore only resumes for the run it was saved with", () => {
  assert.equal(ownsRestore("llama-bench-a", "llama-bench-a"), true);
  // A later run, or none after a restart: the old model must not load over the user's.
  assert.equal(ownsRestore("llama-bench-a", "llama-bench-b"), false);
  assert.equal(ownsRestore("llama-bench-a", undefined), false);
  // Saved before the server answered: no run to tie it to.
  assert.equal(ownsRestore(null, "llama-bench-a"), false);
});

test("chat's model goes back only over an empty slot or the benched model", () => {
  assert.equal(shouldPutBack(none, chat, picked), true);
  assert.equal(shouldPutBack(picked, chat, picked), true);
  // Another tab already put it back, or the run was refused before the unload.
  assert.equal(shouldPutBack(chat, chat, chat), false);
  assert.equal(shouldPutBack(chat, chat, picked), false);
  // The user loaded something else since.
  assert.equal(
    shouldPutBack({ model: "other", variant: null }, chat, picked),
    false,
  );
});
