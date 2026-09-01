// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  FORGET_MODEL_OVERRIDE_FAILED,
  FORGET_MODEL_OVERRIDE_LOCAL_FAILED,
  forgetModelOverride,
} = await import("../src/features/api-monitor/forget-model-override.ts");

type Trace = {
  deps: Parameters<typeof forgetModelOverride>[1];
  removedRemote: [string, string | null][];
  removedLocal: [string, string | null][];
  errors: string[];
  reloads: number;
};

function trace(
  remote: () => Promise<void> = () => Promise.resolve(),
  local = true,
  stored: readonly [string, string | null] | null = null,
): Trace {
  const state: Trace = {
    removedRemote: [],
    removedLocal: [],
    errors: [],
    reloads: 0,
    deps: {
      removeRemote: (modelId, ggufVariant) => {
        state.removedRemote.push([modelId, ggufVariant]);
        return remote();
      },
      removeLocal: (modelId, ggufVariant) => {
        state.removedLocal.push([modelId, ggufVariant]);
        return local;
      },
      resolveLocal: () => stored,
      reload: () => {
        state.reloads += 1;
        return Promise.resolve();
      },
      onError: (message) => {
        state.errors.push(message);
      },
    },
  };
  return state;
}

const PATH_KEY = "/home/santiago/Temp-GGUF/qwen38/UD-IQ3_XXS:UD-IQ3_XXS";

test("a forget clears the server entry, then this browser's, then refetches", async () => {
  const state = trace();

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.removedRemote, [
    ["/home/santiago/Temp-GGUF/qwen38/UD-IQ3_XXS", "UD-IQ3_XXS"],
  ]);
  assert.deepEqual(state.removedLocal, state.removedRemote);
  assert.equal(state.reloads, 1);
  assert.deepEqual(state.errors, []);
});

test("a key with no quant suffix forgets the whole id", async () => {
  const state = trace();

  await forgetModelOverride("unsloth/Qwen3-4B-GGUF", state.deps);

  assert.deepEqual(state.removedRemote, [["unsloth/Qwen3-4B-GGUF", null]]);
});

test("a refused remove leaves this browser's copy and the list alone", async () => {
  const state = trace(() =>
    Promise.reject(new Error("Settings are read-only")),
  );

  await assert.doesNotReject(() => forgetModelOverride(PATH_KEY, state.deps));

  assert.deepEqual(state.errors, ["Settings are read-only"]);
  assert.deepEqual(state.removedLocal, []);
  assert.equal(state.reloads, 0);
});

test("a rejection that is not an Error still reports", async () => {
  const state = trace(() => Promise.reject("offline"));

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.errors, [FORGET_MODEL_OVERRIDE_FAILED]);
  assert.deepEqual(state.removedLocal, []);
});

test("a browser copy that could not be deleted is reported, not swallowed", async () => {
  const state = trace(() => Promise.resolve(), false);

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.errors, [FORGET_MODEL_OVERRIDE_LOCAL_FAILED]);
  // The server entry is gone whatever the browser did, so the list still refetches.
  assert.equal(state.reloads, 1);
});

// A quant can name a directory or a whole filename stem, not just a bare token
// (is_qualified_gguf_variant_key in hub/utils/gguf.py). The key still joins on one colon,
// so the server row is removed either way, but reading the whole key as the model id sent
// the browser cleanup at a record that does not exist and reported success.
test("a path-qualified variant splits off the repo it belongs to", async () => {
  const state = trace();

  await forgetModelOverride(
    "unsloth/Repo-GGUF:distilled/model-Q6_K",
    state.deps,
  );

  assert.deepEqual(state.removedRemote, [
    ["unsloth/Repo-GGUF", "distilled/model-Q6_K"],
  ]);
  assert.deepEqual(state.removedLocal, state.removedRemote);
});

test("a filename-stem variant splits off the repo too", async () => {
  const state = trace();

  await forgetModelOverride(
    "unsloth/H3-GGUF:minimax_h3_ref2va_pruned-Q6_K",
    state.deps,
  );

  assert.deepEqual(state.removedRemote, [
    ["unsloth/H3-GGUF", "minimax_h3_ref2va_pruned-Q6_K"],
  ]);
});

test("a colon inside a local path is part of the name, not a separator", async () => {
  const state = trace();

  await forgetModelOverride("/home/u/models/foo:bar/baz.gguf", state.deps);

  assert.deepEqual(state.removedRemote, [
    ["/home/u/models/foo:bar/baz.gguf", null],
  ]);
});

// The browser knows its own two halves; the key's colon does not say which one it is when a
// path meets a directory-qualified variant, and a guess sends the cleanup at nothing.
test("the stored record wins over the key's own split", async () => {
  const state = trace(() => Promise.resolve(), true, [
    "/home/u/models/repo",
    "distilled/model-q6_k",
  ]);

  await forgetModelOverride(
    "/home/u/models/repo:distilled/model-Q6_K",
    state.deps,
  );

  // The remote call keeps the parsed halves: they rejoin to the key the server holds.
  assert.deepEqual(state.removedRemote, [
    ["/home/u/models/repo:distilled/model-Q6_K", null],
  ]);
  assert.deepEqual(state.removedLocal, [
    ["/home/u/models/repo", "distilled/model-q6_k"],
  ]);
});

test("with nothing stored the parsed halves are used", async () => {
  const state = trace();

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.removedLocal, [
    ["/home/santiago/Temp-GGUF/qwen38/UD-IQ3_XXS", "UD-IQ3_XXS"],
  ]);
});
