// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { FORGET_MODEL_OVERRIDE_FAILED, forgetModelOverride } = await import(
  "../src/features/api-monitor/forget-model-override.ts"
);

type Trace = {
  deps: Parameters<typeof forgetModelOverride>[1];
  removedRemote: [string, string | null][];
  removedLocal: [string, string | null][];
  errors: string[];
  reloads: number;
};

function trace(remote: () => Promise<void> = () => Promise.resolve()): Trace {
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
      },
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
