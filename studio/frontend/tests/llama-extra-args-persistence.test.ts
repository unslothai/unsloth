// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
const {
  DEFAULT_PER_MODEL_CONFIG,
  deletePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { modelStorageKey } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);
const { findModelOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { decideOverrideHydration } = await import(
  "../src/features/model-picker/model-config/override-hydration.ts"
);

const MODEL = "unsloth/Args-GGUF";
const KEY = modelStorageKey(MODEL, "Q4_K_M");

test("absent remains no opinion while explicit empty is saved as schema v3", () => {
  store.clear();
  assert.ok(
    savePerModelConfig(MODEL, "Q4_K_M", { ...DEFAULT_PER_MODEL_CONFIG }),
  );
  assert.equal(store.get("unsloth_model_configs"), undefined);

  assert.ok(
    savePerModelConfig(MODEL, "Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: [],
    }),
  );
  const map = JSON.parse(store.get("unsloth_model_configs") ?? "{}");
  assert.equal(map[KEY].version, 3);
  assert.deepEqual(map[KEY].llamaExtraArgs, []);
  const loaded = resolveInitialConfig(MODEL, "Q4_K_M");
  assert.equal(loaded.remembered, true);
  assert.deepEqual(loaded.config.llamaExtraArgs, []);
});

test("non-empty arguments round-trip and clearing replaces them", () => {
  store.clear();
  assert.ok(
    savePerModelConfig(MODEL, "Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: ["--fit-target", "1024"],
    }),
  );
  assert.deepEqual(
    resolveInitialConfig(MODEL, "Q4_K_M").config.llamaExtraArgs,
    ["--fit-target", "1024"],
  );
  assert.ok(
    savePerModelConfig(MODEL, "Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: [],
    }),
  );
  assert.deepEqual(
    resolveInitialConfig(MODEL, "Q4_K_M").config.llamaExtraArgs,
    [],
  );
});

test("v1 and v2 records stay readable and do not invent authority", () => {
  store.clear();
  store.set(
    "unsloth_model_configs",
    JSON.stringify({
      [KEY]: {
        version: 2,
        ...DEFAULT_PER_MODEL_CONFIG,
        nBatch: 4096,
      },
    }),
  );
  const loaded = resolveInitialConfig(MODEL, "Q4_K_M");
  assert.equal(loaded.config.nBatch, 4096);
  assert.equal(loaded.config.llamaExtraArgs, undefined);
});

test("a future schema cannot be overwritten or deleted by this client", () => {
  store.clear();
  const future = {
    version: 4,
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaExtraArgs: ["--future"],
  };
  store.set("unsloth_model_configs", JSON.stringify({ [KEY]: future }));
  assert.equal(
    savePerModelConfig(MODEL, "Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: [],
    }),
    false,
  );
  assert.equal(deletePerModelConfig(MODEL, "Q4_K_M"), false);
  assert.deepEqual(
    JSON.parse(store.get("unsloth_model_configs") ?? "{}")[KEY],
    future,
  );
});

test("oversize values are refused at the local saved boundary", () => {
  store.clear();
  assert.equal(
    savePerModelConfig(MODEL, "Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: ["x".repeat(32 * 1024 + 1)],
    }),
    false,
  );
});

test("equality and signatures distinguish no opinion from explicit clear", () => {
  const absent = { ...DEFAULT_PER_MODEL_CONFIG };
  const cleared = { ...DEFAULT_PER_MODEL_CONFIG, llamaExtraArgs: [] };
  assert.notEqual(
    loadedConfigSignature(absent),
    loadedConfigSignature(cleared),
  );
  assert.equal(
    loadedConfigSignature(cleared),
    loadedConfigSignature({
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaExtraArgs: [],
    }),
  );
});

test("picker override lookup is ordered exact then unique case-folded", () => {
  const bare = { llama_extra_args: ["--bare"] };
  const qualified = { llama_extra_args: ["--qualified"] };
  assert.deepEqual(
    findModelOverride(
      {
        [MODEL]: bare,
        [`${MODEL}:Q4_K_M`]: qualified,
      },
      MODEL,
      "Q4_K_M",
    ),
    qualified,
  );
  assert.deepEqual(
    findModelOverride({ [MODEL]: bare }, MODEL, "Q4_K_M"),
    bare,
  );
  assert.deepEqual(
    findModelOverride(
      { "UNSLOTH/ARGS-GGUF:q4_k_m": qualified },
      MODEL,
      "Q4_K_M",
    ),
    qualified,
  );
});

test("a qualified explicit clear stops bare fallback", () => {
  assert.deepEqual(
    findModelOverride(
      {
        [MODEL]: { llama_extra_args: ["--bare"] },
        [`${MODEL}:Q4_K_M`]: { llama_extra_args: [] },
      },
      MODEL,
      "Q4_K_M",
    )?.llama_extra_args,
    [],
  );
  assert.deepEqual(
    toApiOverride({ ...DEFAULT_PER_MODEL_CONFIG, llamaExtraArgs: [] }),
    { llama_extra_args: [] },
  );
});

test("an ambiguous qualified fold skips to the unambiguous same-model bare key", () => {
  const bare = { llama_extra_args: ["--bare"] };
  assert.deepEqual(
    findModelOverride(
      {
        "UNSLOTH/ARGS-GGUF:q4_k_m": { llama_extra_args: ["--one"] },
        "unsloth/args-gguf:Q4_K_M": { llama_extra_args: ["--two"] },
        [MODEL]: bare,
        "other/Args-GGUF:Q4_K_M": { llama_extra_args: ["--unrelated"] },
      },
      MODEL,
      "Q4_k_m",
    ),
    bare,
  );
  assert.equal(
    findModelOverride(
      { "other/Args-GGUF:Q4_K_M": { llama_extra_args: ["--wrong"] } },
      MODEL,
      "Q4_K_M",
    ),
    null,
  );
});

test("hydration generations reject stale targets and late Remember ownership", () => {
  assert.deepEqual(
    decideOverrideHydration({
      requestGeneration: 1,
      currentGeneration: 2,
      requestRememberGeneration: 0,
      currentRememberGeneration: 0,
      hasLocalLlamaExtraArgs: false,
      hasServerLlamaExtraArgs: true,
    }),
    { applyArgs: false, applyRemember: false },
  );
  assert.deepEqual(
    decideOverrideHydration({
      requestGeneration: 2,
      currentGeneration: 2,
      requestRememberGeneration: 0,
      currentRememberGeneration: 1,
      hasLocalLlamaExtraArgs: false,
      hasServerLlamaExtraArgs: true,
    }),
    { applyArgs: true, applyRemember: false },
  );
  assert.deepEqual(
    decideOverrideHydration({
      requestGeneration: 2,
      currentGeneration: 2,
      requestRememberGeneration: 1,
      currentRememberGeneration: 1,
      hasLocalLlamaExtraArgs: true,
      hasServerLlamaExtraArgs: true,
    }),
    { applyArgs: false, applyRemember: true },
  );
});
