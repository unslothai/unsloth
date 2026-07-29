// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);

const store = new Map<string, string>();
const storage = {
  getItem: (key: string) => store.get(key) ?? null,
  setItem: (key: string, value: string) => {
    store.set(key, value);
  },
  removeItem: (key: string) => {
    store.delete(key);
  },
};
Object.assign(globalThis, {
  window: { localStorage: storage },
  localStorage: storage,
});

const REPO_KEY = 'v2:["unsloth/repo-gguf","q4_k_m"]';

// The legacy import of unsloth_load_settings runs once, on the first read after
// load, so it has to be staged before the module is imported.
store.set(
  "unsloth_model_configs",
  JSON.stringify({ [REPO_KEY]: { version: 1, maxSeqLength: 32768 } }),
);
store.set(
  "unsloth_load_settings",
  JSON.stringify({ "Unsloth/Repo-GGUF::Q4_K_M": { contextLength: 8192 } }),
);

const { listPerModelConfigs, resolveInitialConfig, savePerModelConfig } =
  await import("../src/features/model-picker/model-config/per-model-config.ts");

function config(maxSeqLength: number, kvCacheDtype: string | null = null) {
  return {
    customContextLength: null,
    maxSeqLength,
    kvCacheDtype,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: null,
    tensorParallel: false,
    chatTemplateOverride: null,
  };
}

function storedKeys(): string[] {
  return Object.keys(
    JSON.parse(storage.getItem("unsloth_model_configs") ?? "{}"),
  );
}

// The one-time backfill re-reads listPerModelConfigs() to pick up a save that
// landed while the override fetch was in flight, and matches on the folded
// identity. That is only unambiguous because storage holds one record per model,
// so these pin that rule rather than the backfill.
test("importing the legacy load settings never doubles up a model", () => {
  // The typed casing in unsloth_load_settings names the model the v2 record
  // already holds, so the import has to leave it alone rather than add a second
  // record the picker would prefer and the backfill would not.
  assert.deepEqual(listPerModelConfigs().length, 1);
  assert.deepEqual(storedKeys(), [REPO_KEY]);
  assert.equal(
    resolveInitialConfig("unsloth/repo-gguf", "q4_k_m").config
      .customContextLength,
    null,
  );
});

test("two spellings of one model id keep a single stored record", () => {
  store.clear();
  savePerModelConfig("Unsloth/Repo-GGUF", "Q4_K_M", config(4096));
  savePerModelConfig("unsloth/repo-gguf", "q4_k_m", config(32768, "q8_0"));

  assert.deepEqual(storedKeys(), [REPO_KEY]);
  const listed = listPerModelConfigs();
  assert.equal(listed.length, 1);
  assert.equal(listed[0]?.config.maxSeqLength, 32768);
  // What the picker applies and the only thing the backfill can see agree.
  assert.equal(
    resolveInitialConfig("Unsloth/Repo-GGUF", "Q4_K_M").config.maxSeqLength,
    32768,
  );
});

test("two spellings of one Windows path keep a single stored record", () => {
  store.clear();
  savePerModelConfig("C:\\Models\\Foo.gguf", null, config(4096));
  savePerModelConfig("c:/models/foo.gguf", null, config(32768, "q8_0"));

  assert.deepEqual(storedKeys(), ['v2:["c:/models/foo.gguf",""]']);
  assert.equal(listPerModelConfigs().length, 1);
});

test("a POSIX path is case sensitive, so its two spellings stay separate", () => {
  store.clear();
  savePerModelConfig("/models/Foo.gguf", null, config(4096));
  savePerModelConfig("/models/foo.gguf", null, config(32768, "q8_0"));

  assert.equal(storedKeys().length, 2);
  assert.equal(
    resolveInitialConfig("/models/Foo.gguf", null).config.maxSeqLength,
    4096,
  );
});
