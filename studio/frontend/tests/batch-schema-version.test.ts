// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const { savePerModelConfig, resolveInitialConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const MODEL = "unsloth/Repo-GGUF";

function config(nBatch: number | null, nUbatch: number | null = null) {
  return {
    customContextLength: null,
    maxSeqLength: null,
    kvCacheDtype: null,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: 4,
    nBatch,
    nUbatch,
    tensorParallel: false,
    chatTemplateOverride: null,
  };
}

function storedVersion(): number {
  const map = JSON.parse(store.get("unsloth_model_configs") ?? "{}");
  const [entry] = Object.values(map) as { version: number }[];
  return entry.version;
}

test("a record with batch fields is stamped v2 so a v1 client cannot rewrite them away", () => {
  store.clear();
  assert.ok(savePerModelConfig(MODEL, "Q4_K_M", config(4096, 1024)));
  assert.equal(storedVersion(), 2);
  // and this client still reads its own v2 record back
  const { config: read, remembered } = resolveInitialConfig(MODEL, "Q4_K_M");
  assert.ok(remembered);
  assert.equal(read.nBatch, 4096);
  assert.equal(read.nUbatch, 1024);
});

test("a record without batch fields keeps v1 so older clients can still read it", () => {
  store.clear();
  assert.ok(savePerModelConfig(MODEL, "Q4_K_M", config(null)));
  assert.equal(storedVersion(), 1);
});
