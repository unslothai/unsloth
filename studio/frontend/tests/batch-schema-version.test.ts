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

function config(
  nBatch: number | null,
  nUbatch: number | null = null,
  mlx: Record<string, unknown> = {},
) {
  return {
    customContextLength: null,
    maxSeqLength: null,
    kvCacheDtype: null,
    ...mlx,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: 4,
    nBatch,
    nUbatch,
    reasoningBudget: -1,
    reasoningBudgetMessage: "",
    tensorParallel: false,
    disableVision: false,
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

for (const mode of ["mtp", "off"] as const) {
  test(`a record asking for ${mode} is stamped v7 so a v6 client cannot rewrite it away`, () => {
    store.clear();
    const mlx = { mlxSpeculativeMode: mode, mlxDraftModel: "org/drafter" };
    assert.ok(savePerModelConfig(MODEL, "Q4_K_M", config(null, null, mlx)));
    assert.equal(storedVersion(), 7);
    const { config: read } = resolveInitialConfig(MODEL, "Q4_K_M");
    assert.equal(read.mlxSpeculativeMode, mode);
    assert.equal(read.mlxDraftModel, mode === "mtp" ? "org/drafter" : null);
  });
}

test("a record left on Auto keeps the version its other fields earn", () => {
  store.clear();
  const mlx = { mlxSpeculativeMode: "auto" };
  assert.ok(savePerModelConfig(MODEL, "Q4_K_M", config(4096, 1024, mlx)));
  assert.equal(storedVersion(), 2);
});
