// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A v2 stamp hides the whole record from v1 clients, so v2 is stamped only on records with a
// batch value, and v1 clients must never overwrite, delete or evict a v2 record.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const { savePerModelConfig, deletePerModelConfig, resolveInitialConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const MODEL = "unsloth/Repo-GGUF";
const VARIANT = "Q4_K_M";
const KEY = "unsloth_model_configs";

function config(overrides: Record<string, unknown> = {}) {
  return {
    customContextLength: null,
    maxSeqLength: null,
    kvCacheDtype: null,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: null,
    nBatch: null,
    nUbatch: null,
    reasoningBudget: -1,
    reasoningBudgetMessage: "",
    tensorParallel: false,
    disableVision: false,
    chatTemplateOverride: null,
    ...overrides,
  };
}

function readMap(): Record<string, Record<string, unknown>> {
  return JSON.parse(store.get(KEY) ?? "{}");
}

function writeMap(map: Record<string, unknown>): void {
  store.set(KEY, JSON.stringify(map));
}

function onlyEntry(): Record<string, unknown> {
  const [entry] = Object.values(readMap());
  return entry;
}

function load() {
  const initial = resolveInitialConfig(MODEL, VARIANT);
  return initial.remembered ? initial.config : null;
}

test("a v0 record with no version key at all still loads every field it carried", () => {
  store.clear();
  writeMap({
    [`${MODEL}::${VARIANT}`]: {
      customContextLength: 8192,
      kvCacheDtype: "q8_0",
      nParallel: 4,
      tensorParallel: true,
    },
  });
  const loaded = load();
  assert.ok(loaded, "a v0 record must remain readable");
  assert.equal(loaded.customContextLength, 8192);
  assert.equal(loaded.kvCacheDtype, "q8_0");
  assert.equal(loaded.nParallel, 4);
  assert.equal(loaded.tensorParallel, true);
  assert.equal(loaded.nBatch, null);
  assert.equal(loaded.nUbatch, null);
});

test("a v1 record loads unchanged and is re-stamped v1, not silently upgraded", () => {
  store.clear();
  writeMap({
    [`${MODEL}::${VARIANT}`]: {
      version: 1,
      customContextLength: 4096,
      nParallel: 2,
    },
  });
  const loaded = load();
  assert.ok(loaded);
  assert.equal(loaded.customContextLength, 4096);
  assert.equal(loaded.nParallel, 2);

  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ customContextLength: 4096, nParallel: 2 })));
  assert.equal(onlyEntry().version, 1, "a batchless record must stay v1");
});

test("only records that actually carry a batch value are stamped v2", () => {
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config()));
  assert.deepEqual(readMap(), {}, "a default config must not be written");

  for (const [patch, expected] of [
    [{ nParallel: 8 }, 1],
    [{ kvCacheDtype: "q8_0", tensorParallel: true }, 1],
    [{ nBatch: 4096 }, 2],
    [{ nUbatch: 512 }, 2],
    [{ nBatch: 4096, nUbatch: 512 }, 2],
  ] as const) {
    store.clear();
    assert.ok(savePerModelConfig(MODEL, VARIANT, config(patch)));
    assert.equal(
      onlyEntry().version,
      expected,
      `version for ${JSON.stringify(patch)}`,
    );
  }
});

test("a v2 record survives byte-for-byte when an old client refuses it", () => {
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ nBatch: 4096, nUbatch: 1024 })));

  const map = readMap();
  const [key] = Object.keys(map);
  map[key].version = 99;
  writeMap(map);
  const poisoned = store.get(KEY);

  assert.equal(load(), null, "load hides a future record");
  assert.equal(
    savePerModelConfig(MODEL, VARIANT, config({ nParallel: 1 })),
    false,
    "save must refuse rather than overwrite a future record",
  );
  assert.equal(
    deletePerModelConfig(MODEL, VARIANT),
    false,
    "delete must refuse a future record",
  );
  assert.equal(store.get(KEY), poisoned, "the stored bytes must be untouched");

  const restored = readMap();
  restored[Object.keys(restored)[0]].version = 2;
  writeMap(restored);
  const loaded = load();
  assert.ok(loaded, "downgrade then upgrade must round-trip");
  assert.equal(loaded.nBatch, 4096);
  assert.equal(loaded.nUbatch, 1024);
});

test("a future record shows defaults rather than another model's settings", () => {
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ nBatch: 4096 })));
  const map = readMap();
  map[Object.keys(map)[0]].version = 99;
  writeMap(map);

  const initial = resolveInitialConfig(MODEL, VARIANT);
  assert.equal(initial.config.nBatch, null);
  assert.equal(initial.remembered, false);
});

test("a status payload from a backend that omits the batch echo is a no-op", async () => {
  const { resolveBatchSizeSeed } = await import(
    "../src/features/chat/lib/resolve-batch-size-seed.ts"
  );
  // An older backend omits requested_n_batch; that is no information, not the default.
  const pinned = { value: 4096, loaded: 4096 };
  assert.deepEqual(
    resolveBatchSizeSeed({
      incoming: undefined,
      isGguf: true,
      previous: pinned,
      seedLoadParams: true,
    }),
    {},
    "an absent echo must be a no-op, never a clear",
  );
  assert.deepEqual(
    resolveBatchSizeSeed({
      incoming: undefined,
      isGguf: true,
      previous: { value: 2048, loaded: 4096 },
      seedLoadParams: true,
    }),
    {},
  );
  assert.deepEqual(
    resolveBatchSizeSeed({
      incoming: null,
      isGguf: true,
      previous: pinned,
      seedLoadParams: true,
    }),
    { loaded: null, value: null },
    "an explicit null is information and must be honoured",
  );
});
