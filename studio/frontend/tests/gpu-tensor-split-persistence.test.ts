// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();
const {
  DEFAULT_PER_MODEL_CONFIG,
  normalizePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { fromApiOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);

const MODEL = "unsloth/split-GGUF";
const VARIANT = "Q4_K_M";
const configured = () => ({
  ...DEFAULT_PER_MODEL_CONFIG,
  gpuMemoryMode: "manual" as const,
  gpuLayers: 66,
  selectedGpuIds: [1, 2, 0],
  selectedGpuIndexKind: "physical" as const,
  tensorSplit: [30, 20, 16],
});

test("saving and reopening keeps the split in GPU order", () => {
  store.clear();
  assert.equal(savePerModelConfig(MODEL, VARIANT, configured()), true);
  const restored = resolveInitialConfig(MODEL, VARIANT);
  assert.equal(restored.remembered, true);
  assert.deepEqual(restored.config.selectedGpuIds, [1, 2, 0]);
  assert.deepEqual(restored.config.tensorSplit, [30, 20, 16]);
  assert.equal(restored.config.gpuLayers, 66);
});

test("a stored split is protected from clients that do not understand it", () => {
  store.clear();
  savePerModelConfig(MODEL, VARIANT, configured());
  const [entry] = Object.values(
    JSON.parse(store.get("unsloth_model_configs")!),
  ) as { version: number }[];
  assert.ok(entry.version > 9);
  store.clear();
  savePerModelConfig(MODEL, VARIANT, { ...configured(), tensorSplit: null });
  const [withoutSplit] = Object.values(
    JSON.parse(store.get("unsloth_model_configs")!),
  ) as { version: number }[];
  assert.equal(withoutSplit.version, 1);
});

test("the API mirror preserves the ratio and GPU order", () => {
  const payload = toApiOverride(configured());
  assert.deepEqual(payload.gpu_ids, [1, 2, 0]);
  assert.deepEqual(
    (payload as Record<string, unknown>).tensor_split,
    [30, 20, 16],
  );
  assert.deepEqual(fromApiOverride(payload).tensorSplit, [30, 20, 16]);
});

test("hydration never attaches a local ratio to another server GPU order", () => {
  const local = configured();
  const restored = fromApiOverride({ gpu_ids: [0, 1, 2] }, local);
  assert.deepEqual(restored.selectedGpuIds, [0, 1, 2]);
  assert.equal(restored.tensorSplit ?? null, null);
  assert.deepEqual(fromApiOverride({}, local).tensorSplit, local.tensorSplit);
  assert.equal(
    fromApiOverride({ gpu_ids: [1, 2, 0], tensor_split: null }, local)
      .tensorSplit,
    null,
  );
  assert.equal(
    fromApiOverride({ gpu_ids: [1, 2, 0], gpu_index_kind: "vulkan" }, local)
      .tensorSplit,
    null,
  );
});

test("a server ratio cannot attach to unrelated local GPU IDs", () => {
  assert.equal(
    fromApiOverride({ tensor_split: [3, 2, 1] }, configured()).tensorSplit,
    null,
  );
  assert.equal(
    fromApiOverride({ tensor_split: null }, configured()).tensorSplit,
    null,
  );
});

test("default and malformed ratios are not persisted", () => {
  for (const tensorSplit of [
    null,
    [],
    [0, 0, 0],
    [-1, 20, 47],
    [NaN, 20, 16],
    [Infinity, 20, 16],
    [30, 36],
    [Number.MAX_VALUE, Number.MAX_VALUE, 1],
  ]) {
    assert.equal(
      normalizePerModelConfig({ ...configured(), tensorSplit }).tensorSplit ??
        null,
      null,
    );
  }
  for (const selectedGpuIds of [
    null,
    undefined,
    [1, 1, 0],
    [1.5, 2, 0],
    [-1, 2, 0],
  ]) {
    assert.equal(
      normalizePerModelConfig({ ...configured(), selectedGpuIds })
        .tensorSplit ?? null,
      null,
    );
  }
});

test("zero shares and fractional ratios survive storage", () => {
  store.clear();
  const value = { ...configured(), tensorSplit: [0, 2.5, 1] };
  savePerModelConfig(MODEL, VARIANT, value);
  assert.deepEqual(
    resolveInitialConfig(MODEL, VARIANT).config.tensorSplit,
    value.tensorSplit,
  );
});

test("clearing the ratio removes it from the next saved config", () => {
  store.clear();
  savePerModelConfig(MODEL, VARIANT, configured());
  savePerModelConfig(MODEL, VARIANT, { ...configured(), tensorSplit: null });
  assert.equal(
    resolveInitialConfig(MODEL, VARIANT).config.tensorSplit ?? null,
    null,
  );
  assert.equal(
    toApiOverride({ ...configured(), tensorSplit: null }).tensor_split,
    null,
  );
});

test("a removed GPU, changed order, or backend namespace discards the saved split", async () => {
  const { reconcileTensorSplit } = await import(
    "../src/hooks/gpu-tensor-split.ts"
  );
  const { reconcileGpuSelection } = await import(
    "../src/hooks/gpu-selection.ts"
  );
  const value = configured();
  const savedIds = value.selectedGpuIds;
  assert.deepEqual(
    reconcileTensorSplit(value.tensorSplit, savedIds, savedIds),
    value.tensorSplit,
  );
  for (const current of [null, [1, 2], [0, 1, 2]]) {
    assert.equal(
      reconcileTensorSplit(value.tensorSplit, savedIds, current),
      null,
    );
  }
  const otherBackend = reconcileGpuSelection(
    savedIds,
    "physical",
    "vulkan",
    savedIds,
  );
  assert.equal(
    reconcileTensorSplit(value.tensorSplit, savedIds, otherBackend.ids),
    null,
  );
});

test("an unpinned split survives a reload that stays unpinned", async () => {
  const { reconcileTensorSplit } = await import(
    "../src/hooks/gpu-tensor-split.ts"
  );
  assert.deepEqual(reconcileTensorSplit([3, 1], null, null), [3, 1]);
  assert.deepEqual(reconcileTensorSplit([3, 1], undefined, null), [3, 1]);
  assert.equal(reconcileTensorSplit([3, 1], null, [0, 1]), null);
  assert.equal(reconcileTensorSplit([0, 0], null, null), null);
  assert.equal(reconcileTensorSplit([3], null, null), null);
});
