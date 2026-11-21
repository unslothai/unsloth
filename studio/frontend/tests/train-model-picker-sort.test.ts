// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const {
  TRAIN_MODEL_SORT_DEFAULT_DESC,
  sortTrainModelDeviceItems,
  sortTrainModelHubIds,
} = await import(
  "../src/features/train-model-picker/components/train-model-picker-view-model.ts"
);

const HUB_IDS = [
  "unsloth/gemma-4-E2B-it",
  "unsloth/gemma-4-E4B-it",
  "unsloth/gemma-4-31B-it",
  "unsloth/gemma-4-26B-A4B-it",
  "unsloth/Qwen3.5-2B",
  "unsloth/Qwen3.5-9B",
  "unsloth/gpt-oss-20b",
  "unsloth/NVIDIA-Nemotron-3-Nano-4B",
  "unsloth/Qwen3-0.6B",
  "unsloth/gemma-3-4b-it",
];

const VRAM_MAP = new Map([
  ["unsloth/gemma-4-E2B-it", { est: 5.1 }],
  ["unsloth/gemma-4-E4B-it", { est: 8.0 }],
  ["unsloth/gemma-4-31B-it", { est: 32.7 }],
  ["unsloth/gemma-4-26B-A4B-it", { est: 25.8 }],
  ["unsloth/Qwen3.5-2B", { est: 2.3 }],
  ["unsloth/Qwen3.5-9B", { est: 9.7 }],
  ["unsloth/gpt-oss-20b", { est: 20.9 }],
  ["unsloth/NVIDIA-Nemotron-3-Nano-4B", { est: 4.0 }],
  ["unsloth/Qwen3-0.6B", { est: 0.6 }],
  ["unsloth/gemma-3-4b-it", { est: 4.3 }],
]);

test("default sort preserves the priority ordering", () => {
  assert.deepEqual(
    sortTrainModelHubIds(HUB_IDS, "default", false, VRAM_MAP),
    HUB_IDS,
  );
});

test("memory sort defaults to ascending so models that fit appear first (#18717)", () => {
  assert.equal(TRAIN_MODEL_SORT_DEFAULT_DESC.memory, false);
  const asc = sortTrainModelHubIds(
    HUB_IDS,
    "memory",
    TRAIN_MODEL_SORT_DEFAULT_DESC.memory,
    VRAM_MAP,
  );
  assert.deepEqual(asc, [
    "unsloth/Qwen3-0.6B",
    "unsloth/Qwen3.5-2B",
    "unsloth/NVIDIA-Nemotron-3-Nano-4B",
    "unsloth/gemma-3-4b-it",
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/Qwen3.5-9B",
    "unsloth/gpt-oss-20b",
    "unsloth/gemma-4-26B-A4B-it",
    "unsloth/gemma-4-31B-it",
  ]);
  const desc = sortTrainModelHubIds(HUB_IDS, "memory", true, VRAM_MAP);
  assert.deepEqual(desc, [...asc].reverse());
});

test("models with unknown memory sort after known-memory models in both directions", () => {
  const ids = ["org/unknown-model", "unsloth/Qwen3.5-9B", "unsloth/Qwen3-0.6B"];
  assert.deepEqual(sortTrainModelHubIds(ids, "memory", false, VRAM_MAP), [
    "unsloth/Qwen3-0.6B",
    "unsloth/Qwen3.5-9B",
    "org/unknown-model",
  ]);
  assert.deepEqual(sortTrainModelHubIds(ids, "memory", true, VRAM_MAP), [
    "unsloth/Qwen3.5-9B",
    "unsloth/Qwen3-0.6B",
    "org/unknown-model",
  ]);
});

test("device items sort by memory (sizeBytes or parameter estimate) and name", () => {
  const base = {
    source: "hf_cache" as const,
    sourceLabel: "HF cache",
    knownCached: true,
    localPath: "/cache",
    modelFormat: "safetensors" as const,
    modelTypeFlags: {
      isVision: false,
      isAudio: false,
      isText: true,
      hasModelTypeSignal: true,
    },
  };
  const items = [
    { ...base, key: "b", id: "unsloth/Qwen3.5-9B", title: "unsloth/Qwen3.5-9B", path: "/b", sizeBytes: 18e9 },
    { ...base, key: "a", id: "unsloth/Qwen3-0.6B", title: "unsloth/Qwen3-0.6B", path: "/a", sizeBytes: 1.2e9 },
    { ...base, key: "c", id: "unsloth/Qwen3.5-2B", title: "unsloth/Qwen3.5-2B", path: "/c", sizeBytes: 4e9 },
  ];
  assert.deepEqual(
    sortTrainModelDeviceItems(items, "memory", false).map((i) => i.id),
    ["unsloth/Qwen3-0.6B", "unsloth/Qwen3.5-2B", "unsloth/Qwen3.5-9B"],
  );
  assert.deepEqual(
    sortTrainModelDeviceItems(items, "memory", true).map((i) => i.id),
    ["unsloth/Qwen3.5-9B", "unsloth/Qwen3.5-2B", "unsloth/Qwen3-0.6B"],
  );
  assert.deepEqual(
    sortTrainModelDeviceItems(items, "name", false).map((i) => i.id),
    ["unsloth/Qwen3-0.6B", "unsloth/Qwen3.5-2B", "unsloth/Qwen3.5-9B"],
  );
  assert.deepEqual(
    sortTrainModelDeviceItems(items, "name", true).map((i) => i.id),
    ["unsloth/Qwen3.5-9B", "unsloth/Qwen3.5-2B", "unsloth/Qwen3-0.6B"],
  );
});
