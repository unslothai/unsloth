// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The GPUs picker's per-row share (unslothai/unsloth#11474): how much of the model each
// selected card gets, sent as tensor_split. Positional against the selected set, and never
// stored per model.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const physical = (ids: number[], tensorSplit?: number[] | null) =>
  ({
    gpuMemoryMode: "manual",
    gpuLayers: 40,
    selectedGpuIds: ids,
    selectedGpuIndexKind: "physical",
    tensorSplit,
  }) as never;

test("a split edit is a config change, so Load and Apply stay reachable", async () => {
  const { gpuFieldsSignature } = await import(
    "../src/features/model-picker/model-config/config-signature.ts"
  );
  assert.notEqual(
    gpuFieldsSignature(physical([0, 1], [26, 14])),
    gpuFieldsSignature(physical([0, 1], [20, 20])),
  );
  assert.notEqual(
    gpuFieldsSignature(physical([0, 1], [26, 14])),
    gpuFieldsSignature(physical([0, 1], null)),
  );
});

test("no split, whether unset or cleared, reads as the default distribution", async () => {
  const { gpuFieldsSignature } = await import(
    "../src/features/model-picker/model-config/config-signature.ts"
  );
  assert.equal(
    gpuFieldsSignature(physical([0, 1], null)),
    gpuFieldsSignature(physical([0, 1], undefined)),
  );
});

test("the split is never stored with the model's config", () => {
  const src = readSrc("features/model-picker/model-config/per-model-config.ts");
  const stored = src.slice(
    src.indexOf("const STORED_CONFIG_FIELDS"),
    src.indexOf("]);", src.indexOf("const STORED_CONFIG_FIELDS")),
  );
  assert.doesNotMatch(stored, /tensorSplit/);
});

test("the picker drops the split when the set changes and carries it on a reorder", () => {
  const src = readSrc("features/model-picker/components/model-config-page.tsx");
  const commit = src.slice(src.indexOf("const commitGpuIds"));
  assert.match(
    commit.slice(0, 600),
    /nextSplit: number\[\] \| null = null[\s\S]*tensorSplit: nextSplit/,
    "a toggle must reset the positional split",
  );
  const move = src.slice(src.indexOf("const moveGpu"));
  assert.match(
    move.slice(0, 700),
    /\[nextSplit\[from\], nextSplit\[to\]\] = \[nextSplit\[to\], nextSplit\[from\]\]/,
    "a reorder must move each share with its GPU",
  );
});

test("the shares show only where the backend emits --tensor-split", () => {
  const src = readSrc("features/model-picker/components/model-config-page.tsx");
  const gate = src.slice(src.indexOf("const showSplit ="));
  assert.match(
    gate.slice(0, 300),
    /!isDiffusion &&\s*isManual &&\s*!autoLayers &&\s*splitTotal > 0 &&\s*showGpuPicker &&\s*orderedGpuIds\.length > 1/,
  );
});

test("the load sends the editor's split, and the dedupe compares it", () => {
  const src = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(
    src,
    /let loadSplitRatio =\s*pendingLoadConfig\?\.tensorSplit !== undefined\s*\? pendingLoadConfig\.tensorSplit/,
  );
  assert.match(src, /loadSplitRatio = pendingLoadConfig\?\.tensorSplit \?\? null;/);
  assert.match(
    src,
    /splitRatio:\s*pendingConfig\?\.tensorSplit !== undefined\s*\? pendingConfig\.tensorSplit/,
  );
  const apply = readSrc("features/model-picker/model-config/apply-per-model-config.ts");
  assert.match(apply, /splitRatio: options\.isDiffusion \? null : cleanTensorSplit\(config\.tensorSplit\)/);
});
