// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  GPU_LAYERS_AUTO,
  resolveComparePlacement,
} from "../src/features/chat/lib/gpu-placement.ts";

// The live-store snapshot a compare run takes at Send: the settings of whatever
// chat GGUF was loaded then, Manual with 12 of ITS layers pinned to the GPU.
const shared = { gpuMemoryMode: "manual" as const, gpuLayers: 12 };

test("a diffusion pane never inherits another model's layer split", () => {
  // An unremembered pane resolves to DEFAULT_PER_MODEL_CONFIG, which carries
  // neither key -- so `??` would fall through to the shared snapshot.
  assert.deepEqual(resolveComparePlacement({}, shared, true), {
    gpuMemoryMode: "auto",
    gpuLayers: GPU_LAYERS_AUTO,
  });
});

test("a leaked zero cannot mask a diffusion pane's devices", () => {
  assert.deepEqual(
    resolveComparePlacement(
      {},
      { gpuMemoryMode: "manual", gpuLayers: 0 },
      true,
    ),
    { gpuMemoryMode: "auto", gpuLayers: GPU_LAYERS_AUTO },
  );
});

test("a diffusion pane's OWN split is honoured (#7574)", () => {
  assert.deepEqual(
    resolveComparePlacement(
      { gpuMemoryMode: "manual", gpuLayers: 0 },
      shared,
      true,
    ),
    { gpuMemoryMode: "manual", gpuLayers: 0 },
  );
});

test("a chat GGUF pane still inherits the Send-time snapshot", () => {
  assert.deepEqual(resolveComparePlacement({}, shared, false), {
    gpuMemoryMode: "manual",
    gpuLayers: 12,
  });
});

test("an own value wins over the snapshot for a chat GGUF too", () => {
  assert.deepEqual(
    resolveComparePlacement(
      { gpuMemoryMode: "auto", gpuLayers: 3 },
      shared,
      false,
    ),
    { gpuMemoryMode: "auto", gpuLayers: 3 },
  );
});
