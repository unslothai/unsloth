// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();
const { normalizePerModelConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);

test("a MIG instance pick keeps its CUDA-device namespace through normalization", () => {
  const config = normalizePerModelConfig({
    selectedGpuIds: [2, 0],
    selectedGpuIndexKind: "cuda_device",
  });
  assert.deepEqual(config.selectedGpuIds, [2, 0]);
  assert.equal(config.selectedGpuIndexKind, "cuda_device");
  // And the saved override carries it, so the server never reads these ids as physical cards.
  assert.equal(toApiOverride(config).gpu_index_kind, "cuda_device");
});

test("an unknown namespace is still dropped", () => {
  const config = normalizePerModelConfig({
    selectedGpuIds: [0],
    selectedGpuIndexKind: "relative",
  });
  assert.equal(config.selectedGpuIndexKind, undefined);
});
