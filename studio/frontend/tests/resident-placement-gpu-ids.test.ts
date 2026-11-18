// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { residentPlacementGpuIds } from "../src/features/model-picker/model-config/memory-fit.ts";

const resident = {
  isActiveModel: true,
  loadedCpuFallback: false,
  loadedGpuLayers: 41,
  loadedGpuIds: [0, 1],
  loadedGpuIndexKind: "physical",
  gpuIndexKind: "physical",
};

test("a resident on the GPUs credits a reload for its pinned cards", () => {
  assert.deepEqual(residentPlacementGpuIds(resident), [0, 1]);
});

test("a zero-layer resident releases no VRAM, so free memory is not excused", () => {
  assert.equal(residentPlacementGpuIds({ ...resident, loadedGpuLayers: 0 }), undefined);
});

test("a CPU fallback or another model is a fresh load", () => {
  assert.equal(residentPlacementGpuIds({ ...resident, loadedCpuFallback: true }), undefined);
  assert.equal(residentPlacementGpuIds({ ...resident, isActiveModel: false }), undefined);
});

test("unknown identities stay unknown", () => {
  assert.equal(residentPlacementGpuIds({ ...resident, gpuIndexKind: "vulkan" }), null);
  assert.equal(residentPlacementGpuIds({ ...resident, loadedGpuIds: [] }), null);
});
