import assert from "node:assert/strict";
import test from "node:test";
import type { MemoryEstimate } from "../src/features/model-picker/api/memory-estimate.ts";
import { estimatePerGpuMemory } from "../src/features/model-picker/model-config/per-gpu-memory-estimate.ts";
import type { SystemGpuDevice } from "../src/hooks/gpu-selection.ts";

const G = 1024 ** 3;

function device(index: number, memoryFreeGb: number): SystemGpuDevice {
  return {
    index,
    indexKind: "physical",
    name: `GPU ${index}`,
    memoryTotalGb: 16,
    memoryFreeGb,
    memoryFreeKnown: true,
    sharedMemory: false,
    pinnable: true,
    diffusionPinnable: true,
  };
}

const estimate = {
  gpuBytes: 10 * G,
  kvBytes: 4 * G,
  kvCheckpointBytes: 0,
  kvOnGpu: true,
  computeBytes: 2 * G,
  gpuLayers: 12,
} as MemoryEstimate;

test("defaults the per-GPU preview to current free VRAM in selected order", () => {
  const rows = estimatePerGpuMemory(
    estimate,
    [device(0, 4), device(1, 12)],
    [1, 0],
    null,
    false,
  );
  assert.deepEqual(rows.map((row) => row.device.index), [1, 0]);
  assert.deepEqual(rows.map((row) => row.layers), [9, 3]);
  assert.deepEqual(rows.map((row) => row.totalBytes), [7.5 * G, 2.5 * G]);
  assert.deepEqual(rows.map((row) => row.kvBytes), [3 * G, 1 * G]);
  assert.equal(rows.reduce((sum, row) => sum + row.totalBytes, 0), estimate.gpuBytes);
});

test("honors an explicit split and keeps integer layer counts summing to the requested total", () => {
  const rows = estimatePerGpuMemory(
    estimate,
    [device(0, 8), device(1, 8)],
    [0, 1],
    [2, 10],
    false,
  );
  assert.deepEqual(rows.map((row) => row.layers), [2, 10]);
  assert.deepEqual(rows.map((row) => row.totalBytes), [1_789_569_707, 8_947_848_533]);
});

test("does not invent per-device layer counts for tensor-parallel placement", () => {
  const rows = estimatePerGpuMemory(
    estimate,
    [device(0, 8), device(1, 8)],
    [0, 1],
    [75, 25],
    true,
  );
  assert.deepEqual(rows.map((row) => row.layers), [null, null]);
  assert.deepEqual(rows.map((row) => row.totalBytes), [7.5 * G, 2.5 * G]);
});

test("omits the preview unless at least two pinnable GPUs are available", () => {
  assert.deepEqual(
    estimatePerGpuMemory(estimate, [device(0, 8)], null, null, false),
    [],
  );
  assert.deepEqual(
    estimatePerGpuMemory(
      estimate,
      [{ ...device(0, 8), pinnable: false }, { ...device(1, 8), pinnable: false }],
      null,
      null,
      false,
    ),
    [],
  );
});
