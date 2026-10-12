// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SystemGpuDevice } from "@/hooks/gpu-selection";
import type { MemoryEstimate } from "../api/memory-estimate";

export interface PerGpuMemoryEstimate {
  device: SystemGpuDevice;
  /** null when the backend is choosing the layer count (Auto / --fit). */
  layers: number | null;
  layerTotal: number | null;
  modelBytes: number;
  kvBytes: number;
  computeBytes: number;
  totalBytes: number;
}

function normalizedWeights(weights: readonly number[]): number[] | null {
  if (
    weights.length < 2 ||
    weights.some((value) => !Number.isFinite(value) || value < 0)
  ) {
    return null;
  }
  const total = weights.reduce((sum, value) => sum + value, 0);
  return Number.isFinite(total) && total > 0
    ? weights.map((value) => value / total)
    : null;
}

function allocateBytes(total: number, ratios: readonly number[]): number[] {
  const exact = ratios.map((ratio) => Math.max(0, total) * ratio);
  const allocated = exact.map(Math.floor);
  let remainder = Math.max(0, Math.floor(total) - allocated.reduce((a, b) => a + b, 0));
  const order = exact
    .map((value, index) => ({ index, fraction: value - allocated[index] }))
    .sort((a, b) => b.fraction - a.fraction);
  for (let i = 0; i < order.length && remainder > 0; i += 1, remainder -= 1) {
    allocated[order[i].index] += 1;
  }
  return allocated;
}

function allocateLayers(total: number, ratios: readonly number[]): number[] {
  return allocateBytes(total, ratios);
}

/**
 * A pre-load per-device projection. The aggregate estimator does not expose each tensor's
 * placement, so this apportions its totals by the selected split (or current free VRAM) and
 * explicitly remains an estimate. It never changes the loader's placement settings.
 */
export function estimatePerGpuMemory(
  estimate: MemoryEstimate,
  devices: readonly SystemGpuDevice[],
  selectedGpuIds: readonly number[] | null,
  tensorSplit: readonly number[] | null,
  tensorParallel: boolean,
): PerGpuMemoryEstimate[] {
  const orderedDevices = (
    selectedGpuIds == null
      ? [...devices].filter((device) => device.pinnable)
      : selectedGpuIds
          .map((id) => devices.find((device) => device.index === id))
          .filter((device): device is SystemGpuDevice => Boolean(device?.pinnable))
  );
  if (orderedDevices.length < 2) return [];

  const explicit = tensorSplit?.length === orderedDevices.length
    ? normalizedWeights(tensorSplit)
    : null;
  const fallback = normalizedWeights(
    orderedDevices.map((device) =>
      device.memoryFreeKnown && device.memoryFreeGb > 0
        ? device.memoryFreeGb
        : device.memoryTotalGb,
    ),
  );
  const ratios = explicit ?? fallback;
  if (!ratios) return [];

  const gpuBytes = Math.max(0, estimate.gpuBytes);
  // The aggregate API does not report the GPU-resident share of each category. Keep the
  // component figures bounded by its GPU total so stale or older backends cannot render
  // per-card rows whose parts add up to more than the headline.
  const kvBytes = estimate.kvOnGpu
    ? Math.min(
        gpuBytes,
        Math.max(0, estimate.kvBytes - estimate.kvCheckpointBytes),
      )
    : 0;
  const computeBytes = Math.min(
    Math.max(0, gpuBytes - kvBytes),
    Math.max(0, estimate.computeBytes),
  );
  const otherBytes = gpuBytes - kvBytes - computeBytes;
  const weightsByGpu = allocateBytes(otherBytes, ratios);
  const kvByGpu = allocateBytes(kvBytes, ratios);
  const computeByGpu = allocateBytes(computeBytes, ratios);
  const layersByGpu =
    tensorParallel || estimate.gpuLayers == null
      ? null
      : allocateLayers(estimate.gpuLayers, ratios);
  const layerTotal =
    tensorParallel || estimate.gpuLayers == null ? null : estimate.gpuLayers;

  return orderedDevices.map((device, index) => {
    // The server's aggregate breakdown does not distinguish base weights from projector and
    // drafter files. Keep that remainder together instead of claiming it is weights alone.
    const model = weightsByGpu[index];
    const kv = kvByGpu[index];
    const compute = computeByGpu[index];
    return {
      device,
      layers: layersByGpu?.[index] ?? null,
      layerTotal,
      modelBytes: model,
      kvBytes: kv,
      computeBytes: compute,
      totalBytes: model + kv + compute,
    };
  });
}
