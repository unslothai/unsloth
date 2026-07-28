// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type GpuIndexKind = "physical" | "vulkan";

export interface SystemGpuDevice {
  index: number;
  indexKind: GpuIndexKind | null;
  name: string;
  memoryTotalGb: number;
  /** Free VRAM at fetch time, or total VRAM when usage is unavailable. */
  memoryFreeGb: number;
  /** Whether `index` is safe to send as gpu_ids. */
  pinnable: boolean;
  /** Whether the separate DiffusionGemma runner can use this physical ID. */
  diffusionPinnable: boolean;
}

export interface ReconciledGpuSelection {
  ids: number[] | null;
  indexKind: GpuIndexKind | null;
}

export function sameGpuSelection(
  left: ReconciledGpuSelection,
  right: ReconciledGpuSelection,
): boolean {
  if (left.indexKind !== right.indexKind) return false;
  if (left.ids === right.ids) return true;
  if (left.ids === null || right.ids === null) return false;
  return (
    left.ids.length === right.ids.length &&
    left.ids.every((id, index) => id === right.ids?.[index])
  );
}

export interface PinnableGpuContext {
  devices: SystemGpuDevice[] | null;
  ids: number[] | null;
  indexKind: GpuIndexKind | null | undefined;
}

export function pinnableGpuContext(
  devices: SystemGpuDevice[] | null,
  forDiffusion = false,
): PinnableGpuContext {
  if (devices === null) {
    return { devices: null, ids: null, indexKind: undefined };
  }
  const pinnable = devices.filter((device) =>
    forDiffusion ? device.diffusionPinnable : device.pinnable,
  );
  const indexKind = pinnable[0]?.indexKind ?? null;
  if (
    pinnable.length <= 1 ||
    indexKind === null ||
    !pinnable.every((device) => device.indexKind === indexKind)
  ) {
    return { devices: pinnable, ids: [], indexKind: null };
  }
  return {
    devices: pinnable,
    ids: pinnable.map((device) => device.index),
    indexKind,
  };
}

/**
 * Selection context when the backend namespace may be known before its device
 * membership. A temporarily unavailable Vulkan inventory still proves that
 * physical IDs are incompatible, but cannot yet range-check Vulkan ordinals.
 */
export function resolveGpuSelectionContext(
  devices: SystemGpuDevice[] | null,
  forDiffusion = false,
  unavailableIndexKind?: GpuIndexKind,
): PinnableGpuContext {
  if (unavailableIndexKind !== undefined) {
    if (forDiffusion) {
      return { devices: [], ids: [], indexKind: null };
    }
    return {
      devices: [],
      ids: null,
      indexKind: unavailableIndexKind,
    };
  }
  return pinnableGpuContext(devices, forDiffusion);
}

export function reconcileGpuSelection(
  ids: number[] | null,
  savedIndexKind: GpuIndexKind | null | undefined,
  currentIndexKind: GpuIndexKind | null | undefined,
  pinnableIds: number[] | null,
): ReconciledGpuSelection {
  if (ids == null) {
    return { ids: null, indexKind: null };
  }
  const expectedIndexKind =
    savedIndexKind === undefined ? "physical" : savedIndexKind;
  // Namespace knowledge is authoritative even while membership is unavailable.
  // This is what prevents a saved CUDA/ROCm ID from becoming Vulkan<i>.
  // A null namespace is only safe while discovery is also unresolved.
  if (
    currentIndexKind !== undefined &&
    expectedIndexKind !== currentIndexKind
  ) {
    return { ids: null, indexKind: null };
  }
  if (currentIndexKind === null) {
    return { ids: null, indexKind: null };
  }
  if (currentIndexKind === undefined || pinnableIds === null) {
    return { ids, indexKind: expectedIndexKind };
  }
  const kept = ids.filter((id) => pinnableIds.includes(id));
  return kept.length > 0
    ? { ids: kept, indexKind: currentIndexKind }
    : { ids: null, indexKind: null };
}
