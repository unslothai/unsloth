// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type GpuIndexKind = "physical" | "vulkan";

export interface SystemGpuDevice {
  index: number;
  indexKind: GpuIndexKind | null;
  name: string;
  memoryTotalGb: number;
  memoryFreeGb: number;
  /** Whether free memory was reported, including a real zero. */
  memoryFreeKnown?: boolean;
  /** A Vulkan iGPU: memoryTotalGb is a capped view of system RAM. Per device, since a mixed
   * inventory pairs it with a discrete card. */
  sharedMemory: boolean;
  sharedMemoryHostBackedGb?: number | null;
  /** Device and host are ONE pool (Apple Silicon, ROCm APU). Per device, like sharedMemory. */
  unifiedMemory?: boolean;
  pinnable: boolean;
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

/** A known namespace still rules out physical IDs even before Vulkan membership is known. */
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

/** Visible ordinal 0, not the minimum physical `index`: a reordering CUDA_VISIBLE_DEVICES like
 * "3,1" maps ordinal 0 to GPU 3. Older backends without visible_ordinal fall back to `index`. */
export function pickLoadDevice<
  T extends { index?: number; visible_ordinal?: number },
>(devices: T[]): T | undefined {
  const ordered = devices.filter((d) => typeof d.visible_ordinal === "number");
  if (ordered.length > 0) {
    return ordered.reduce((pick, d) =>
      (d.visible_ordinal as number) < (pick.visible_ordinal as number)
        ? d
        : pick,
    );
  }
  if (devices.length === 0) return undefined;
  return devices.reduce(
    (pick, d) => ((d.index ?? 0) < (pick.index ?? 0) ? d : pick),
    devices[0],
  );
}
