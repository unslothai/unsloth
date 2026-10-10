// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept apart from use-system.ts so it is testable without the auth/React graph; typed structurally.

export interface VramReportingDevice {
  vram_used_gb?: number;
}

export interface MemoryTotalDevice {
  memory_total_gb?: number;
  shared_memory?: boolean;
  /** Host-backed portion of the shared pool; the rest is reserved GPU memory. */
  shared_memory_host_backed_gb?: number | null;
  /** `hardware.py` sets `shared_memory` only on Windows, so a Linux ROCm APU arrives as
   * `unified_memory: true, shared_memory: false`. */
  unified_memory?: boolean;
}

export interface GpuMemoryTotalsGb {
  dedicated: number;
  shared: number;
  total: number;
}

export interface VramReportingGpu {
  devices?: VramReportingDevice[];
  /** Windows ROCm only, when no single device's usage could be attributed (#7452). */
  vram_used_gb_aggregate?: number | null;
}

function sharesHostMemoryDevice(device: MemoryTotalDevice): boolean {
  return sharesHostMemory({
    sharedMemory: device.shared_memory === true,
    unifiedMemory: device.unified_memory === true,
  });
}

/** Counts a shared host pool once, and rounds back to the devices' 2dp to drop float error. */
export function aggregateGpuMemoryTotalGb(
  devices: MemoryTotalDevice[],
): number {
  return gpuMemoryTotalsGb(devices).total;
}

export function gpuMemoryTotalsGb(
  devices: MemoryTotalDevice[],
): GpuMemoryTotalsGb {
  const size = (device: MemoryTotalDevice) => {
    const total = device.memory_total_gb ?? 0;
    return Number.isFinite(total) && total > 0 ? total : 0;
  };
  const dedicatedDevices = roundToDevicePrecision(
    devices
      .filter((device) => !sharesHostMemoryDevice(device))
      .reduce((sum, device) => sum + size(device), 0),
  );
  const sharedPool = devices
    .filter(sharesHostMemoryDevice)
    .reduce(
      (totals, device) => {
        const total = size(device);
        const hostBackedReported = device.shared_memory_host_backed_gb;
        const hostBackedKnown =
          Number.isFinite(hostBackedReported) &&
          (hostBackedReported as number) >= 0;
        const hostBacked = hostBackedKnown
          ? Math.min(total, hostBackedReported as number)
          : total;
        return {
          // `shared_memory` devices are views of one host pool (take the largest);
          // `unified_memory` alone is a pool per socket (MI300A), which stays additive.
          hostBacked:
            device.shared_memory === true
              ? Math.max(totals.hostBacked, hostBacked)
              : totals.hostBacked,
          perDevice:
            device.shared_memory === true
              ? totals.perDevice
              : totals.perDevice + hostBacked,
          reserved:
            totals.reserved + (hostBackedKnown ? total - hostBacked : 0),
        };
      },
      { hostBacked: 0, perDevice: 0, reserved: 0 },
    );
  const shared = roundToDevicePrecision(
    sharedPool.hostBacked + sharedPool.perDevice,
  );
  const dedicated = roundToDevicePrecision(
    dedicatedDevices + sharedPool.reserved,
  );
  return {
    dedicated,
    shared,
    total: roundToDevicePrecision(dedicated + shared),
  };
}

export function gpuSharedHostMemoryGb(devices: MemoryTotalDevice[]): number {
  return gpuMemoryTotalsGb(devices).shared;
}

export function systemRamAvailableOutsideSharedPoolGb(
  availableGb: number,
  hostBackedSharedPoolGb: number,
): number {
  const available =
    Number.isFinite(availableGb) && availableGb > 0 ? availableGb : 0;
  const shared =
    Number.isFinite(hostBackedSharedPoolGb) && hostBackedSharedPoolGb > 0
      ? hostBackedSharedPoolGb
      : 0;
  return roundToDevicePrecision(
    Math.max(0, available - shared),
  );
}

function roundToDevicePrecision(value: number): number {
  return Math.round(value * 100) / 100;
}

export interface MemoryCapacityDevice {
  memoryTotalGb: number;
  sharedMemory: boolean;
  sharedMemoryHostBackedGb?: number | null;
  /** Linux ROCm APUs report only this flag; see `sharesHostMemory`, which folds both. */
  unifiedMemory?: boolean;
}

export function sharesHostMemory(device: {
  sharedMemory?: boolean;
  unifiedMemory?: boolean;
}): boolean {
  return device.sharedMemory === true || device.unifiedMemory === true;
}

function budgetedGpuMemoryGb(
  devices: MemoryCapacityDevice[],
  budget: number,
): { total: number; independent: number } {
  let dedicated = 0;
  let reserved = 0;
  let partialSharedDemand = 0;
  let fullySharedDemand = 0;
  let sharedPool = 0;
  // Same split as `gpuMemoryTotalsGb`, or capacity and totals disagree on the same inventory.
  let perDevicePool = 0;
  let perDeviceDemand = 0;
  for (const device of devices) {
    const total =
      Number.isFinite(device.memoryTotalGb) && device.memoryTotalGb > 0
        ? device.memoryTotalGb
        : 0;
    const deviceCapacity = total * budget;
    if (!sharesHostMemory(device)) {
      dedicated += deviceCapacity;
      continue;
    }
    const reportedHostBacked = device.sharedMemoryHostBackedGb;
    const hostBacked =
      Number.isFinite(reportedHostBacked) &&
      (reportedHostBacked as number) >= 0
        ? Math.min(total, reportedHostBacked as number)
        : total;
    const deviceReserved = total - hostBacked;
    reserved += Math.min(deviceReserved, deviceCapacity);
    const deviceSharedDemand = Math.max(0, deviceCapacity - deviceReserved);
    if (device.sharedMemory !== true) {
      perDeviceDemand += deviceSharedDemand;
      perDevicePool += hostBacked;
      continue;
    }
    if (deviceReserved > 0) {
      partialSharedDemand += deviceSharedDemand;
    } else {
      // fully shared vulkan rows may be duplicate icd views of one physical device.
      fullySharedDemand = Math.max(fullySharedDemand, deviceSharedDemand);
    }
    sharedPool = Math.max(sharedPool, hostBacked);
  }
  return {
    total: roundToDevicePrecision(
      dedicated +
        reserved +
        Math.min(sharedPool, partialSharedDemand + fullySharedDemand) +
        Math.min(perDevicePool, perDeviceDemand),
    ),
    independent: roundToDevicePrecision(dedicated + reserved),
  };
}

export interface MemoryCapacityInput {
  pinnedDevices: MemoryCapacityDevice[];
  hostDevices?: MemoryCapacityDevice[];
  hostGpuTotalGb: number;
  /** Excludes shared-memory devices, whose budget is already inside system RAM. Absent means
   * the same as `hostGpuTotalGb`. */
  hostDedicatedGpuTotalGb?: number;
  /** Only consulted when nothing is pinned. */
  hostSharesSystemRam: boolean;
  systemRamTotalGb: number;
  /** True on Apple Silicon and ROCm APUs, which report the pool's size differently. */
  unifiedMemory: boolean;
  /**
   * True on Apple, where memory_total_gb IS the unified memory. False on a ROCm APU, whose GPU figure
   * is a BIOS-carved window onto RAM (see llama_cpp.py::_available_system_memory_mib). Defaults true.
   */
  unifiedPoolReportedAsGpuMemory?: boolean;
  /** Fraction of each GPU a load may claim; applied to the GPU figure only, not RAM. */
  gpuBudgetFraction?: number;
}

/**
 * RAM is added only where it is a pool BESIDE the GPU budget, judged per device: a Vulkan iGPU's
 * budget is already a view of RAM. Both figures are 0 when nothing was probed ("no verdict").
 */
export function resolveMemoryCapacityGb(input: MemoryCapacityInput): {
  gpuCapacityGb: number;
  totalCapacityGb: number;
  singleMemoryPool: boolean;
} {
  const pinnedTotals = gpuMemoryTotalsGb(
    input.pinnedDevices.map((device) => ({
      memory_total_gb: device.memoryTotalGb,
      // Both flags raw: pre-folding made this path answer 48 GiB where totals said 96 (#11366).
      shared_memory: device.sharedMemory === true,
      unified_memory: device.unifiedMemory === true,
      shared_memory_host_backed_gb: device.sharedMemoryHostBackedGb,
    })),
  );
  // One flag for both answers so capacity and pool cannot disagree.
  const pinGoverns = pinnedTotals.total > 0;
  // Guard a non-finite number on this public interface, which would NaN every figure.
  const hostGpuTotalGb =
    Number.isFinite(input.hostGpuTotalGb) && input.hostGpuTotalGb > 0
      ? input.hostGpuTotalGb
      : 0;
  const rawGpuCapacityGb = pinGoverns ? pinnedTotals.total : hostGpuTotalGb;
  // Unpinned falls back to the full aggregate, correct on hosts with no shared device.
  const rawDedicatedCapacityGb = pinGoverns
    ? pinnedTotals.dedicated
    : Number.isFinite(input.hostDedicatedGpuTotalGb) &&
        (input.hostDedicatedGpuTotalGb as number) >= 0
      ? (input.hostDedicatedGpuTotalGb as number)
      : hostGpuTotalGb;
  // Measure against the budgeted capacity the slider draws; a 0 or missing value would zero it.
  const budget =
    typeof input.gpuBudgetFraction === "number" &&
    input.gpuBudgetFraction > 0 &&
    input.gpuBudgetFraction <= 1
      ? input.gpuBudgetFraction
      : 1;
  const capacityDevices = pinGoverns
    ? input.pinnedDevices
    : (input.hostDevices ?? []);
  const capacityDeviceTotals = gpuMemoryTotalsGb(
    capacityDevices.map((device) => ({
      memory_total_gb: device.memoryTotalGb,
      // Both flags raw, as above.
      shared_memory: device.sharedMemory === true,
      unified_memory: device.unifiedMemory === true,
      shared_memory_host_backed_gb: device.sharedMemoryHostBackedGb,
    })),
  );
  const canBudgetByDevice =
    capacityDevices.length > 0 &&
    Math.abs(capacityDeviceTotals.total - rawGpuCapacityGb) <= 0.01;
  const budgetedGpuMemory = canBudgetByDevice
    ? budgetedGpuMemoryGb(capacityDevices, budget)
    : null;
  const gpuCapacityGb =
    budgetedGpuMemory?.total ??
    Math.round(rawGpuCapacityGb * budget * 100) / 100;
  const dedicatedCapacityGb =
    budgetedGpuMemory?.independent ??
    Math.round(rawDedicatedCapacityGb * budget * 100) / 100;
  // A non-finite RAM figure would NaN the ceiling, which classifyMemoryFit reads as no verdict.
  const systemRamTotalGb =
    Number.isFinite(input.systemRamTotalGb) && input.systemRamTotalGb > 0
      ? input.systemRamTotalGb
      : 0;
  // Every, not some: a pin with a discrete card still has dedicated VRAM beside RAM.
  const sharesSystemRam = pinGoverns
    ? pinnedTotals.shared > 0 && pinnedTotals.dedicated === 0
    : input.hostSharesSystemRam;
  const singleMemoryPool = input.unifiedMemory || sharesSystemRam;
  return {
    gpuCapacityGb,
    totalCapacityGb: singleMemoryPool
      ? input.unifiedMemory && input.unifiedPoolReportedAsGpuMemory !== false
        // Apple reports the one pool as the GPU budget already.
        ? gpuCapacityGb
        // A capped iGPU/APU view must not be added to RAM nor replace it: the larger is the pool.
        : Math.max(gpuCapacityGb, systemRamTotalGb)
      // Dedicated VRAM only: the iGPU's budget is already inside systemRamTotalGb.
      : dedicatedCapacityGb + systemRamTotalGb,
    singleMemoryPool,
  };
}

export function gpuVramUsedIsPerDevice(
  devices: VramReportingDevice[],
): boolean {
  return (
    devices.length > 0 &&
    devices.every((device) => Number.isFinite(device.vram_used_gb))
  );
}

/** Null when unknown, never 0 (#7072). Prefers per-device usage; Windows ROCm cannot attribute
 * usage per card, so the backend's sum is used there (#7452). */
export function resolveGpuVramUsedGb(
  gpu: VramReportingGpu | null | undefined,
): number | null {
  const devices = gpu?.devices ?? [];
  if (gpuVramUsedIsPerDevice(devices)) {
    return devices.reduce((sum, device) => sum + (device.vram_used_gb ?? 0), 0);
  }
  const aggregate = gpu?.vram_used_gb_aggregate;
  return Number.isFinite(aggregate) ? (aggregate as number) : null;
}

/** The loader's default VRAM fraction (`_CTX_FIT_VRAM_FRACTION`). */
export const DEFAULT_VRAM_FRACTION = 0.97;
/** The loader's floor reserve (`_VRAM_FLOOR_RESERVE_MIB`), in GB. */
const VRAM_FLOOR_RESERVE_GB = 512 / 1024;

/**
 * Mirrors `_vram_usable_mib`: an ABSOLUTE reserve is subtracted from free memory, not a fraction
 * of it. The floor is capped at the default's reserve so raising the budget never returns less.
 */
export function usableFreeVramGb(
  freeGb: number,
  totalGb: number,
  fraction: number,
): number {
  // Probe readings off the wire: reject NaN/Infinity here rather than at each call site.
  if (!Number.isFinite(freeGb) || freeGb <= 0) {
    return 0;
  }
  const total = Number.isFinite(totalGb) ? totalGb : 0;
  const frac =
    Number.isFinite(fraction) && fraction > 0 && fraction <= 1 ? fraction : 1;
  if (!(total > 0)) {
    // No total to take a percentage of; the loader falls back the same way.
    return Math.max(0, freeGb * frac);
  }
  const floor = Math.min(VRAM_FLOOR_RESERVE_GB, (1 - DEFAULT_VRAM_FRACTION) * total);
  const reserve = Math.max((1 - frac) * total, floor);
  return Math.max(0, freeGb - reserve);
}

export interface FreeVramDevice {
  memoryFreeGb?: number;
  memoryTotalGb?: number;
  sharedMemory?: boolean;
  sharedMemoryHostBackedGb?: number | null;
  /** A Linux ROCm APU reports only this, and its window is already inside host RAM. */
  unifiedMemory?: boolean;
}

export interface FreeCapacityDevice extends FreeVramDevice {
  index: number;
  indexKind?: string | null;
  /** Whether free memory was reported, including a real zero. */
  memoryFreeKnown?: boolean;
}

export interface FreeCapacityInput {
  devices: FreeCapacityDevice[];
  pinnedGpuIds?: number[] | null;
  budgetFraction: number;
  unifiedMemory: boolean;
  /** True for Apple, false for a ROCm APU's BIOS-carved window. */
  unifiedPoolReportedAsGpuMemory: boolean;
  /** Already less the loader's reserve. */
  usableSystemRamGb: number;
  systemRamReserveDeficitGb: number;
  systemRamAvailableKnown?: boolean;
  loadedGpuIds?: number[] | null;
  loadedGpuIndexKind?: string | null;
}

export interface FreeCapacityGb {
  gb: number;
  known: boolean;
  reserveDeficitGb: number;
}

/**
 * On a non-Apple unified pool, free memory is the HOST's: the per-device window would fail a load
 * that fits the machine. Reports `known: false` when the host reading is missing.
 */
export function resolveFreeGpuCapacityGb({
  devices,
  pinnedGpuIds,
  budgetFraction,
  unifiedMemory,
  unifiedPoolReportedAsGpuMemory,
  usableSystemRamGb,
  systemRamReserveDeficitGb,
  systemRamAvailableKnown,
  loadedGpuIds,
  loadedGpuIndexKind,
}: FreeCapacityInput): FreeCapacityGb {
  if (unifiedMemory && !unifiedPoolReportedAsGpuMemory) {
    return {
      gb: systemRamAvailableKnown ? usableSystemRamGb : 0,
      known: systemRamAvailableKnown === true,
      reserveDeficitGb: systemRamReserveDeficitGb,
    };
  }
  const pinned =
    pinnedGpuIds && pinnedGpuIds.length > 0
      ? devices.filter((device) => pinnedGpuIds.includes(device.index))
      : devices;
  // Per device by the loader's absolute-reserve rule, aggregated counting the shared pool once.
  const residentDevices = loadedGpuIds?.length
    ? pinned.filter(
        (device) =>
          device.indexKind === loadedGpuIndexKind &&
          loadedGpuIds.includes(device.index),
      )
    : pinned;
  return {
    gb: aggregateUsableFreeVramGb(pinned, budgetFraction),
    known:
      pinned.length > 0 &&
      pinned.every((device) => device.memoryFreeKnown === true),
    reserveDeficitGb: aggregateVramReserveDeficitGb(
      residentDevices,
      budgetFraction,
    ),
  };
}

export function aggregateVramReserveDeficitGb(
	devices: FreeVramDevice[],
	fraction: number,
): number {
	let independent = 0;
	let shared: number | undefined;
	for (const device of devices) {
		const total = device.memoryTotalGb ?? 0;
		const free = device.memoryFreeGb ?? 0;
		if (
			!Number.isFinite(total) ||
			total <= 0 ||
			!Number.isFinite(free) ||
			free < 0
		)
			continue;
		const reserve = total - usableFreeVramGb(total, total, fraction);
		const deficit = Math.max(0, reserve - free);
		const hostBacked = device.sharedMemoryHostBackedGb;
		if (
			sharesHostMemory(device) &&
			(hostBacked == null ||
				!Number.isFinite(hostBacked) ||
				hostBacked < 0 ||
				hostBacked >= total)
		) {
			// One pool becomes usable as soon as any of its views clears its reserve.
			shared = Math.min(shared ?? deficit, deficit);
		} else {
			independent += deficit;
		}
	}
	return independent + (shared ?? 0);
}

/**
 * Counts a shared host pool once, as the totals do. A shared device can report total 0, making
 * usableFreeVramGb skip the reserve. Partially shared APUs keep their reserved segment additive.
 */
export function aggregateUsableFreeVramGb(
  devices: FreeVramDevice[],
  fraction: number,
): number {
  const usable = (device: FreeVramDevice) =>
    usableFreeVramGb(device.memoryFreeGb ?? 0, device.memoryTotalGb ?? 0, fraction);
  let dedicated = 0;
  let reserved = 0;
  let partialSharedDemand = 0;
  let partialSharedFree = 0;
  let fullySharedDemand = 0;
  let fullySharedFree = 0;
  for (const device of devices) {
    const deviceUsable = usable(device);
    if (!sharesHostMemory(device)) {
      dedicated += deviceUsable;
      continue;
    }
    const total =
      Number.isFinite(device.memoryTotalGb) && (device.memoryTotalGb as number) > 0
        ? (device.memoryTotalGb as number)
        : 0;
    if (total === 0) {
      fullySharedDemand = Math.max(fullySharedDemand, deviceUsable);
      fullySharedFree = Math.max(fullySharedFree, deviceUsable);
      continue;
    }
    const reportedHostBacked = device.sharedMemoryHostBackedGb;
    const hostBacked =
      Number.isFinite(reportedHostBacked) &&
      (reportedHostBacked as number) >= 0
        ? Math.min(total, reportedHostBacked as number)
        : total;
    const deviceReserved = total - hostBacked;
    const deviceCapacity = usableFreeVramGb(total, total, fraction);
    const deviceSharedCapacity = Math.max(0, deviceCapacity - deviceReserved);
    const deviceSharedDemand = Math.min(deviceUsable, deviceSharedCapacity);
    reserved += Math.min(
      deviceReserved,
      Math.max(0, deviceUsable - deviceSharedDemand),
    );
    if (deviceReserved > 0) {
      partialSharedDemand += deviceSharedDemand;
      const free =
        Number.isFinite(device.memoryFreeGb) && (device.memoryFreeGb as number) > 0
          ? (device.memoryFreeGb as number)
          : 0;
      const deviceSharedFree = Math.min(hostBacked, free);
      partialSharedFree = Math.max(partialSharedFree, deviceSharedFree);
    } else {
      fullySharedDemand = Math.max(fullySharedDemand, deviceSharedDemand);
      fullySharedFree = Math.max(fullySharedFree, deviceSharedDemand);
    }
  }
  const sharedFree = Math.max(partialSharedFree, fullySharedFree);
  // Round back to 2dp, as the totals aggregate does.
  return (
    Math.round(
      (dedicated +
        reserved +
        Math.min(sharedFree, partialSharedDemand + fullySharedDemand)) *
        100,
    ) / 100
  );
}
