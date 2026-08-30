// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Split out of use-system.ts so the VRAM rules can be unit tested without pulling
// the auth and React graph in behind them. Typed structurally, so SystemGpuInfo and
// GpuDevice satisfy these without importing them.

export interface VramReportingDevice {
  vram_used_gb?: number;
}

export interface MemoryTotalDevice {
  memory_total_gb?: number;
  /** True when the reported GPU budget comes from shared system memory. */
  shared_memory?: boolean;
  /** host-backed portion of the shared pool; the rest is reserved GPU memory. */
  shared_memory_host_backed_gb?: number | null;
}

export interface GpuMemoryTotalsGb {
  dedicated: number;
  shared: number;
  total: number;
}

export interface VramReportingGpu {
  devices?: VramReportingDevice[];
  /** Used VRAM across the visible GPUs when no single device's usage could be
   * attributed. Windows ROCm only; null everywhere else. See #7452. */
  vram_used_gb_aggregate?: number | null;
}

/** Sum dedicated VRAM while counting a shared host-memory pool only once.
 *
 * Devices arrive rounded to 2dp, so summing them reintroduces float error
 * (three B200s at 179.06 give 537.1800000000001) and not every caller rounds
 * again before printing. Round back to the precision they arrived with. */
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
      .filter((device) => !device.shared_memory)
      .reduce((sum, device) => sum + size(device), 0),
  );
  const sharedPool = devices
    .filter((device) => device.shared_memory)
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
          hostBacked: Math.max(totals.hostBacked, hostBacked),
          reserved:
            totals.reserved + (hostBackedKnown ? total - hostBacked : 0),
        };
      },
      { hostBacked: 0, reserved: 0 },
    );
  const shared = roundToDevicePrecision(sharedPool.hostBacked);
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
  /** The backend's per-device `unified_memory`, for a ROCm APU.
   *
   *  Separate from `sharedMemory` because the backend reports them separately and
   *  they do not coincide: `hardware.py` sets `shared_memory` only on Windows, so
   *  on Linux the very same APU arrives as `unified_memory: true,
   *  shared_memory: false`. Reading only `sharedMemory` there counts the APU's
   *  carved window as VRAM standing BESIDE system RAM, when it is a view INTO it.
   *
   *  Both flags mean the same thing for capacity, which is why
   *  `sharesHostMemory` below folds them together rather than either one being
   *  taught about the other. */
  unifiedMemory?: boolean;
}

/** Whether this device's memory is a view into host RAM rather than beside it.
 *  The one question capacity cares about; the two flags are how the backend
 *  happens to report it on two platforms. */
function sharesHostMemory(device: {
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
        Math.min(sharedPool, partialSharedDemand + fullySharedDemand),
    ),
    independent: roundToDevicePrecision(dedicated + reserved),
  };
}

export interface MemoryCapacityInput {
  /** The devices a pin names, or empty when the load may use the whole host. */
  pinnedDevices: MemoryCapacityDevice[];
  hostDevices?: MemoryCapacityDevice[];
  /** Aggregate GPU budget of the whole inventory, for the unpinned case. */
  hostGpuTotalGb: number;
  /** The same aggregate with shared-memory devices left out, for the unpinned case.
   *  Only the dedicated cards are memory BESIDE system RAM; an iGPU's budget is a
   *  capped view of that same RAM, so adding both to reach a machine-wide ceiling
   *  counts the shared bytes twice. Absent means "no shared device", i.e. the same
   *  figure as `hostGpuTotalGb`. */
  hostDedicatedGpuTotalGb?: number;
  /** Whether ANY device on the host reports a shared pool. Only consulted when
   *  nothing is pinned; a pin answers for itself. */
  hostSharesSystemRam: boolean;
  systemRamTotalGb: number;
  /** One pool for everything, whatever the shared-memory flags say. True on Apple
   *  Silicon and on a ROCm APU, which is why the pool's SIZE is a separate question
   *  below: the two report it differently. */
  unifiedMemory: boolean;
  /** Whether the GPU figure already describes the whole pool.
   *
   *  True on Apple, where `memory_total_gb` IS the machine's unified memory, so the
   *  budgeted GPU capacity is the ceiling and adding RAM would double count.
   *
   *  False on a ROCm APU, where the GPU figure is a BIOS-carved window onto system
   *  RAM -- 48 GiB of a 96 GiB machine is typical -- and taking it as the ceiling
   *  discards the rest of the pool the weights actually load into. The backend says
   *  so in as many words (`llama_cpp.py::_available_system_memory_mib`: "On a
   *  unified-memory APU this, not the ROCm-reported VRAM, is the real ceiling").
   *
   *  Defaults to true, which is the answer for every caller that predates the ROCm
   *  case and passed `unifiedMemory` meaning Apple. */
  unifiedPoolReportedAsGpuMemory?: boolean;
  /** VRAM Budget: the fraction of each GPU a load may claim, from the setting beside
   *  this row. 1 (or absent) means the whole card. Applied to the GPU figure only --
   *  it caps what llama.cpp is allowed to take, not how much RAM the host has. */
  gpuBudgetFraction?: number;
}

/** What a prospective load may draw on: the GPU pool, and the ceiling once layers
 *  spill to host RAM.
 *
 *  System RAM is added only where it is a pool BESIDE the GPU budget. A Vulkan
 *  iGPU's reported budget is already a capped view of that same RAM, so adding it
 *  would count the bytes twice. That question is per device, not per host: a mixed
 *  inventory pairs a discrete card with an iGPU, and a pin naming only the discrete
 *  card can still spill into RAM the iGPU would have been sharing.
 *
 *  Both figures are 0 when nothing was probed, which callers read as "no verdict"
 *  rather than as a fit. */
export function resolveMemoryCapacityGb(input: MemoryCapacityInput): {
  gpuCapacityGb: number;
  totalCapacityGb: number;
  /** One pool for both figures, so a caller shows one number rather than two. */
  singleMemoryPool: boolean;
} {
  const pinnedTotals = gpuMemoryTotalsGb(
    input.pinnedDevices.map((device) => ({
      memory_total_gb: device.memoryTotalGb,
      // Folded, not passed through: a Linux ROCm APU reports shared_memory false
      // and unified_memory true, and counting its carved window as dedicated VRAM
      // added 46.56 GiB of capacity that is already inside system RAM.
      shared_memory: sharesHostMemory(device),
      shared_memory_host_backed_gb: device.sharedMemoryHostBackedGb,
    })),
  );
  // One flag for both answers, so the capacity and the pool question cannot
  // disagree about which devices they are describing.
  const pinGoverns = pinnedTotals.total > 0;
  // The host figure comes from aggregateGpuMemoryTotalGb on the caller's side, which
  // guards itself, but this is a plain number on a public interface and a non-finite
  // one here would make every figure below NaN.
  const hostGpuTotalGb =
    Number.isFinite(input.hostGpuTotalGb) && input.hostGpuTotalGb > 0
      ? input.hostGpuTotalGb
      : 0;
  const rawGpuCapacityGb = pinGoverns ? pinnedTotals.total : hostGpuTotalGb;
  // The dedicated-only figure for the same governing set. A pin answers for itself;
  // unpinned takes the caller's, falling back to the full aggregate, which is the
  // right answer on every host that has no shared device at all.
  const rawDedicatedCapacityGb = pinGoverns
    ? pinnedTotals.dedicated
    : Number.isFinite(input.hostDedicatedGpuTotalGb) &&
        (input.hostDedicatedGpuTotalGb as number) >= 0
      ? (input.hostDedicatedGpuTotalGb as number)
      : hostGpuTotalGb;
  // The budget is what the next load is ALLOWED to claim, so it is the capacity a
  // verdict should be measured against: at 80% a 20 GB footprint on a 24 GB card is
  // over the line the slider draws, and reading the raw total called it comfortable.
  // Guarded rather than trusted: a 0 or a missing value would silently zero the
  // capacity, which every caller reads as "nothing probed".
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
      // Folded, not passed through: a Linux ROCm APU reports shared_memory false
      // and unified_memory true, and counting its carved window as dedicated VRAM
      // added 46.56 GiB of capacity that is already inside system RAM.
      shared_memory: sharesHostMemory(device),
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
  // Same guard as the device totals: psutil's figure arrives over the wire too, and a
  // non-finite one would turn the ceiling below into NaN, which classifyMemoryFit
  // reads as no verdict at all rather than as the RAM the machine plainly has.
  const systemRamTotalGb =
    Number.isFinite(input.systemRamTotalGb) && input.systemRamTotalGb > 0
      ? input.systemRamTotalGb
      : 0;
  // Every, for the same reason the host-level flag uses every: a pin naming a
  // discrete card alongside an iGPU still has dedicated VRAM beside system RAM, and
  // calling that one pool hides the GPU verdict on the only figure that would catch
  // a fixed placement too large for the card.
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
        // A Vulkan iGPU's budget, and a ROCm APU's, are both a CAPPED view of system
        // RAM, so RAM must not be added to it -- but it must not replace it either.
        // The pool's real size is the RAM, and taking the capped figure as the
        // machine's whole capacity called a 20 GB CPU-offloaded load impossible on a
        // 91 GiB host because the iGPU was allowed 12. The larger of the two is the
        // pool; on a mixed inventory that under-counts a discrete card sitting beside
        // it, which is the side that refuses a load rather than admitting one that
        // cannot run.
        : Math.max(gpuCapacityGb, systemRamTotalGb)
      // Dedicated VRAM, not the whole GPU figure: on a mixed inventory the iGPU's
      // budget is already inside systemRamTotalGb, and adding it again inflated the
      // ceiling in the direction that admits a load. Identical to gpuCapacityGb on
      // any host without a shared device, which is every discrete-only machine.
      : dedicatedCapacityGb + systemRamTotalGb,
    singleMemoryPool,
  };
}

/** Whether every device reports its own usage, so each row and their sum are real. */
export function gpuVramUsedIsPerDevice(
  devices: VramReportingDevice[],
): boolean {
  return (
    devices.length > 0 &&
    devices.every((device) => Number.isFinite(device.vram_used_gb))
  );
}

/** Used VRAM across the GPUs, or null when it is genuinely unknown.
 *
 * Per-device usage is preferred. On Windows ROCm nothing keys the LUID usage
 * counters to torch ordinals, so a usage that fits more than one card cannot be
 * attributed to either and every device reports unknown -- which is idle and every
 * small model on an asymmetric pair. The sum does not depend on that attribution,
 * so the backend still reports it, and rendering Unknown for a figure it already
 * has is what #7452 was.
 *
 * Never falls back to 0: a fabricated 0 used / full free is the #7072 symptom. */
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
 * Free VRAM a load may actually claim on one card, by the loader's own rule.
 *
 * `_vram_usable_mib` subtracts an ABSOLUTE reserve from what is free -- it does not
 * scale free memory by the fraction. The two agree only on an idle card. On a 24 GB
 * card with 10 GB free at an 80% budget the loader offers 5.2 GB while a
 * multiplication says 8, so a 7 GB load looked comfortable and will be fitted down.
 *
 * The floor keeps the budget monotonic: capped at the default's own reserve so that
 * nudging the slider up never hands back less, which a flat 512 MiB would do on any
 * card under about 17 GB.
 */
export function usableFreeVramGb(
  freeGb: number,
  totalGb: number,
  fraction: number,
): number {
  // Every argument is a probe reading off the wire. A NaN or an Infinity propagates
  // through the arithmetic below into a figure a fit verdict is drawn from, so it is
  // rejected here rather than at each of the four call sites.
  if (!Number.isFinite(freeGb) || freeGb <= 0) {
    return 0;
  }
  const total = Number.isFinite(totalGb) ? totalGb : 0;
  const frac =
    Number.isFinite(fraction) && fraction > 0 && fraction <= 1 ? fraction : 1;
  if (!(total > 0)) {
    // No total to take a percentage of; the free reading is the only scale there is,
    // and the loader falls back the same way.
    return Math.max(0, freeGb * frac);
  }
  const floor = Math.min(VRAM_FLOOR_RESERVE_GB, (1 - DEFAULT_VRAM_FRACTION) * total);
  const reserve = Math.max((1 - frac) * total, floor);
  return Math.max(0, freeGb - reserve);
}

/** A device as the free-VRAM aggregate sees it. Structural, so a SystemGpuDevice fits. */
export interface FreeVramDevice {
  memoryFreeGb?: number;
  memoryTotalGb?: number;
  /** True when this device's budget is a capped view of the host's own RAM. */
  sharedMemory?: boolean;
  /** Host-backed portion of memoryTotalGb when sharedMemory is true. */
  sharedMemoryHostBackedGb?: number | null;
  /** The backend's per-device `unified_memory`. Same reason it exists on
   *  MemoryCapacityDevice: a Linux ROCm APU reports `unified_memory: true,
   *  shared_memory: false`, and summing its window into the DEDICATED free total
   *  counts memory that is already inside the host's RAM. */
  unifiedMemory?: boolean;
}

/**
 * Free memory a prospective load may claim across an inventory, counting a shared
 * host-memory pool only once.
 *
 * The same rule `aggregateGpuMemoryTotalGb` applies to totals, and for the same
 * reason: on a discrete-plus-iGPU host the iGPU's free reading IS the host's free RAM
 * seen through a cap, so adding it to the discrete card's free VRAM counts those
 * bytes twice and hands the free-memory verdict a capacity the machine does not have.
 * Summing blindly is worse here than for totals, because a shared device can report a
 * total of 0 (`hardware.py` zeroes it for an iGPU on one path), and `usableFreeVramGb`
 * then falls back to the raw free reading with no reserve subtracted at all.
 *
 * Fully shared devices are views of one pool, so only the largest is taken. A
 * partially shared APU also has an independent reserved segment, which remains
 * additive while its host-backed demand is capped by the common pool.
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
  // Devices arrive rounded to 2dp and the reserve is a fraction of them, so the sum
  // reintroduces float error the same way the totals aggregate does.
  return (
    Math.round(
      (dedicated +
        reserved +
        Math.min(sharedFree, partialSharedDemand + fullySharedDemand)) *
        100,
    ) / 100
  );
}
