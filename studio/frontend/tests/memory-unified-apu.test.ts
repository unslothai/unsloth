// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A ROCm APU shares one pool like Apple Silicon; the panel must read the device flag.

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

import {
  aggregateUsableFreeVramGb,
  resolveFreeGpuCapacityGb,
  resolveMemoryCapacityGb,
} from "../src/hooks/gpu-vram.ts";

const PANEL = new URL(
  "../src/features/model-picker/components/model-config-page.tsx",
  import.meta.url,
);

const APU = { memoryTotalGb: 48, sharedMemory: true };
const APU_HOST = {
  hostGpuTotalGb: 48,
  hostSharesSystemRam: true,
  systemRamTotalGb: 96,
  pinnedDevices: [],
  hostDevices: [APU],
};

test("an APU's pool is counted once, not as VRAM plus RAM", () => {
  const unified = resolveMemoryCapacityGb({ ...APU_HOST, unifiedMemory: true });
  assert.equal(unified.singleMemoryPool, true);
  assert.ok(
    unified.totalCapacityGb <= 96,
    `one pool cannot exceed the machine's 96 GiB, got ${unified.totalCapacityGb}`,
  );
});

test("reading the platform instead of the hardware double counts the pool", () => {
  const asDiscrete = resolveMemoryCapacityGb({ ...APU_HOST, unifiedMemory: false });
  const unified = resolveMemoryCapacityGb({ ...APU_HOST, unifiedMemory: true });
  assert.notEqual(
    asDiscrete.totalCapacityGb,
    unified.totalCapacityGb,
    "the unified flag no longer changes the ceiling on an APU, so the panel " +
      "reading the wrong flag would now be undetectable",
  );
  assert.ok(
    asDiscrete.totalCapacityGb > unified.totalCapacityGb,
    `treating one pool as two must overstate the ceiling; got ` +
      `${asDiscrete.totalCapacityGb} vs ${unified.totalCapacityGb}`,
  );
});

test("the panel passes the hardware signal, not the platform one", () => {
  // The fix is in a .tsx this runner cannot render, so it is asserted on source.
  const source = readFileSync(PANEL, "utf8");
  const call = source.match(/resolveMemoryCapacityGb\(\{[\s\S]*?\n\s*\}\)/);
  assert.ok(call, "resolveMemoryCapacityGb is no longer called here");
  assert.match(
    call[0],
    /unifiedMemory:\s*hasUnifiedMemory/,
    "the capacity call must take the general unified-memory signal. Passing the " +
      "Apple-only one charges a ROCm APU's single pool as VRAM plus host RAM.",
  );
  assert.match(
    source,
    /const hasUnifiedMemory[\s\S]{0,900}?device\.unifiedMemory === true/,
    "hasUnifiedMemory must be derived from the backend's per-device flag",
  );
});

test("an old backend that never sends unified_memory keeps the old behaviour", () => {
  // An older backend lacks unified_memory, so this collapses to the old appleSilicon check.
  const legacyDevices: { memory_total_gb: number; unified_memory?: boolean }[] = [
    { memory_total_gb: 24 },
  ];
  const probed = legacyDevices.some((d) => d.unified_memory === true);
  assert.equal(probed, false, "a missing key must not read as unified");
  for (const appleSilicon of [false, true]) {
    assert.equal(
      probed || appleSilicon,
      appleSilicon,
      `old backend with appleSilicon=${appleSilicon} changed answer`,
    );
  }
});

test("a real discrete card is unaffected by the change", () => {
  const discrete = resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostDevices: [{ memoryTotalGb: 24, sharedMemory: false }],
    hostGpuTotalGb: 24,
    hostSharesSystemRam: false,
    systemRamTotalGb: 64,
    unifiedMemory: false,
  });
  assert.equal(discrete.singleMemoryPool, false);
  assert.equal(discrete.gpuCapacityGb, 24);
  assert.equal(discrete.totalCapacityGb, 88);
});

test("a discrete pin on a mixed APU host keeps system RAM as a pool beside it", () => {
  // A host-wide some() would treat a discrete pin on a mixed host as one pool.
  const APU = { memoryTotalGb: 48, sharedMemory: true };
  const DGPU = { memoryTotalGb: 16, sharedMemory: false };
  const base = {
    pinnedDevices: [DGPU],
    hostDevices: [APU, DGPU],
    hostGpuTotalGb: 64,
    hostSharesSystemRam: true,
    systemRamTotalGb: 128,
    gpuBudgetFraction: 0.97,
  };
  const scoped = resolveMemoryCapacityGb({ ...base, unifiedMemory: false });
  const hostWide = resolveMemoryCapacityGb({ ...base, unifiedMemory: true });
  assert.equal(scoped.singleMemoryPool, false);
  assert.ok(
    scoped.totalCapacityGb > 100,
    `a discrete pin must keep host RAM; got ${scoped.totalCapacityGb} GiB`,
  );
  assert.ok(
    hostWide.totalCapacityGb < scoped.totalCapacityGb,
    "the host-wide flag must be the one that throws RAM away, or this test is moot",
  );
});

test("the panel scopes the unified flag to the pinned devices", () => {
  const source = readFileSync(PANEL, "utf8");
  const decl = source.match(/const hasUnifiedMemory = useMemo\(\(\) => \{[\s\S]*?\}, \[[^\]]*\]\);/);
  assert.ok(decl, "hasUnifiedMemory is no longer a scoped useMemo");
  assert.match(
    decl[0],
    /pinnedGpuIds[\s\S]*includes\(device\.index\)/,
    "hasUnifiedMemory must narrow to the pinned devices before calling .some()",
  );
  assert.match(
    decl[0],
    /isAppleUnifiedMemory/,
    "the Apple fallback must stay, for the window before the per-device probe lands",
  );
});

test("a mixed governing set is not unified, pinned or unpinned", () => {
  // One independent-memory device in the set means VRAM beside RAM, so not one pool.
  const APU = { memoryTotalGb: 48, sharedMemory: true, unifiedMemory: true };
  const DGPU = { memoryTotalGb: 16, sharedMemory: false, unifiedMemory: false };
  const mixed = [APU, DGPU];

  const unified = (governing: typeof mixed) =>
    governing.length > 0 && governing.every((d) => d.unifiedMemory === true);

  assert.equal(unified(mixed), false, "a mixed set must not read as unified");
  assert.equal(unified([APU]), true, "an APU-only set is unified");
  assert.equal(unified([DGPU]), false, "a discrete-only set is not unified");
  // `[].every()` is true, which would make a host with no devices at all read as
  // a unified-memory machine.
  assert.equal(unified([]), false, "the empty set must not read as unified");

  const cap = resolveMemoryCapacityGb({
    pinnedDevices: mixed,
    hostDevices: mixed,
    hostGpuTotalGb: 64,
    hostSharesSystemRam: true,
    systemRamTotalGb: 128,
    unifiedMemory: unified(mixed),
    gpuBudgetFraction: 0.97,
  });
  assert.ok(
    cap.totalCapacityGb > 100,
    `a mixed pin must keep host RAM beside the discrete card; got ${cap.totalCapacityGb} GiB`,
  );
});

test("the panel asks whether EVERY governing device is unified", () => {
  const source = readFileSync(PANEL, "utf8");
  const decl = source.match(/const hasUnifiedMemory = useMemo\(\(\) => \{[\s\S]*?\}, \[[^\]]*\]\);/);
  assert.ok(decl, "hasUnifiedMemory is no longer a scoped useMemo");
  assert.match(
    decl[0],
    /\.every\(\(device\) => device\.unifiedMemory === true\)/,
    "must be .every(): .some() marks a mixed APU-plus-discrete set unified and " +
      "throws away the system RAM beside the discrete card",
  );
  assert.match(
    decl[0],
    /governing\.length === 0/,
    "the empty set needs an explicit guard, since [].every() is true",
  );
});

test("a ROCm APU's ceiling is system RAM, not its GPU-visible window", () => {
  // A ROCm APU's memory_total_gb is a BIOS window; system RAM is the real ceiling.
  // sharedMemory false is what Linux reports, so the unified flag alone classifies it.
  const APU = { memoryTotalGb: 48, sharedMemory: false };
  const base = {
    pinnedDevices: [],
    hostDevices: [APU],
    hostGpuTotalGb: 48,
    hostSharesSystemRam: false,
    systemRamTotalGb: 96,
    gpuBudgetFraction: 0.97,
    unifiedMemory: true,
  };
  const rocm = resolveMemoryCapacityGb({
    ...base,
    unifiedPoolReportedAsGpuMemory: false,
  });
  assert.equal(rocm.singleMemoryPool, true, "still one pool");
  assert.equal(
    rocm.totalCapacityGb,
    96,
    "the pool is the machine's RAM, not the window the BIOS carved out of it",
  );
  assert.equal(rocm.gpuCapacityGb, 46.56);
});

test("Apple is unchanged, since its GPU figure already is the whole pool", () => {
  // Apple must keep the budgeted GPU figure, or the user's VRAM Budget is dropped.
  const MAC = { memoryTotalGb: 96, sharedMemory: false };
  const base = {
    pinnedDevices: [],
    hostDevices: [MAC],
    hostGpuTotalGb: 96,
    hostSharesSystemRam: false,
    systemRamTotalGb: 96,
    gpuBudgetFraction: 0.97,
    unifiedMemory: true,
  };
  const explicit = resolveMemoryCapacityGb({
    ...base,
    unifiedPoolReportedAsGpuMemory: true,
  });
  const byDefault = resolveMemoryCapacityGb(base);
  assert.equal(explicit.totalCapacityGb, 93.12);
  assert.equal(
    byDefault.totalCapacityGb,
    explicit.totalCapacityGb,
    "omitting the new flag must mean Apple, or existing callers silently move",
  );
  assert.notEqual(
    explicit.totalCapacityGb,
    96,
    "Apple's ceiling must still respect the VRAM Budget",
  );
});

test("the panel tells the resolver which kind of unified memory it has", () => {
  const source = readFileSync(PANEL, "utf8");
  const call = source.match(/resolveMemoryCapacityGb\(\{[\s\S]*?\n\s*\}\)/);
  assert.ok(call, "resolveMemoryCapacityGb is no longer called here");
  assert.match(
    call[0],
    /unifiedPoolReportedAsGpuMemory:\s*isAppleUnifiedMemory/,
    "only the Apple half may be read as the whole pool. Passing the general " +
      "signal here takes a ROCm APU's carved window as the machine's ceiling.",
  );
});

test("a Linux APU beside a discrete card is not independent VRAM", () => {
  // A Linux ROCm APU reports sharedMemory false, so its window was double counted with RAM.
  const APU = { memoryTotalGb: 48, sharedMemory: false, unifiedMemory: true };
  const DGPU = { memoryTotalGb: 16, sharedMemory: false, unifiedMemory: false };
  const mixed = [APU, DGPU];
  const r = resolveMemoryCapacityGb({
    pinnedDevices: mixed,
    hostDevices: mixed,
    hostGpuTotalGb: 64,
    hostSharesSystemRam: false,
    systemRamTotalGb: 128,
    gpuBudgetFraction: 0.97,
    unifiedMemory: false,
    unifiedPoolReportedAsGpuMemory: false,
  });
  // Only the discrete card is memory BESIDE system RAM: 16 * 0.97 + 128.
  assert.equal(r.totalCapacityGb, 143.52);
  assert.ok(
    r.totalCapacityGb <= 128 + 16,
    `the ceiling cannot exceed RAM plus the one real card; got ${r.totalCapacityGb}`,
  );
});

test("the unified flag is read from the device, not just shared_memory", () => {
  // Windows sends shared_memory and Linux unified_memory for the same silicon.
  const base = {
    pinnedDevices: [] as never[],
    hostGpuTotalGb: 48,
    hostSharesSystemRam: false,
    systemRamTotalGb: 96,
    gpuBudgetFraction: 0.97,
    unifiedMemory: false,
  };
  const viaShared = resolveMemoryCapacityGb({
    ...base,
    hostDevices: [{ memoryTotalGb: 48, sharedMemory: true }],
  });
  const viaUnified = resolveMemoryCapacityGb({
    ...base,
    hostDevices: [{ memoryTotalGb: 48, sharedMemory: false, unifiedMemory: true }],
  });
  assert.deepEqual(
    viaUnified,
    viaShared,
    "Windows and Linux must price the same APU identically",
  );
});

test("a Linux APU's FREE memory is the pool's, not the window's", () => {
  // Free capacity on a ROCm APU must be measured against host RAM, not the carved window.
  const APU = { memoryFreeGb: 44, memoryTotalGb: 48, sharedMemory: false, unifiedMemory: true };
  const freeVram = aggregateUsableFreeVramGb([APU], 0.97);
  const usableSystemRamGb = 62; // 64 GiB available, less the loader's 2 GiB headroom

  const pooledFree = Math.max(freeVram, usableSystemRamGb);
  assert.ok(
    pooledFree >= usableSystemRamGb,
    `the pool's free memory cannot be smaller than the host's; got ${pooledFree}`,
  );
  assert.ok(freeVram < 60, `the window must be the smaller figure; got ${freeVram}`);
  assert.ok(pooledFree > 60, `the pool must hold the load; got ${pooledFree}`);
});

test("two views of one host pool are not counted as two pools", () => {
  // An APU and a Vulkan iGPU are two views of the same memory; count it once.
  const APU = { memoryFreeGb: 40, memoryTotalGb: 48, sharedMemory: false, unifiedMemory: true };
  const IGPU = { memoryFreeGb: 40, memoryTotalGb: 48, sharedMemory: true };
  const folded = aggregateUsableFreeVramGb([APU, IGPU], 0.97);
  const asDedicated = aggregateUsableFreeVramGb(
    [{ ...APU, unifiedMemory: false }, IGPU],
    0.97,
  );
  assert.equal(folded, 38.56);
  assert.ok(
    asDedicated > folded,
    "treating the APU as its own memory must be the larger, wrong answer, " +
      `or this test is measuring nothing; got ${asDedicated} vs ${folded}`,
  );
  assert.ok(folded < asDedicated / 1.5);
});

// Run the rule from gpu-vram.ts rather than matching panel text, which broke on rewrites.
const APU_POOL = {
  devices: [
    {
      index: 0,
      indexKind: "physical",
      // The BIOS-carved window: 48 GiB visible to the GPU on a 96 GiB machine.
      memoryTotalGb: 48,
      memoryFreeGb: 44,
      memoryFreeKnown: true,
      sharedMemory: false,
      unifiedMemory: true,
    },
  ],
  budgetFraction: 0.97,
  unifiedMemory: true,
  unifiedPoolReportedAsGpuMemory: false,
  // 64 GiB free on the host, less the loader's 2 GiB headroom.
  usableSystemRamGb: 62,
  systemRamReserveDeficitGb: 0,
  systemRamAvailableKnown: true,
};

const PANEL_FREE_CAPACITY_CALL =
  /resolveFreeGpuCapacityGb\(\{[\s\S]*?\n\s*\}\)/;
const PANEL_UNIFIED_FLAG = /unifiedMemory:\s*hasUnifiedMemory/;
const PANEL_APPLE_FLAG =
  /unifiedPoolReportedAsGpuMemory:\s*isAppleUnifiedMemory/;
const PANEL_HOST_VIEW = /usableSystemRamGb:\s*\w/;

function carvedWindowGb(): number {
  return aggregateUsableFreeVramGb(APU_POOL.devices, APU_POOL.budgetFraction);
}

test("a non-Apple unified pool is measured against the host, not the window", () => {
  const carved = carvedWindowGb();
  const pooled = resolveFreeGpuCapacityGb(APU_POOL);
  assert.ok(
    carved < 60,
    `the carved window must be the smaller figure, or this proves nothing; got ${carved}`,
  );
  assert.equal(
    pooled.gb,
    62,
    "a non-Apple unified pool's free capacity must be the host view; the " +
      "carved window cannot answer a whole-load question",
  );
  assert.equal(pooled.known, true);
  assert.ok(pooled.gb > 60, `the pool must hold the load; got ${pooled.gb}`);
});

test("an unread host RAM figure is reported unread, not replaced by the window", () => {
  const unread = resolveFreeGpuCapacityGb({
    ...APU_POOL,
    systemRamAvailableKnown: false,
    usableSystemRamGb: 0,
  });
  assert.equal(
    unread.known,
    false,
    "with no host reading the panel has no free figure for this pool and must " +
      "say so; answering with the carved window is the same double count in " +
      "friendlier clothes",
  );
  assert.notEqual(
    unread.gb,
    carvedWindowGb(),
    "the carved window must not stand in for the missing host reading",
  );
});

test("Apple and discrete hosts still answer from their own free VRAM", () => {
  const apple = resolveFreeGpuCapacityGb({
    ...APU_POOL,
    unifiedPoolReportedAsGpuMemory: true,
  });
  const discrete = resolveFreeGpuCapacityGb({
    ...APU_POOL,
    unifiedMemory: false,
    unifiedPoolReportedAsGpuMemory: false,
  });
  assert.equal(apple.gb, carvedWindowGb());
  assert.equal(discrete.gb, carvedWindowGb());
  assert.equal(apple.known, true);
});

test("the panel hands the free-capacity rule its own inputs", () => {
  const source = readFileSync(PANEL, "utf8");
  const call = source.match(PANEL_FREE_CAPACITY_CALL);
  assert.ok(
    call,
    "the panel no longer resolves free capacity through the shared rule",
  );
  assert.match(
    call[0],
    PANEL_UNIFIED_FLAG,
    "the free-capacity call must take the general unified-memory signal",
  );
  assert.match(
    call[0],
    PANEL_APPLE_FLAG,
    "only Apple's GPU figure is the whole pool; passing the general signal " +
      "here would measure a ROCm APU against its carved window again",
  );
  assert.match(
    call[0],
    PANEL_HOST_VIEW,
    "the host view must reach the rule, or the unified branch has nothing to " +
      "answer with",
  );
});
