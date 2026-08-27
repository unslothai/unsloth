// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A ROCm APU shares one memory pool between the GPU and the rest of the system,
// exactly as Apple Silicon does. The Load Model panel decided that question from
// `usePlatformStore(s => s.appleSilicon)`, so on an APU it charged the pool as
// discrete VRAM PLUS host RAM: the same bytes counted twice, and a verdict of
// "fits" for a load that cannot open.
//
// The signal was already there. `use-gpu-info.ts` derives `unifiedMemory` from
// the backend's per-device `unified_memory` flag, which the ROCm probe sets, and
// the Hub memory bar already abstains on it. Only the panel was reading the
// platform instead of the hardware.
//
// These tests are on `resolveMemoryCapacityGb` rather than on the component,
// because that is where the double count happens and it is reachable from the
// test runner. The component's part is one line: which flag it passes.

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

import { resolveMemoryCapacityGb } from "../src/hooks/gpu-vram.ts";

const PANEL = new URL(
  "../src/features/model-picker/components/model-config-page.tsx",
  import.meta.url,
);

// A Strix Halo style APU: one 96 GiB pool, reported as GPU-visible memory and as
// system RAM, because it is the same silicon.
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
  // The ceiling is the machine's memory, not the machine's memory plus a copy of
  // the part of it the GPU can see.
  assert.ok(
    unified.totalCapacityGb <= 96,
    `one pool cannot exceed the machine's 96 GiB, got ${unified.totalCapacityGb}`,
  );
});

test("reading the platform instead of the hardware double counts the pool", () => {
  // The state the panel was in on an APU: unifiedMemory false, because the host
  // is not Apple. This is the bug, pinned as a contrast so the assertion above
  // is measuring something rather than restating a default.
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
  // The fix itself is one line in a 3,400-line .tsx that this runner cannot
  // render, so it is asserted on the source. Without this the two tests above
  // pass on a panel that still reads appleSilicon: they pin what
  // resolveMemoryCapacityGb does with each flag, not which flag it is handed.
  const source = readFileSync(PANEL, "utf8");
  const call = source.match(/resolveMemoryCapacityGb\(\{[\s\S]*?\n\s*\}\)/);
  assert.ok(call, "resolveMemoryCapacityGb is no longer called here");
  assert.match(
    call[0],
    /unifiedMemory:\s*hasUnifiedMemory/,
    "the capacity call must take the general unified-memory signal. Passing the " +
      "Apple-only one charges a ROCm APU's single pool as VRAM plus host RAM.",
  );
  // And the general signal must come from the probed DEVICES rather than being
  // aliased back to the platform check. WHICH devices, and with which quantifier,
  // are asserted separately below.
  assert.match(
    source,
    /const hasUnifiedMemory[\s\S]{0,900}?device\.unifiedMemory === true/,
    "hasUnifiedMemory must be derived from the backend's per-device flag",
  );
});

test("an old backend that never sends unified_memory keeps the old behaviour", () => {
  // Backwards compatibility for an existing install whose backend predates the
  // per-device flag. use-gpu-info derives unifiedMemory with
  // `devices.some(d => d.unified_memory === true)`, so a missing key is false,
  // and the panel's `inferenceGpu.unifiedMemory || isAppleUnifiedMemory` then
  // collapses to exactly the appleSilicon check it replaced. Neither better nor
  // worse than before, which is the requirement.
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
  // The fix must not turn an ordinary NVIDIA host into a shared-pool one. Its
  // devices do not report unified_memory, so nothing here moves.
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
  // VRAM beside RAM, which is what a discrete card actually offers.
  assert.equal(discrete.totalCapacityGb, 88);
});

// ---------------------------------------------------------------------------
// Scoping the flag to the pin (Codex P2 on 9830)

test("a discrete pin on a mixed APU host keeps system RAM as a pool beside it", () => {
  // The regression the first version of this fix introduced. A ROCm APU beside a
  // discrete card makes a HOST-WIDE `devices.some(...)` true, and passing that
  // for a pin naming only the discrete card told resolveMemoryCapacityGb the two
  // are one pool. Measured: totalCapacityGb collapsed 143.52 -> 15.52 GiB,
  // discarding 128 GiB of RAM the load can really spill into, and the panel then
  // warned "more than this machine holds" for a load that fits comfortably.
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
  // The fix is in a .tsx this runner cannot render, so it is asserted on source.
  // Without this, the capacity test above passes on a panel that still hands the
  // host-wide flag over: it pins what the resolver does, not what it is given.
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
  // The second half of the same bug. Narrowing to the pin fixed the discrete-only
  // case but `.some()` still marked a MIXED set unified: an unpinned load, or a
  // pin naming both an APU and a discrete card, reported 62.08 GiB instead of
  // 143.52 GiB. One independent-memory device in the set means there is real VRAM
  // beside system RAM, so the two are not one pool.
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

  // And the capacity that follows from it.
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

// ---------------------------------------------------------------------------
// The pool's SIZE, which is a different question from whether it is one pool
// (Codex P2 on 9830)

test("a ROCm APU's ceiling is system RAM, not its GPU-visible window", () => {
  // Classifying the APU as one pool was right. Taking the GPU figure as the size
  // of that pool was not: Apple's memory_total_gb IS the machine's unified
  // memory, while a ROCm APU's is a BIOS-carved window onto system RAM. On a
  // 96 GiB machine carving 48 GiB, the ceiling came out at 46.56 GiB, so the
  // panel warned that a 60 GiB load exceeds a machine that holds 96.
  //
  // The backend already says which one is real, in
  // llama_cpp.py::_available_system_memory_mib: "On a unified-memory APU this,
  // not the ROCm-reported VRAM, is the real ceiling: the weights load into
  // shared system RAM."
  //
  // Driven with sharedMemory false, which is what a ROCm APU reports on Linux
  // (hardware.py sets shared_memory only on Windows). That matters: it is the
  // case where the unified flag is the ONLY thing classifying the pool, so the
  // capacity comes entirely from this branch.
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
  // The GPU figure is still the budgeted window: what may sit on the GPU and how
  // much the machine holds are different numbers even when they share silicon.
  assert.equal(rocm.gpuCapacityGb, 46.56);
});

test("Apple is unchanged, since its GPU figure already is the whole pool", () => {
  // The guard on the fix. Apple must keep the budgeted GPU figure as its ceiling;
  // routing it through the RAM branch would hand back the raw 96 and quietly drop
  // the VRAM Budget the user set.
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
  // Absent must behave as Apple, so every caller written before the ROCm case
  // keeps its answer without being updated.
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
  // Source-level, like the sibling assertions: the fix is one argument in a .tsx
  // this runner cannot render, and without this the two tests above pass on a
  // panel that never passes the flag.
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
