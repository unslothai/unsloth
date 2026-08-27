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
  // And the general signal must come from the probed devices rather than being
  // aliased back to the platform check.
  assert.match(
    source,
    /const hasUnifiedMemory\s*=\s*inferenceGpu\.unifiedMemory/,
    "hasUnifiedMemory must be derived from the backend's per-device flag",
  );
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
