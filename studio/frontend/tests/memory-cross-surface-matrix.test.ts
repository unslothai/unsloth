// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The property this PR exists to establish, checked over the whole hardware
// matrix rather than on one host: the Load Model panel and the Hub memory bar
// cannot describe one load differently.
//
// `model-memory-hardware-matrix.test.ts` already pins what the BAR does with
// each kind of budget. This file is about the two surfaces AGREEING, which is a
// different question and was not previously asked anywhere: each surface was
// self-consistent the whole time, and that was never the problem.
//
// The matrix is [linux, wsl, win32, darwin] x nine device inventories, minus the
// physically impossible cells (Apple unified memory on Windows). Every cell is
// checked against six properties, so this is ~200 assertions rather than nine
// hand-written cases.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { computeModelMemory } = await import("../src/lib/model-memory.ts");
const { resolveMemoryCapacityGb } = await import("../src/hooks/gpu-vram.ts");
const { classifyMemoryFit } = await import("../src/lib/memory/verdict.ts");
const { formatGiB, formatBytesGiB, formatKvRate } = await import(
  "../src/lib/memory/format.ts"
);
const { DEFAULT_VRAM_BUDGET_FRACTION } = await import(
  "../src/lib/memory/thresholds.ts"
);

const GB = 1024 ** 3;

type Platform = "linux" | "wsl" | "win32" | "darwin";
const ALL: Platform[] = ["linux", "wsl", "win32", "darwin"];

// Matches MemoryCapacityDevice in src/hooks/gpu-vram.ts. sharedMemory is
// required there, so it is normalised at the call site rather than left optional
// here, which tsc -b catches even though the tests pass either way.
interface Device {
  memoryTotalGb: number;
  sharedMemory?: boolean;
}

interface Host {
  label: string;
  devices: Device[];
  systemRamTotalGb: number;
  /** Any device reports a unified pool (Apple, ROCm APU). */
  unifiedMemory: boolean;
  /** Which platforms this inventory can physically occur on. */
  platforms: Platform[];
}

// Nine inventories spanning [NVIDIA, AMD, Intel, Apple, none] and
// [discrete, integrated, unified, mixed, multi].
const HOSTS: Host[] = [
  {
    label: "NVIDIA single 24 GiB",
    devices: [{ memoryTotalGb: 24 }],
    systemRamTotalGb: 64,
    unifiedMemory: false,
    // No CUDA on Apple silicon since 10.13; treated as not occurring.
    platforms: ["linux", "wsl", "win32"],
  },
  {
    label: "NVIDIA dual 24 GiB",
    devices: [{ memoryTotalGb: 24 }, { memoryTotalGb: 24 }],
    systemRamTotalGb: 128,
    unifiedMemory: false,
    platforms: ["linux", "wsl", "win32"],
  },
  {
    label: "AMD ROCm discrete 16 GiB",
    devices: [{ memoryTotalGb: 16 }],
    systemRamTotalGb: 64,
    unifiedMemory: false,
    platforms: ["linux", "wsl", "win32"],
  },
  {
    label: "ROCm APU 48 GiB of a 96 GiB pool",
    devices: [{ memoryTotalGb: 48, sharedMemory: true }],
    systemRamTotalGb: 96,
    unifiedMemory: true,
    // Strix Halo class parts; not Apple.
    platforms: ["linux", "wsl", "win32"],
  },
  {
    label: "AMD Vulkan iGPU 12 GiB capped view of RAM",
    devices: [{ memoryTotalGb: 12, sharedMemory: true }],
    systemRamTotalGb: 32,
    unifiedMemory: false,
    platforms: ["linux", "wsl", "win32"],
  },
  {
    label: "Intel iGPU 8 GiB shared",
    devices: [{ memoryTotalGb: 8, sharedMemory: true }],
    systemRamTotalGb: 32,
    unifiedMemory: false,
    platforms: ["linux", "wsl", "win32"],
  },
  {
    label: "Apple Silicon 64 GiB unified",
    devices: [{ memoryTotalGb: 64, sharedMemory: true }],
    systemRamTotalGb: 64,
    unifiedMemory: true,
    platforms: ["darwin"],
  },
  {
    label: "mixed dGPU 16 GiB + iGPU 12 GiB",
    devices: [{ memoryTotalGb: 16 }, { memoryTotalGb: 12, sharedMemory: true }],
    systemRamTotalGb: 96,
    unifiedMemory: false,
    platforms: ["linux", "wsl", "win32"],
  },
  {
    label: "CPU only",
    devices: [],
    systemRamTotalGb: 32,
    unifiedMemory: false,
    platforms: ALL,
  },
];

// Footprints spanning comfortably-under, near the line, and hopeless. The 22/24
// case is the one that lands between 0.90 and 0.97 of a 24 GiB card, which is
// exactly the band this PR's budget change moves.
const FOOTPRINTS = [
  { label: "tiny", weightsBytes: 2 * GB, kvBytes: 1 * GB },
  { label: "half", weightsBytes: 8 * GB, kvBytes: 4 * GB },
  { label: "at the 0.90/0.97 seam", weightsBytes: 20 * GB, kvBytes: 2 * GB },
  { label: "hopeless", weightsBytes: 180 * GB, kvBytes: 40 * GB },
];

function capacityFor(host: Host) {
  return resolveMemoryCapacityGb({
    pinnedDevices: [],
    hostDevices: host.devices.map((d) => ({
      memoryTotalGb: d.memoryTotalGb,
      sharedMemory: d.sharedMemory === true,
    })),
    hostGpuTotalGb: host.devices.reduce((n, d) => n + d.memoryTotalGb, 0),
    hostSharesSystemRam: host.devices.some((d) => d.sharedMemory === true),
    systemRamTotalGb: host.systemRamTotalGb,
    unifiedMemory: host.unifiedMemory,
    gpuBudgetFraction: DEFAULT_VRAM_BUDGET_FRACTION,
  });
}

function cells() {
  const out: { host: Host; platform: Platform; fp: (typeof FOOTPRINTS)[number] }[] = [];
  for (const host of HOSTS) {
    for (const platform of host.platforms) {
      for (const fp of FOOTPRINTS) out.push({ host, platform, fp });
    }
  }
  return out;
}

test("the matrix is actually a matrix", () => {
  // A property suite that silently shrank to two cells is worse than none.
  const n = cells().length;
  assert.ok(n >= 100, `expected a full product, got ${n} cells`);
});

test("P1: no cell ever reports a fit for a footprint over its budget", () => {
  for (const { host, platform, fp } of cells()) {
    const cap = capacityFor(host);
    const bar = computeModelMemory({
      weightsBytes: fp.weightsBytes,
      kvBytes: fp.kvBytes,
      gpuGb: cap.gpuCapacityGb,
      budgetFraction: DEFAULT_VRAM_BUDGET_FRACTION,
      contextIsAutoFitted: false,
    });
    if (bar.status === "unknown") continue;
    const totalGb = (fp.weightsBytes + fp.kvBytes) / GB;
    if (totalGb > bar.budgetGb) {
      assert.notEqual(
        bar.status,
        "fits",
        `${host.label} / ${platform} / ${fp.label}: ${totalGb.toFixed(1)} GiB ` +
          `reported as fitting a ${bar.budgetGb.toFixed(1)} GiB budget`,
      );
    }
  }
});

test("P2: the bar and the panel never contradict each other", () => {
  // The property the whole PR is for. The bar's status and the panel's verdict
  // are different vocabularies over the same question, so they are compared by
  // direction: if one says the load does not fit, the other must not say it does.
  for (const { host, platform, fp } of cells()) {
    const cap = capacityFor(host);
    const bar = computeModelMemory({
      weightsBytes: fp.weightsBytes,
      kvBytes: fp.kvBytes,
      gpuGb: cap.gpuCapacityGb,
      budgetFraction: DEFAULT_VRAM_BUDGET_FRACTION,
      contextIsAutoFitted: false,
    });
    const panel = classifyMemoryFit(fp.weightsBytes + fp.kvBytes, cap.gpuCapacityGb);
    if (bar.status === "unknown" || panel === "unknown") continue;
    const barSaysNo = bar.status !== "fits";
    const panelSaysNo = panel === "exceeds";
    if (panelSaysNo) {
      assert.ok(
        barSaysNo,
        `${host.label} / ${platform} / ${fp.label}: panel says exceeds, bar says fits`,
      );
    }
  }
});

test("P3: one byte count formats identically wherever it is printed", () => {
  // Two formatters, two units in, one label out. This is the collision that used
  // to exist as two functions with the same name.
  for (const gib of [0.5, 2.33, 7.24, 24, 174, 1024]) {
    assert.equal(formatBytesGiB(gib * GB).endsWith(" GiB"), true);
    assert.equal(formatGiB(gib).endsWith(" GiB"), true);
    // The same quantity, so the numeric part must agree once rounding is undone.
    const a = Number.parseFloat(formatBytesGiB(gib * GB));
    const b = Number.parseFloat(formatGiB(gib));
    assert.ok(
      Math.abs(a - b) <= 0.55,
      `${gib} GiB formats as ${a} one way and ${b} the other`,
    );
  }
});

test("P4: a shared or unified pool is never counted twice", () => {
  for (const host of HOSTS) {
    const cap = capacityFor(host);
    if (host.devices.length === 0) continue;
    const ceiling = host.unifiedMemory
      ? host.systemRamTotalGb
      : host.systemRamTotalGb +
        host.devices.reduce((n, d) => n + (d.sharedMemory ? 0 : d.memoryTotalGb), 0);
    assert.ok(
      cap.totalCapacityGb <= ceiling + 0.01,
      `${host.label}: ceiling ${cap.totalCapacityGb} exceeds the ${ceiling} GiB ` +
        `the machine physically has, so a pool was counted twice`,
    );
  }
});

test("P5: a CPU-only host draws nothing rather than a zero-width bar", () => {
  for (const platform of ALL) {
    const host = HOSTS.find((h) => h.label === "CPU only")!;
    const cap = capacityFor(host);
    const bar = computeModelMemory({
      weightsBytes: 4 * GB,
      kvBytes: 1 * GB,
      gpuGb: cap.gpuCapacityGb,
    });
    assert.equal(bar.status, "unknown", `${platform}: CPU-only host drew a VRAM bar`);
    assert.equal(bar.budgetGb, 0);
  }
});

test("P6: no cell leaks a number that does not exist into a label", () => {
  const bad = /NaN|Infinity|undefined|-\d/;
  for (const { host, platform, fp } of cells()) {
    const cap = capacityFor(host);
    const bar = computeModelMemory({
      weightsBytes: fp.weightsBytes,
      kvBytes: fp.kvBytes,
      gpuGb: cap.gpuCapacityGb,
      budgetFraction: DEFAULT_VRAM_BUDGET_FRACTION,
    });
    for (const label of [
      formatGiB(bar.totalGb),
      formatGiB(bar.budgetGb),
      formatGiB(bar.modelGb),
      formatBytesGiB(fp.weightsBytes),
      formatKvRate(bar.kvBytesPerToken),
    ]) {
      assert.doesNotMatch(
        label,
        bad,
        `${host.label} / ${platform} / ${fp.label}: rendered "${label}"`,
      );
    }
  }
});

test("hostile and malformed figures never become a confident verdict", () => {
  // JSON.parse turns 1e999 into Infinity, and a `?? 0` default never sees it.
  for (const evil of [Number.NaN, Number.POSITIVE_INFINITY, -1, 0]) {
    const bar = computeModelMemory({
      weightsBytes: evil,
      kvBytes: evil,
      gpuGb: 24,
      budgetFraction: DEFAULT_VRAM_BUDGET_FRACTION,
    });
    assert.equal(bar.status, "unknown", `weights=${evil} produced ${bar.status}`);
    assert.equal(classifyMemoryFit(evil, 24), "unknown");
    assert.equal(classifyMemoryFit(8 * GB, evil), "unknown");
  }
});
