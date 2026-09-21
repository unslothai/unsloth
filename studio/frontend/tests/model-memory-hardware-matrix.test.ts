// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The memory bar draws a hard "OOM likely" line against whatever budget it is
// handed, so what that number means on each host is the whole correctness
// question. It is not the same quantity everywhere:
//
//   NVIDIA / Intel / discrete AMD  card total, GiB, per device, summed
//   AMD or Intel iGPU on Vulkan    FREE shared system RAM minus a host reserve
//   Apple Silicon                  the entire machine's RAM
//   CPU-only                       zero
//
// Only the first is a VRAM ceiling. These cases pin what the bar does with each
// of the others, across the four platform keys the backend already
// distinguishes.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { computeModelMemory } = await import("../src/lib/model-memory.ts");
const { aggregateGpuMemoryTotalGb } = await import("../src/hooks/gpu-vram.ts");

const GB = 1024 ** 3;
const PLATFORMS = ["linux", "wsl", "win32", "darwin"] as const;

type Device = { memory_total_gb: number; shared_memory?: boolean };

/** One row of the hardware matrix, as /api/system reports it. */
interface Host {
  label: string;
  devices: Device[];
  backend: string;
  /** Whether the aggregate is a dedicated VRAM pool the bar may judge against. */
  dedicated: boolean;
}

const HOSTS: Host[] = [
  {
    label: "NVIDIA single 24 GB",
    devices: [{ memory_total_gb: 24 }],
    backend: "cuda",
    dedicated: true,
  },
  {
    label: "NVIDIA 2x24 GB",
    devices: [{ memory_total_gb: 24 }, { memory_total_gb: 24 }],
    backend: "cuda",
    dedicated: true,
  },
  {
    label: "AMD ROCm discrete 32 GB",
    devices: [{ memory_total_gb: 32 }],
    backend: "rocm",
    dedicated: true,
  },
  {
    label: "Intel XPU 16 GB",
    devices: [{ memory_total_gb: 16 }],
    backend: "xpu",
    dedicated: true,
  },
  {
    label: "AMD Vulkan iGPU (shared)",
    devices: [{ memory_total_gb: 12, shared_memory: true }],
    backend: "vulkan",
    dedicated: false,
  },
  {
    label: "Apple Silicon unified 128 GB",
    devices: [{ memory_total_gb: 128 }],
    backend: "mlx",
    dedicated: false,
  },
  {
    label: "CPU only",
    devices: [],
    backend: "cpu",
    dedicated: false,
  },
];

/** The gate the hook applies before it lets the bar draw. */
function budgetIsDedicatedVram(host: Host): boolean {
  return (
    !host.devices.some((d) => d.shared_memory === true) &&
    host.backend !== "mlx" &&
    host.devices.length > 0
  );
}

for (const platform of PLATFORMS) {
  for (const host of HOSTS) {
    test(`${platform} / ${host.label}: the gate matches what the budget means`, () => {
      assert.equal(
        budgetIsDedicatedVram(host),
        host.dedicated,
        `${host.label} is ${host.dedicated ? "" : "not "}a dedicated VRAM pool`,
      );
    });
  }
}

test("a shared-memory iGPU never draws, however roomy the pool looks", () => {
  const host = HOSTS.find((h) => h.label.includes("Vulkan"));
  assert.ok(host);
  assert.equal(budgetIsDedicatedVram(host), false);
  // The figure itself is generous, which is exactly why drawing against it is
  // dangerous: it is free RAM at probe time and shrinks as the desktop is used.
  assert.equal(aggregateGpuMemoryTotalGb(host.devices), 12);
});

test("Apple's unified pool is the whole machine's RAM, so the bar stands down", () => {
  const host = HOSTS.find((h) => h.label.includes("Apple"));
  assert.ok(host);
  assert.equal(budgetIsDedicatedVram(host), false);
  // Drawn against 128 GB at any sane fraction, a 70 GB model reads "fits" while
  // Metal's working set would refuse it.
  const wouldHaveSaid = computeModelMemory({
    weightsBytes: 70 * GB,
    gpuGb: aggregateGpuMemoryTotalGb(host.devices),
  });
  assert.equal(wouldHaveSaid.status, "fits");
});

test("a CPU-only host draws nothing rather than warning", () => {
  const result = computeModelMemory({ weightsBytes: 8 * GB, gpuGb: 0 });
  assert.equal(result.status, "unknown");
});

test("multi-GPU reports the sum, which only a tensor-split load may use", () => {
  const host = HOSTS.find((h) => h.label === "NVIDIA 2x24 GB");
  assert.ok(host);
  assert.equal(aggregateGpuMemoryTotalGb(host.devices), 48);
  // A 30 GB quant "fits" in 48 GB and does not fit on either card alone, which
  // is why a pin has to suppress the bar rather than rescale it.
  const summed = computeModelMemory({ weightsBytes: 30 * GB, gpuGb: 48 });
  const oneCard = computeModelMemory({ weightsBytes: 30 * GB, gpuGb: 24 });
  assert.equal(summed.status, "fits");
  assert.equal(oneCard.status, "model-exceeds");
});

test("a shared pool is counted once, not summed with the dedicated cards", () => {
  assert.equal(
    aggregateGpuMemoryTotalGb([
      { memory_total_gb: 24 },
      { memory_total_gb: 12, shared_memory: true },
      { memory_total_gb: 12, shared_memory: true },
    ]),
    36,
  );
});

test("the budget follows the loader's fraction, not a hardcoded one", () => {
  // 0.90 vs the loader's 0.97 default on a 24 GB card is 1.68 GiB of headroom
  // the loader would have admitted. llama_cpp.py records that 0.90 was tried
  // and reverted because it dropped 91-94% fits to CPU offload (#5106).
  // 24 GiB card: 21.6 usable at 0.90, 23.28 at 0.97. A 22 GiB model sits in
  // the band between them, which is the band that got a false OOM warning.
  const at90 = computeModelMemory({
    weightsBytes: 22 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
  });
  const at97 = computeModelMemory({
    weightsBytes: 22 * GB,
    gpuGb: 24,
    budgetFraction: 0.97,
  });
  assert.equal(at90.status, "model-exceeds");
  assert.equal(at97.status, "fits");
});

test("an absent or nonsense fraction falls back to the shared headroom ratio", () => {
  const fallback = computeModelMemory({ weightsBytes: 8 * GB, gpuGb: 16 });
  for (const budgetFraction of [null, undefined, 0, -1]) {
    assert.equal(
      computeModelMemory({ weightsBytes: 8 * GB, gpuGb: 16, budgetFraction })
        .budgetGb,
      fallback.budgetGb,
    );
  }
});

test("a user-narrowed budget is respected", () => {
  // The fraction is user-settable, so the bar must move with it in both
  // directions rather than only widening.
  const narrow = computeModelMemory({
    weightsBytes: 12 * GB,
    gpuGb: 24,
    budgetFraction: 0.5,
  });
  assert.equal(narrow.budgetGb, 12);
  assert.equal(narrow.status, "fits");
  assert.equal(
    computeModelMemory({
      weightsBytes: 13 * GB,
      gpuGb: 24,
      budgetFraction: 0.5,
    }).status,
    "model-exceeds",
  );
});

test("segments never sum past the track on any host in the matrix", () => {
  for (const host of HOSTS) {
    const gpuGb = aggregateGpuMemoryTotalGb(host.devices);
    for (const fraction of [0.5, 0.9, 0.97, 1]) {
      for (const weights of [1, 8, 64, 512]) {
        const r = computeModelMemory({
          weightsBytes: weights * GB,
          kvBytes: weights * GB,
          specBytes: weights * GB,
          gpuGb,
          budgetFraction: fraction,
        });
        const sum = r.modelPct + r.kvPct + r.specPct;
        assert.ok(sum <= 100.0001, `${host.label}: segments sum to ${sum}`);
        assert.ok(r.modelPct >= 0 && r.kvPct >= 0 && r.specPct >= 0);
        for (const v of [r.budgetGb, r.totalGb, r.fillPct]) {
          assert.ok(Number.isFinite(v), `${host.label}: ${v} is not finite`);
        }
        if (r.status === "fits") {
          assert.ok(
            r.totalGb <= r.budgetGb,
            `${host.label}: reported fits while over budget`,
          );
        }
      }
    }
  }
});
