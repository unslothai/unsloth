// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The picker's GGUF badge on an Apple host pool: total RAM is not what llama.cpp admits, the
// loader's `min(working set, free) * 0.85` is. Past it a quant warns instead of reading `fits`.

import assert from "node:assert/strict";
import test from "node:test";
import type * as GpuHooks from "../src/hooks/use-gpu-info.ts";
import { classifyGgufFit } from "../src/lib/gguf-fit.ts";
import { normalizeDenseQuantSchemes } from "../src/lib/dense-quant-schemes.ts";
import { normalizeReportedOffloadFitTiers } from "../src/lib/offload-fit-tiers.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const GiB = 1024 ** 3;

let snapshot: unknown = null;
const hooks = loadWithStubs<typeof GpuHooks>(
  new URL("../src/hooks/use-gpu-info.ts", import.meta.url),
  {
    react: {
      useState: (initial: unknown) => [
        typeof initial === "function" ? initial() : initial,
        () => {},
      ],
      useEffect: () => {},
      useMemo: (read: () => unknown) => read(),
    },
    "./use-system": { getCachedSystemInfo: () => snapshot },
    "@/lib/dense-quant-schemes": { normalizeDenseQuantSchemes },
    "@/lib/offload-fit-tiers": { normalizeReportedOffloadFitTiers },
  },
  { relativePassthrough: true },
);

/** Bytes whose `requiredGgufMemoryGb` (size * 1.15 + 1) is exactly `requiredGb`. */
const bytesRequiring = (requiredGb: number) => ((requiredGb - 1) / 1.15) * GiB;

function macInfo(device: Record<string, unknown>, availableGb = 25) {
  const gpu = {
    available: true,
    backend: "mlx",
    devices: [
      {
        index: 0,
        index_kind: "relative",
        name: "Apple M3 Max",
        memory_total_gb: 64,
        shared_memory: true,
        shared_memory_host_backed_gb: 64,
        ...device,
      },
    ],
  };
  snapshot = {
    status: "ready",
    device_backend: "mlx",
    gpu,
    inference_gpu: gpu,
    memory: { total_gb: 64, available_gb: availableGb },
  };
  return hooks.useInferenceGpuInfo();
}

function fitOn(gpu: ReturnType<typeof macInfo>, requiredGb: number) {
  return classifyGgufFit(bytesRequiring(requiredGb), {
    gpuGb: gpu.memoryTotalGb,
    systemRamGb: gpu.systemRamAvailableGb,
    gpuCount: gpu.deviceCount,
    unifiedLoadBudgetGb: gpu.unifiedLoadBudgetGb,
  });
}

test("an Apple host pool publishes min(working set, free) * 0.85, floored to 0.25 GiB, as its load budget", () => {
  const gpu = macInfo({ metal_working_set_gb: 51.84, vram_free_gb: 22 });
  assert.equal(gpu.memoryTotalGb, 64);
  assert.ok(
    Math.abs((gpu.unifiedLoadBudgetGb ?? 0) - Math.floor(22 * 0.85 * 4) / 4) <
      1e-9,
  );
  // Idle Mac: the working set binds.
  const idle = macInfo({ metal_working_set_gb: 51.84, vram_free_gb: 60 });
  assert.ok(
    Math.abs(
      (idle.unifiedLoadBudgetGb ?? 0) - Math.floor(51.84 * 0.85 * 4) / 4,
    ) < 1e-9,
  );
  // No device free figure: the host's available RAM stands in.
  const noFree = macInfo({ metal_working_set_gb: 51.84 }, 30);
  assert.ok(
    Math.abs(
      (noFree.unifiedLoadBudgetGb ?? 0) - Math.floor(30 * 0.85 * 4) / 4,
    ) < 1e-9,
  );
});

test("an older backend without the working set keeps today's verdicts", () => {
  for (const ws of [undefined, null, 0, -1, Number.NaN]) {
    assert.equal(
      macInfo({ metal_working_set_gb: ws, vram_free_gb: 22 })
        .unifiedLoadBudgetGb,
      null,
    );
  }
});

test("64 GiB Mac, 51.84 GiB working set, 22 GiB free", () => {
  const gpu = macInfo({ metal_working_set_gb: 51.84, vram_free_gb: 22 });
  const today = (requiredGb: number) =>
    classifyGgufFit(bytesRequiring(requiredGb), {
      gpuGb: gpu.memoryTotalGb,
      systemRamGb: gpu.systemRamAvailableGb,
      gpuCount: gpu.deviceCount,
    });
  // Inside today's total-RAM budget but past the loader's: a warning, never a refusal.
  assert.equal(today(22.2), "fits");
  assert.equal(fitOn(gpu, 22.2), "marginal");
  assert.equal(fitOn(gpu, 15), "fits");
  // Past today's budget nothing moves.
  assert.equal(fitOn(gpu, 63.5), today(63.5));
  assert.equal(fitOn(gpu, 63.5), "marginal");
  assert.equal(fitOn(gpu, 200), today(200));
});

test("discrete NVIDIA 24 GiB is untouched", () => {
  snapshot = {
    status: "ready",
    device_backend: "cuda",
    gpu: {
      available: true,
      backend: "cuda",
      devices: [
        {
          index: 0,
          index_kind: "physical",
          memory_total_gb: 24,
          vram_free_gb: 4,
          // Never honoured off Apple, even if a future backend sent it.
          metal_working_set_gb: 10,
        },
      ],
    },
    memory: { total_gb: 64, available_gb: 48 },
  };
  const gpu = hooks.useInferenceGpuInfo();
  assert.equal(gpu.unifiedLoadBudgetGb, null);
  for (const requiredGb of [10, 22.2, 23.5, 40, 200]) {
    assert.equal(
      fitOn(gpu, requiredGb),
      classifyGgufFit(bytesRequiring(requiredGb), {
        gpuGb: 24,
        systemRamGb: gpu.systemRamAvailableGb,
        gpuCount: 1,
      }),
    );
  }
  assert.equal(fitOn(gpu, 22.2), "fits");
});

test("a loader budget floored to zero still warns rather than reading as unknown", () => {
  const fit = classifyGgufFit(bytesRequiring(10), {
    gpuGb: 64,
    systemRamGb: 0,
    unifiedLoadBudgetGb: 0,
  });
  assert.equal(fit, "marginal");
});
