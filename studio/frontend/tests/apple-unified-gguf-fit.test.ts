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

test("the ceiling ignores free memory, which a resident model the loader frees still holds", () => {
  // 20 GiB model resident: 22 GiB reads free, but the loader sees ~42 once it has freed it.
  const gpu = macInfo({ metal_working_set_gb: 51.84, vram_free_gb: 22 });
  assert.ok(
    Math.abs(
      (gpu.unifiedLoadCeilingGb ?? 0) - Math.floor(51.84 * 0.85 * 4) / 4,
    ) < 1e-9,
  );
  const reloaded = classifyGgufFit(bytesRequiring(22.2), {
    gpuGb: gpu.memoryTotalGb,
    systemRamGb: gpu.systemRamAvailableGb,
    gpuCount: gpu.deviceCount,
    unifiedLoadBudgetGb: gpu.unifiedLoadCeilingGb,
  });
  assert.equal(reloaded, "fits");
});

test("an older backend without the working set keeps today's verdicts", () => {
  for (const ws of [undefined, null, 0, -1, Number.NaN]) {
    const gpu = macInfo({ metal_working_set_gb: ws, vram_free_gb: 22 });
    assert.equal(gpu.unifiedLoadBudgetGb, null);
    assert.equal(gpu.unifiedLoadCeilingGb, null);
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
  assert.equal(gpu.unifiedLoadCeilingGb, null);
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

test("keep-loaded mode keeps the live budget: the resident may stay beside the new load", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    src,
    /const loadFreesResident =\s*!keepModelsLoaded &&\s*typeof residentCheckpoint === "string"/,
  );
  assert.equal(src.match(/loadFreesResident\s*\?/g)?.length, 2);
});

test("after unload the budget gains the resident's AGX memory, not every app's", () => {
  // 20 GiB resident in AGX: the loader sees 22 + 20 = 42 GiB once it frees it.
  const resident = macInfo({ metal_working_set_gb: 51.84, vram_free_gb: 22, vram_used_gb: 20 });
  assert.equal(resident.unifiedLoadCeilingGb, Math.floor(42 * 0.85 * 4) / 4);
  // 5 GiB resident, 25 GiB held by other apps: only 10 + 5 GiB returns, so 22.2 GiB stays tight.
  const pressured = macInfo({ metal_working_set_gb: 51.84, vram_free_gb: 10, vram_used_gb: 5 });
  assert.equal(pressured.unifiedLoadCeilingGb, Math.floor(15 * 0.85 * 4) / 4);
  const fit = classifyGgufFit(bytesRequiring(22.2), {
    gpuGb: pressured.memoryTotalGb,
    systemRamGb: pressured.systemRamAvailableGb,
    gpuCount: pressured.deviceCount,
    unifiedLoadBudgetGb: pressured.unifiedLoadCeilingGb,
  });
  assert.equal(fit, "marginal");
});

test("a 0 GiB free reading is the tightest budget, not an unknown one", () => {
  const gpu = macInfo({ metal_working_set_gb: 51.84, vram_free_gb: 0 });
  assert.equal(gpu.unifiedLoadBudgetGb, 0);
});

test("only audio.cpp GGUFs skip the llama.cpp budget; Mac TTS GGUFs load through llama.cpp", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(src, /unifiedLoadBudgetGb: loadsOnAudioCpp\(repoId\)\s*\?\s*null/);
  assert.match(src, /diffusionLoad \|\| loadsOnAudioCpp\(repoId\) \? null : chatLoadBudgetGb/);
  assert.doesNotMatch(src, /unifiedLoadBudgetGb: audioTaskPicker/);
});

function ggufOnlyMac(
  inferenceGpu: Record<string, unknown> | undefined,
  availableGb = 6,
) {
  snapshot = {
    status: "ready",
    device_backend: "cpu",
    gpu: { available: false, backend: "cpu", devices: [] },
    inference_gpu: inferenceGpu,
    memory: { total_gb: 16, available_gb: availableGb },
  };
  return {
    training: hooks.useGpuInfo(),
    inference: hooks.useInferenceGpuInfo(),
  };
}

const METAL_INVENTORY = {
  available: true,
  backend: "metal",
  gguf_gpu_ids_supported: false,
  devices: [
    {
      index: 0,
      index_kind: "relative",
      name: "Apple Silicon (Metal)",
      memory_total_gb: 16,
      shared_memory: true,
      shared_memory_host_backed_gb: 16,
      metal_working_set_gb: 10.67,
    },
  ],
};

test("a GGUF-only 16 GiB Mac with 6 GiB free rates a 2.2 GB quant by the Metal budget, not half of RAM", () => {
  const hubFit = (
    gpu: ReturnType<typeof hooks.useInferenceGpuInfo>,
    bytes = 2_200_000_000,
  ) =>
    classifyGgufFit(bytes, {
      gpuGb: gpu.available ? gpu.memoryTotalGb : undefined,
      systemRamGb:
        gpu.systemRamAvailableGb > 0 ? gpu.systemRamAvailableGb : undefined,
      gpuCount: gpu.deviceCount,
      unifiedLoadBudgetGb: gpu.unifiedLoadBudgetGb,
    });

  const before = ggufOnlyMac(undefined);
  assert.equal(before.inference.available, false);
  assert.equal(hubFit(before.inference), "oom");

  const after = ggufOnlyMac(METAL_INVENTORY);
  assert.equal(after.training.available, false);
  assert.equal(after.inference.available, true);
  assert.equal(after.inference.memorySharedGb, 16);
  // min(10.67 working set, 6 free) * 0.85 = 5.1, floored to 0.25 GiB.
  assert.equal(after.inference.unifiedLoadBudgetGb, 5);
  assert.equal(hubFit(after.inference), "fits");
  // 4.2 GB needs about 5.5 GiB: past the 5 GiB Metal budget, though the 16 GiB pool alone would pass it.
  assert.equal(hubFit(after.inference, 4_200_000_000), "marginal");
});

test("with more free than the Metal working set, the working set caps a GGUF-only Mac's budget", () => {
  // min(10.67, 12) * 0.85 = 9.07, floored to 0.25 GiB; not 12 * 0.85 = 10.2.
  assert.equal(ggufOnlyMac(METAL_INVENTORY, 12).inference.unifiedLoadBudgetGb, 9);
});

test("the unified budget needs an Apple inventory: a Vulkan one on a CPU host stays null", () => {
  const { inference } = ggufOnlyMac({ ...METAL_INVENTORY, backend: "vulkan" });
  assert.equal(inference.unifiedLoadBudgetGb, null);
});

test("a model switch on the Metal inventory credits nothing it cannot attribute to the resident", () => {
  const device = {
    ...METAL_INVENTORY.devices[0],
    metal_working_set_gb: 16,
  };
  snapshot = {
    status: "ready",
    device_backend: "cpu",
    gpu: { available: false, backend: "cpu", devices: [] },
    inference_gpu: { ...METAL_INVENTORY, devices: [device] },
    memory: { total_gb: 16, available_gb: 4 },
  };
  const inference = hooks.useInferenceGpuInfo();
  // 4 * 0.85 = 3.4, floored to 0.25 GiB; not the working set's 16 * 0.85 = 13.6.
  assert.equal(inference.unifiedLoadBudgetGb, 3.25);
  assert.equal(inference.unifiedLoadCeilingGb, 3.25);
  const switchFit = classifyGgufFit(bytesRequiring(5), {
    gpuGb: inference.memoryTotalGb,
    gpuCount: inference.deviceCount,
    unifiedLoadBudgetGb: inference.unifiedLoadCeilingGb,
  });
  assert.equal(switchFit, "marginal");

  const mlx = macInfo({ metal_working_set_gb: 16, vram_free_gb: 4, vram_used_gb: 6 }, 4);
  assert.equal(mlx.unifiedLoadCeilingGb, 8.5);
});
