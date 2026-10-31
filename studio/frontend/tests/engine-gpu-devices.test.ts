// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as GpuHooks from "../src/hooks/use-gpu-info.ts";
import { normalizeDenseQuantSchemes } from "../src/lib/dense-quant-schemes.ts";
import { normalizeReportedOffloadFitTiers } from "../src/lib/offload-fit-tiers.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

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

function host(
  deviceBackend: string,
  devices: Record<string, unknown>[],
  inference?: Record<string, unknown>,
) {
  const gpu = { available: true, backend: deviceBackend, devices };
  snapshot = {
    status: "ready",
    device_backend: deviceBackend,
    gpu,
    inference_gpu: inference ?? gpu,
    memory: { total_gb: 128, available_gb: 100 },
  };
}

const engineGpus = () =>
  hooks
    .useGpuDevices(true)
    .filter(hooks.isEngineGpuDevice)
    .map((device) => device.index);

test("an AMD GPU on ROCm is offered to the optional engines, named or not", () => {
  // The Linux ROCm inventory comes from sysfs and may carry no device name.
  host("rocm", [{ index: 0, index_kind: "physical", memory_total_gb: 64 }]);
  assert.deepEqual(engineGpus(), [0]);
  assert.deepEqual(hooks.defaultEngineGpuIds(), [0]);
});

test("the engine GPUs are torch's physical ids, even beside a Vulkan llama.cpp build", () => {
  host(
    "rocm",
    [
      { index: 1, index_kind: "physical", name: "AMD Radeon RX 7900 XTX" },
      { index: 2, index_kind: "physical", name: "AMD Radeon RX 7900 XTX" },
    ],
    {
      available: true,
      backend: "vulkan",
      devices: [{ index: 0, index_kind: "vulkan", name: "Radeon (Vulkan)" }],
    },
  );
  assert.deepEqual(engineGpus(), [1, 2]);
  assert.deepEqual(hooks.defaultEngineGpuIds(), [1]);
});

test("NVIDIA GPUs stay offered, and XPU or relative ids are not", () => {
  host("cuda", [
    {
      index: 0,
      index_kind: "physical",
      name: "NVIDIA RTX 6000 Ada Generation",
    },
    { index: 1, index_kind: "physical", name: "NVIDIA GeForce RTX 3090" },
  ]);
  assert.deepEqual(engineGpus(), [0, 1]);
  host("xpu", [{ index: 0, index_kind: "physical", name: "Intel Arc A770" }]);
  assert.deepEqual(engineGpus(), []);
  assert.deepEqual(hooks.defaultEngineGpuIds(), [0]);
  host("cuda", [{ index: 0, index_kind: "relative", name: "NVIDIA H100" }]);
  assert.deepEqual(engineGpus(), []);
});
