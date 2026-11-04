// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as GpuHooks from "../src/hooks/use-gpu-info.ts";
import {
  pinnableGpuContext,
  reconcileGpuSelection,
} from "../src/hooks/gpu-selection.ts";
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

// One A100 split into three GPU instances on an R570+ driver: /api/system lists each CUDA device.
function migHost() {
  const devices = [
    { index: 0, index_kind: "cuda_device", name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 39.25 },
    { index: 1, index_kind: "cuda_device", name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 19.5 },
    { index: 2, index_kind: "cuda_device", name: "NVIDIA A100-SXM4-80GB", memory_total_gb: 19.5 },
  ];
  const gpu = { available: true, backend: "cuda", devices };
  snapshot = {
    status: "ready",
    device_backend: "cuda",
    gpu,
    inference_gpu: gpu,
    memory: { total_gb: 128, available_gb: 100 },
  };
}

test("every MIG instance is pickable and tensor-splittable in the CUDA-device namespace", () => {
  migHost();
  const devices = hooks.useGpuDevices();
  assert.deepEqual(
    devices.map((d) => [d.index, d.indexKind, d.pinnable]),
    [
      [0, "cuda_device", true],
      [1, "cuda_device", true],
      [2, "cuda_device", true],
    ],
  );
  assert.deepEqual(pinnableGpuContext(devices), {
    devices,
    ids: [0, 1, 2],
    indexKind: "cuda_device",
  });
});

test("DiffusionGemma and the optional engines can pin a MIG instance", () => {
  migHost();
  assert.deepEqual(
    hooks
      .useGpuDevices(true)
      .filter(hooks.isEngineGpuDevice)
      .map((d) => d.index),
    [0, 1, 2],
  );
});

test("a pick saved in another namespace is never read as a CUDA-device id", () => {
  for (const saved of ["physical", "vulkan", undefined] as const) {
    assert.deepEqual(reconcileGpuSelection([1], saved, "cuda_device", [0, 1, 2]), {
      ids: null,
      indexKind: null,
    });
  }
  assert.deepEqual(reconcileGpuSelection([1], "cuda_device", "cuda_device", [0, 1, 2]), {
    ids: [1],
    indexKind: "cuda_device",
  });
  // And the reverse: a CUDA-device pick does not survive a move to physical ids.
  assert.deepEqual(reconcileGpuSelection([1], "cuda_device", "physical", [0, 1]), {
    ids: null,
    indexKind: null,
  });
});
