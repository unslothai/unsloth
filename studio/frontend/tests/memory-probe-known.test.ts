// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as GpuHooks from "../src/hooks/use-gpu-info.ts";
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
  },
  { relativePassthrough: true },
);

test("GPU and RAM probes distinguish zero from missing or invalid readings", () => {
  for (const backend of ["cuda", "vulkan"]) {
    for (const reading of [0, 8, undefined, null, -1, Number.NaN, Infinity]) {
      const gpu = {
        available: true,
        backend,
        devices: [
          {
            index: 0,
            index_kind: backend === "vulkan" ? "vulkan" : "physical",
            memory_total_gb: 24,
            vram_free_gb: reading,
          },
        ],
      };
      snapshot = {
        status: "ready",
        device_backend: "cuda",
        gpu,
        inference_gpu: gpu,
        memory: { total_gb: 32, available_gb: reading },
      };
      const known = reading === 0 || reading === 8;
      assert.equal(hooks.useGpuDevices()[0].memoryFreeKnown, known);
      assert.equal(hooks.useInferenceGpuInfo().systemRamAvailableKnown, known);
    }
  }
});
