// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as GpuHooks from "../src/hooks/use-gpu-info.ts";
import type * as SystemHooks from "../src/hooks/use-system.ts";
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

test("resident refresh bypasses cached and in-flight pre-load memory", async () => {
  const requests: { url: string; resolve: (response: Response) => void }[] = [];
  const system = loadWithStubs<typeof SystemHooks>(
    new URL("../src/hooks/use-system.ts", import.meta.url),
    {
      react: {},
      "@/features/auth": {
        authFetch: (url: string) =>
          new Promise<Response>((resolve) => {
            requests.push({ url, resolve });
          }),
      },
    },
    { relativePassthrough: true },
  );
  const initial = system.fetchSystemInfo();
  const refreshed = system.fetchSystemInfo({ refreshMemory: true });
  assert.equal(requests.length, 1);
  requests[0].resolve(Response.json({ memory: { available_gb: 20 } }));
  const idle = await initial;
  await Promise.resolve();
  assert.equal(requests[1].url, "/api/system?refresh_memory=true");
  requests[1].resolve(
    Response.json({
      memory_refreshed: true,
      memory: { available_gb: 10 },
    }),
  );
  const resident = await refreshed;
  assert.equal(idle?.memory.available_gb, 20);
  assert.equal(resident?.memory.available_gb, 10);
  assert.equal(await system.fetchSystemInfo(), resident);
  const again = system.fetchSystemInfo({ refreshMemory: true });
  assert.equal(requests.length, 3);
  requests[2].resolve(new Response(null, { status: 503 }));
  assert.equal(await again, null);
});
