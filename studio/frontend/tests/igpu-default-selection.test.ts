// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * 2x R9700 (32 GiB) plus a Ryzen 9950X iGPU whose 36 GiB is shared system RAM: the Train page read
 * "VRAM 100 GiB" and the chat GPU picker ticked the iGPU by default. Device rows are pinned here,
 * never read from the host.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type SystemGpuDevice,
  defaultGpuIds,
  reconcileGpuSelection,
} from "../src/hooks/gpu-selection.ts";
import {
  type MemoryTotalDevice,
  aggregateGpuMemoryTotalGb,
  defaultGpuMemoryTotalGb,
} from "../src/hooks/gpu-vram.ts";
import type * as GpuHooks from "../src/hooks/use-gpu-info.ts";
import { normalizeDenseQuantSchemes } from "../src/lib/dense-quant-schemes.ts";
import { normalizeReportedOffloadFitTiers } from "../src/lib/offload-fit-tiers.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const R9700: MemoryTotalDevice = { memory_total_gb: 31.86 };
const IGPU_SHARED: MemoryTotalDevice = {
  memory_total_gb: 36.23,
  shared_memory: true,
  shared_memory_host_backed_gb: 34.26,
  unified_memory: true,
};
// Same iGPU on a stack where the driver did not mark it integrated, but torch has no gfx1036 kernels.
const IGPU_NO_KERNELS: MemoryTotalDevice = {
  memory_total_gb: 36.23,
  no_torch_kernels: true,
};

function device(
  index: number,
  overrides: Partial<SystemGpuDevice> = {},
): SystemGpuDevice {
  return {
    index,
    indexKind: "physical",
    name: `GPU ${index}`,
    memoryTotalGb: 31.86,
    memoryFreeGb: 31.86,
    sharedMemory: false,
    pinnable: true,
    diffusionPinnable: true,
    ...overrides,
  };
}

test("the default VRAM total leaves a shared-memory iGPU out", () => {
  const devices = [R9700, R9700, IGPU_SHARED];
  assert.equal(aggregateGpuMemoryTotalGb(devices), 99.95);
  assert.equal(defaultGpuMemoryTotalGb(devices), 63.72);
});

test("the default VRAM total leaves a GPU torch cannot run out", () => {
  assert.equal(defaultGpuMemoryTotalGb([R9700, R9700, IGPU_NO_KERNELS]), 63.72);
});
test("no GPU torch can run leaves no default VRAM", () => {
  const noKernelCard: MemoryTotalDevice = {
    memory_total_gb: 16,
    no_torch_kernels: true,
  };
  assert.equal(defaultGpuMemoryTotalGb([IGPU_NO_KERNELS]), 0);
  assert.equal(defaultGpuMemoryTotalGb([noKernelCard, IGPU_NO_KERNELS]), 0);
});

test("a shared-memory GPU beside a card torch cannot run counts alone", () => {
  const noKernelCard: MemoryTotalDevice = {
    memory_total_gb: 16,
    no_torch_kernels: true,
  };
  assert.equal(defaultGpuMemoryTotalGb([IGPU_SHARED, noKernelCard]), 36.23);
});

test("an iGPU alone, or discrete cards alone, count as before", () => {
  assert.equal(defaultGpuMemoryTotalGb([IGPU_SHARED]), 36.23);
  assert.equal(defaultGpuMemoryTotalGb([R9700, R9700]), 63.72);
  assert.equal(defaultGpuMemoryTotalGb([]), 0);
});

test("an untouched picker ticks only the discrete cards", () => {
  assert.deepEqual(
    defaultGpuIds([
      device(0),
      device(1),
      device(2, { sharedMemory: true, unifiedMemory: true }),
    ]),
    [0, 1],
  );
  assert.deepEqual(
    defaultGpuIds([device(0), device(1), device(2, { noTorchKernels: true })]),
    [0, 1],
  );
});

test("no default narrowing when there is nothing to narrow", () => {
  assert.equal(defaultGpuIds([device(0), device(1)]), null);
  assert.equal(
    defaultGpuIds([
      device(0, { sharedMemory: true }),
      device(1, { unifiedMemory: true }),
    ]),
    null,
  );
  assert.equal(defaultGpuIds([]), null);
  assert.equal(defaultGpuIds(null), null);
});

test("an Automatic selection sends the default ids, a manual pick is kept", () => {
  assert.deepEqual(
    reconcileGpuSelection(null, undefined, "physical", [0, 1, 2], [0, 1]),
    { ids: [0, 1], indexKind: "physical" },
  );
  // Ticking the iGPU by hand still reaches /load.
  assert.deepEqual(
    reconcileGpuSelection([0, 2], "physical", "physical", [0, 1, 2], [0, 1]),
    { ids: [0, 2], indexKind: "physical" },
  );
  // Without a default, or before discovery settles, Automatic stays null.
  assert.deepEqual(
    reconcileGpuSelection(null, undefined, "physical", [0, 1], null),
    { ids: null, indexKind: null },
  );
  assert.deepEqual(
    reconcileGpuSelection(null, undefined, undefined, null, [0, 1]),
    { ids: null, indexKind: null },
  );
});

let snapshot: unknown = null;
let fetchResults: unknown[] = [];
const hooks = loadWithStubs<typeof GpuHooks>(
  new URL("../src/hooks/use-gpu-info.ts", import.meta.url),
  {
    react: {
      useState: (initial: unknown) => [initial, () => {}],
      useEffect: () => {},
      useMemo: (read: () => unknown) => read(),
    },
    "./use-system": {
      getCachedSystemInfo: () => snapshot,
      fetchSystemInfo: async () => fetchResults.shift() ?? null,
    },
    "@/lib/dense-quant-schemes": { normalizeDenseQuantSchemes },
    "@/lib/offload-fit-tiers": { normalizeReportedOffloadFitTiers },
  },
  { relativePassthrough: true },
);

test("Automatic gets its default pick only once devices are cached", () => {
  snapshot = null;
  assert.deepEqual(hooks.reconcileCachedGpuSelection(null, null, false, true), {
    ids: null,
    indexKind: null,
  });
  const gpu = {
    available: true,
    backend: "rocm",
    devices: [
      { index: 0, index_kind: "physical", ...R9700 },
      { index: 1, index_kind: "physical", ...R9700 },
      { index: 2, index_kind: "physical", ...IGPU_SHARED },
    ],
  };
  snapshot = {
    status: "ready",
    device_backend: "rocm",
    gpu,
    inference_gpu: gpu,
  };
  assert.deepEqual(hooks.reconcileCachedGpuSelection(null, null, false, true), {
    ids: [0, 1],
    indexKind: "physical",
  });
});

test("a device cache warm that left no devices is retried once", async () => {
  const ready = {
    status: "ready",
    inference_gpu: { backend: "rocm", available: true },
  };
  const vulkanPending = {
    status: "ready",
    inference_gpu: { backend: "vulkan", available: false },
  };
  snapshot = ready;
  fetchResults = [null, ready, ready];
  await hooks.ensureGpuDeviceCache();
  assert.equal(fetchResults.length, 1);
  snapshot = vulkanPending;
  fetchResults = [vulkanPending, vulkanPending, ready];
  await hooks.ensureGpuDeviceCache();
  assert.equal(fetchResults.length, 1);
  snapshot = ready;
  fetchResults = [ready, ready];
  await hooks.ensureGpuDeviceCache();
  assert.equal(fetchResults.length, 1);
});

test("chat load paths warm the device cache even for an Automatic pick", () => {
  // A warm gated on an explicit pick leaves Automatic on a cold cache, so /load omits gpu_ids.
  const gated =
    /if \([^{]*selectedGpuIds[^{]*\)\s*\{\s*await ensureGpuDeviceCache\(/;
  for (const file of [
    "../src/features/chat/api/chat-adapter.ts",
    "../src/features/chat/shared-composer.tsx",
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  ]) {
    const source = readFileSync(new URL(file, import.meta.url), "utf8");
    assert.match(source, /await ensureGpuDeviceCache\(/, file);
    assert.doesNotMatch(source, gated, file);
  }
});

test("a background load with no saved pick still reconciles as Automatic", () => {
  // An unconfigured model's config has no selectedGpuIds; skipping the reconcile sent no gpu_ids.
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /const effectiveGpuIds = reconcilePersistedGpuIds\(\s*config\.selectedGpuIds \?\? null,\s*config\.selectedGpuIndexKind,\s*isDiffusion,\s*candidate\.kind === "gguf",/,
  );
});
test("only a GGUF load gets the default pick; other engines keep their own device", () => {
  const gpu = {
    available: true,
    backend: "rocm",
    devices: [
      { index: 0, index_kind: "physical", ...R9700 },
      { index: 1, index_kind: "physical", ...R9700 },
      { index: 2, index_kind: "physical", ...IGPU_SHARED },
    ],
  };
  snapshot = { status: "ready", device_backend: "rocm", gpu, inference_gpu: gpu };
  assert.deepEqual(hooks.reconcileCachedGpuSelection(null, null, false, true), {
    ids: [0, 1],
    indexKind: "physical",
  });
  // A vLLM/SGLang or transformers load on Automatic would otherwise go tensor-parallel over both cards.
  assert.deepEqual(hooks.reconcileCachedGpuSelection(null), { ids: null, indexKind: null });
  assert.deepEqual(hooks.reconcileCachedGpuSelection(null, null, true, true), {
    ids: null,
    indexKind: null,
  });
  // A manual pick is kept whatever the model kind.
  assert.deepEqual(hooks.reconcileCachedGpuSelection([1], "physical"), {
    ids: [1],
    indexKind: "physical",
  });
});

test("Automatic classifies a GGUF before pinning its default pick", () => {
  // /load rejects gpu_ids for DiffusionGemma on Vulkan, and the probe used to skip Automatic.
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /config\.tensorParallel === true \|\|\s*automaticPinsDefault\)/,
  );
  assert.match(
    source,
    /automaticPinsDefault =\s*config\.selectedGpuIds == null &&\s*reconcilePersistedGpuIds\(null, undefined, false, true\) != null;/,
  );
});
