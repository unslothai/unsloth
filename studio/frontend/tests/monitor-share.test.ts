// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  inferenceGpuDisjoint,
  memorySegments,
  unslothPoolGb,
  memoryShare,
  modelBreakdown,
  modelShares,
  poolSplit,
} from "../src/components/monitor-share.ts";
import { readSrc } from "./helpers/kit.ts";

test("Unsloth's share is split out of used memory and never exceeds it", () => {
  assert.deepEqual(memoryShare(20, 32, 6), { unsloth: 6, other: 14, free: 12 });
  // RSS can overshoot the host's used figure (shared pages): clamp, never a negative "other".
  assert.deepEqual(memoryShare(4, 32, 6), { unsloth: 4, other: 0, free: 28 });
  assert.equal(memoryShare(20, 0, 6), null);
  assert.equal(memoryShare(20, 32, null), null);
});

test("models without a size are left out, largest first, labels preferred", () => {
  const shares = modelShares([
    { id: "a", name: "unsloth/small", residentMib: 512 },
    { id: "b", name: "none" },
    { id: "c", name: "repo/big", label: "Big", residentMib: 4096 },
  ]);
  assert.deepEqual(shares, [
    { id: "c", name: "Big", gb: 4 },
    { id: "a", name: "small", gb: 0.5 },
  ]);
});

test("the app row is what is left of Unsloth's share once the models in that pool are taken out", () => {
  const models = [{ id: "m", name: "m", gb: 4 }];
  assert.deepEqual(modelBreakdown(6, models, true), { models, app: 2 });
  // Discrete GPU: the models are in VRAM, so they do not come out of RAM.
  assert.deepEqual(modelBreakdown(6, models, false), { models, app: 6 });
  // RSS below the models (Metal buffers on Apple): no app row rather than "0.00 GiB".
  assert.deepEqual(modelBreakdown(3, models, true), { models, app: null });
  assert.deepEqual(modelBreakdown(null, models, true), { models, app: null });
});

test("both toggles are off by default and only poll loaded models when one is on", () => {
  const store = readSrc("features/settings/stores/monitor-overlay-store.ts");
  assert.match(store, /showUnslothShare: false/);
  assert.match(store, /showModelBreakdown: false/);
  assert.match(
    readSrc("features/settings/tabs/resources-tab.tsx"),
    /useLoadedModels\(\s*wantsModels,\s*wantsModels,?\s*\)/,
  );
  // Floating monitor: mounted only while a toggle is on, and lazy (a static import breaks the startup chunking).
  const monitor = readSrc("components/floating-monitor.tsx");
  assert.match(monitor, /lazy\(\(\) => import\("\.\/monitor-loaded-models"\)\)/);
  // Not while suppressed beside Run settings: nothing shows it.
  assert.match(monitor, /\{wantsModels && !suppressed && \(\s*<Suspense/);
});

test("Unsloth's share never reads below the models it holds in that pool", () => {
  assert.equal(unslothPoolGb(2.26, 2.57, true), 2.57);
  assert.equal(unslothPoolGb(4, 2.57, true), 4);
  // Discrete GPU: the models are in VRAM, RAM is the process tree alone.
  assert.equal(unslothPoolGb(2.26, 2.57, false), 2.26);
  assert.equal(unslothPoolGb(null, 2.57, true), null);
});

test("the model split replaces Unsloth's segment in the same bar, other apps and free unchanged", () => {
  const share = { unsloth: 6, other: 14, free: 12 };
  assert.deepEqual(
    memorySegments(share, null).map((segment) => [segment.kind, segment.gb]),
    [["unsloth", 6], ["other", 14], ["free", 12]],
  );
  const models = [{ id: "m", name: "Qwen", gb: 4 }];
  assert.deepEqual(
    memorySegments(share, modelBreakdown(6, models, true)).map((segment) => [segment.kind, segment.gb]),
    [["model", 4], ["app", 2], ["other", 14], ["free", 12]],
  );
  // No model loaded: still one "Unsloth" segment, not a lone "Unsloth Studio app".
  assert.deepEqual(
    memorySegments(share, modelBreakdown(6, [], true)).map((segment) => segment.kind),
    ["unsloth", "other", "free"],
  );
});

test("no separate By model list: the split lives in the memory legend", () => {
  for (const file of ["components/floating-monitor.tsx", "features/settings/tabs/resources-tab.tsx"]) {
    assert.doesNotMatch(readSrc(file), /ModelBreakdownList|liveMonitor\.byModel/);
  }
});

test("on VRAM the rest is unattributed: Unsloth's KV cache and training land there, not in other apps", () => {
  const share = { unsloth: 6.46, other: 10.8, free: 46.5 };
  assert.deepEqual(
    memorySegments(share, null, false).map((segment) => segment.kind),
    ["unsloth", "unattributed", "free"],
  );
  assert.match(readSrc("components/monitor-share.ts"), /memorySegments\(vramShare, showBreakdown \? \{ models: vramModels, app: null \} : null, false\)/);
});

test("the VRAM split counts only models known to sit on the reported GPUs", () => {
  // CPU (RSS), Vulkan or a separate inference GPU: not this pool's bytes, and the RAM bar already counts host memory.
  const entries = [
    { id: "gpu", name: "a/On-GPU", residentMib: 4096, residentOnGpu: true },
    { id: "other", name: "a/Elsewhere", residentMib: 3072 },
  ];
  assert.deepEqual(modelShares(entries, true).map((model) => model.id), ["gpu"]);
  assert.deepEqual(modelShares(entries).map((model) => model.id), ["gpu", "other"]);
  for (const file of ["components/floating-monitor.tsx", "features/settings/tabs/resources-tab.tsx"]) {
    const src = readSrc(file);
    assert.match(src, /inferenceGpuDisjoint\(\s*systemInfo\.gpu\?\.backend,\s*separateInferenceGpu\?\.backend,?\s*\)/);
  }
  // "No model loaded" only when nothing is loaded, not when a model sits outside this pool.
  assert.match(readSrc("components/monitor-share.ts"), /noModels: showBreakdown && input\.entries\.length === 0/);
});

test("a model size ahead of the memory sample is scaled into the measured share", () => {
  // Just after a 20 GiB load, the pool still reads 5 GiB used.
  const share = memoryShare(5, 32, 20)!;
  const segments = memorySegments(share, modelBreakdown(null, [{ id: "m", name: "Big", gb: 20 }], true));
  const used = segments.filter((segment) => segment.kind !== "free").reduce((sum, s) => sum + s.gb, 0);
  assert.ok(Math.abs(used - 5) < 1e-9);
  assert.ok(Math.abs(segments.reduce((sum, s) => sum + s.gb, 0) - 32) < 1e-9);
  // In step, nothing is rescaled.
  const steady = memorySegments(memoryShare(20, 32, 12)!, modelBreakdown(12, [{ id: "m", name: "A", gb: 8 }], true));
  assert.deepEqual(steady.map((segment) => [segment.kind, segment.gb]), [["model", 8], ["app", 4], ["other", 8], ["free", 12]]);
});

test("llama.cpp on a separate, other-vendor inference GPU is not billed to the displayed cards", () => {
  const entries = [
    { id: "gguf", name: "a/Chat-GGUF", source: "chat", residentMib: 4096, residentOnGpu: true },
    { id: "img", name: "a/Image", source: "image", residentMib: 2048, residentOnGpu: true },
  ];
  assert.deepEqual(modelShares(entries, true, true).map((model) => model.id), ["img"]);
  assert.deepEqual(modelShares(entries, true, false).map((model) => model.id), ["gguf", "img"]);
  // The RAM split is untouched.
  assert.deepEqual(modelShares(entries, false, true).map((model) => model.id), ["gguf", "img"]);
});

test("a combined inference inventory that includes the displayed backend is not elsewhere", () => {
  // get_mixed_vulkan_inference_gpu_info: llama.cpp built with CUDA and Vulkan reports the CUDA cards too.
  assert.equal(inferenceGpuDisjoint("cuda", "cuda+vulkan"), false);
  assert.equal(inferenceGpuDisjoint("cuda", "vulkan"), true);
  assert.equal(inferenceGpuDisjoint("cuda", "rocm"), true);
  assert.equal(inferenceGpuDisjoint("cuda", null), false);
  assert.equal(inferenceGpuDisjoint(undefined, "vulkan"), false);
});

test("CPU is a core row, so Detailed view offers no CPU load option that would do nothing", async () => {
  const { MONITOR_DETAIL_METRICS, selectedDetailMetrics } = await import("../src/components/monitor-details.ts");
  assert.ok(!(MONITOR_DETAIL_METRICS as readonly string[]).includes("cpuLoad"));
  // A stored selection from before still resolves to the GPU metrics it named.
  assert.deepEqual([...selectedDetailMetrics(["cpuLoad", "gpuLoad"])], ["gpuLoad"]);
  assert.match(readSrc("components/floating-monitor.tsx"), /testId="floating-monitor-cpu"/);
});

test("an inflated image estimate shrinks before a measured chat model does", () => {
  // Colab L4: image status says 13.26 GiB, chat buffers 4.35 GiB, the card has 14.04 GiB of Unsloth's in use.
  const models = modelShares(
    [
      { id: "img", name: "sdxl-turbo", source: "image", residentMib: 13579, residentOnGpu: true },
      { id: "chat", name: "Qwen3-8B", source: "chat", residentMib: 4455, residentOnGpu: true },
    ],
    true,
  );
  const segments = memorySegments(memoryShare(14.04, 22.5, 14.04)!, modelBreakdown(null, models, false), false);
  const gb = Object.fromEntries(segments.map((segment) => [segment.id, segment.gb]));
  assert.ok(Math.abs(gb["model:chat"] - 4455 / 1024) < 1e-9);
  assert.ok(Math.abs(gb["model:img"] - (14.04 - 4455 / 1024)) < 1e-9);
  // Measured rows alone over the share are still scaled to fit.
  const over = memorySegments(memoryShare(2, 8, 2)!, modelBreakdown(null, [{ ...models[1], gb: 4 }], false), false);
  assert.ok(Math.abs(over[0].gb - 2) < 1e-9);
});

test("kept chat models are sized by routing on their checkpoint, not a public name they may share", () => {
  assert.match(
    readSrc("features/loaded-models/loaded-models-api.ts"),
    /readInferenceStatus\(signal, entry\.servingCheckpoint\)/,
  );
});

test("one split serves the floating monitor and Settings, models in the pool that holds them", () => {
  const entries = [{ id: "a", name: "org/A-GGUF", source: "chat", residentMib: 4096, residentOnGpu: true }];
  const base = {
    entries, showShare: true, showBreakdown: true, inferenceGpuElsewhere: false,
    processGb: 6, ramUsedGb: 20, ramTotalGb: 32, vramUsedGb: 10, vramTotalGb: 24,
  };
  const off = poolSplit({ ...base, showShare: false, showBreakdown: false, modelsInVram: true });
  assert.deepEqual(off, { ram: null, vram: null, noModels: false });
  const discrete = poolSplit({ ...base, modelsInVram: true });
  assert.deepEqual(discrete.ram?.map((s) => s.kind), ["unsloth", "other", "free"]);
  assert.deepEqual(discrete.vram?.map((s) => [s.kind, s.gb]), [["model", 4], ["unattributed", 6], ["free", 14]]);
  const unified = poolSplit({ ...base, modelsInVram: false });
  assert.equal(unified.vram, null);
  assert.deepEqual(unified.ram?.map((s) => [s.kind, s.gb]), [["model", 4], ["app", 2], ["other", 14], ["free", 12]]);
  assert.equal(poolSplit({ ...base, entries: [], modelsInVram: false }).noModels, true);
  for (const file of ["components/floating-monitor.tsx", "features/settings/tabs/resources-tab.tsx"]) {
    assert.match(readSrc(file), /poolSplit\(\{/, file);
  }
});

test("an unsized loaded model leaves the VRAM bar unsplit rather than claiming Unsloth holds 0 GiB", () => {
  const split = poolSplit({
    entries: [{ id: "hf", name: "unsloth/Qwen3-0.6B", source: "chat" }],
    showShare: true, showBreakdown: false, modelsInVram: true, inferenceGpuElsewhere: false,
    processGb: 3, ramUsedGb: 20, ramTotalGb: 32, vramUsedGb: 10, vramTotalGb: 24,
  });
  assert.equal(split.vram, null);
  assert.notEqual(split.ram, null);
});

test("a model whose pool is unknown (Vulkan) leaves the VRAM bar unsplit; a proven CPU one does not", () => {
  const base = {
    showShare: true, showBreakdown: false, modelsInVram: true, inferenceGpuElsewhere: false,
    processGb: 3, ramUsedGb: 20, ramTotalGb: 32, vramUsedGb: 10, vramTotalGb: 24,
  };
  const vulkan = [{ id: "v", name: "org/V-GGUF", source: "chat", residentMib: 3072 }];
  assert.equal(poolSplit({ ...base, entries: vulkan }).vram, null);
  const cpu = [{ ...vulkan[0], residentOnGpu: false }];
  assert.notEqual(poolSplit({ ...base, entries: cpu }).vram, null);
});

test("on a GPU host a proven CPU model is broken down in RAM, a GPU one in VRAM", () => {
  const split = poolSplit({
    entries: [
      { id: "g", name: "org/G-GGUF", source: "chat", residentMib: 4096, residentOnGpu: true },
      { id: "c", name: "org/C", source: "image", residentMib: 2048, residentOnGpu: false },
    ],
    showShare: false, showBreakdown: true, modelsInVram: true, inferenceGpuElsewhere: false,
    processGb: 6, ramUsedGb: 20, ramTotalGb: 32, vramUsedGb: 10, vramTotalGb: 24,
  });
  assert.deepEqual(split.ram?.map((s) => [s.kind, s.name ?? null, s.gb]), [
    ["model", "C", 2], ["app", null, 4], ["other", null, 14], ["free", null, 12],
  ]);
  assert.deepEqual(split.vram?.map((s) => [s.kind, s.gb]), [["model", 4], ["unattributed", 6], ["free", 14]]);
});
