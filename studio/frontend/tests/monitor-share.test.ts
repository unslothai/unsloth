// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  memorySegments,
  unslothPoolGb,
  memoryShare,
  modelBreakdown,
  modelShares,
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
  for (const file of ["components/floating-monitor.tsx", "features/settings/tabs/resources-tab.tsx"]) {
    assert.match(readSrc(file), /memorySegments\(vramShare, breakdown, false\)/);
  }
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
    assert.match(src, /modelShares\(loadedModels, discreteVram, separateInferenceGpu !== null\)/);
    // "No model loaded" only when nothing is loaded, not when a model sits outside this pool.
    assert.match(src, /noModels=\{breakdown !== null && loadedModels\.length === 0\}/);
  }
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
