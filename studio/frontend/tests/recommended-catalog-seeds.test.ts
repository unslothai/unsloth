// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  artifactForRepoId,
  curatedSizeBytesFor,
  groupForRepoId,
  loadSpecFor,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { classifyGgufFit } from "../src/lib/gguf-fit.ts";
import {
  hfModelFitsDevice,
  loadScopedGpu,
  orderRecommendedRows,
  recommendedEmptyState,
  searchRowFitsDevice,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

interface Row {
  id: string;
  isGguf?: boolean;
  totalParams?: number;
  curatedSizeBytes?: number;
  pipelineTag?: string;
}

const LTX = "unsloth/LTX-2.3-GGUF";
const KLEIN = "unsloth/FLUX.2-klein-9B-GGUF";
const SEEDS: Row[] = [
  { id: LTX, isGguf: true },
  { id: KLEIN, isGguf: true },
];
const LTX_PARAMS = 21_005_004_544;
const KLEIN_PARAMS = 9_078_581_248;
const OTHER = "unsloth/Wan2.2-T2V-A14B-GGUF";

const VIDEO_TASKS = ["text-to-video", "image-to-video"];
const keepVideo = (r: Row) =>
  r.pipelineTag != null && VIDEO_TASKS.includes(r.pipelineTag);

const ids = (rows: Row[]) => rows.map((r) => r.id);

test("a curated row the listing reports but the filters drop keeps its seed", () => {
  // The task gate rejects LTX-2.3 (no pipeline tag), so its painted row stays.
  const results: Row[] = [
    { id: LTX, isGguf: true, totalParams: LTX_PARAMS },
    { id: OTHER, isGguf: true, pipelineTag: "text-to-video" },
  ];
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds: SEEDS,
        results,
        keep: keepVideo,
        deviceFiltered: false,
        fits: () => true,
      }),
    ),
    [LTX, KLEIN, OTHER],
  );
});

test("a listing row that passes the filters takes over its seed, in catalog order", () => {
  const listedLtx: Row = {
    id: LTX,
    isGguf: true,
    totalParams: LTX_PARAMS,
    pipelineTag: "image-to-video",
  };
  const extra: Row = { id: OTHER, isGguf: true, pipelineTag: "text-to-video" };
  const out = orderRecommendedRows({
    seeds: SEEDS,
    results: [extra, listedLtx],
    keep: keepVideo,
    deviceFiltered: false,
    fits: () => true,
  });
  assert.deepEqual(ids(out), [LTX, KLEIN, OTHER]);
  assert.equal(out[0], listedLtx);
});

test("device fit is judged on whichever row renders", () => {
  const small = {
    memoryTotalGb: 6,
    systemRamAvailableGb: 0,
    budgetKnown: true,
  };
  const big = { memoryTotalGb: 80, systemRamAvailableGb: 0, budgetKnown: true };
  const listedLtx: Row = {
    id: LTX,
    isGguf: true,
    totalParams: LTX_PARAMS,
    pipelineTag: "image-to-video",
  };
  const listedKlein: Row = {
    id: KLEIN,
    isGguf: true,
    totalParams: KLEIN_PARAMS,
    pipelineTag: "text-to-video",
  };
  const results = [listedLtx, listedKlein];
  // 21B -> 8.4 GB smallest quant, past a 6 GB card's 4.2 GB budget; 9B -> 3.6 GB.
  assert.equal(hfModelFitsDevice(listedLtx, small), false);
  assert.equal(hfModelFitsDevice(listedKlein, small), true);
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds: SEEDS,
        results,
        keep: keepVideo,
        deviceFiltered: true,
        fits: (r: Row) => hfModelFitsDevice(r, small),
      }),
    ),
    [KLEIN],
  );
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds: SEEDS,
        results,
        keep: keepVideo,
        deviceFiltered: true,
        fits: (r: Row) => hfModelFitsDevice(r, big),
      }),
    ),
    [LTX, KLEIN],
  );
});

test("an unlisted seed is sized from its id, and hidden when it cannot be", () => {
  const small = {
    memoryTotalGb: 6,
    systemRamAvailableGb: 0,
    budgetKnown: true,
  };
  // "LTX-2.3" has no "<n>B" token, so requireKnown hides it; klein-9B fits at 3.6 GB.
  assert.equal(hfModelFitsDevice(SEEDS[0], small), false);
  assert.equal(hfModelFitsDevice(SEEDS[1], small), true);
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds: SEEDS,
        results: [],
        keep: keepVideo,
        deviceFiltered: true,
        fits: (r: Row) => hfModelFitsDevice(r, small),
      }),
    ),
    [KLEIN],
  );
});

// Recommended lists owner unsloth only, so the seed is the only row these get.
const SDXL = "stabilityai/sdxl-turbo";
const WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";
const seed = (id: string, catalog = IMAGE_CATALOG): Row => ({
  id,
  isGguf: false,
  curatedSizeBytes: curatedSizeBytesFor(id, catalog),
});

test("a catalog-sized seed is judged on the catalog size, not on its id", () => {
  // 24 GB card -> 16.8 GB budget.
  const card = {
    memoryTotalGb: 24,
    systemRamAvailableGb: 0,
    budgetKnown: true,
  };
  const sdxl = seed(SDXL);
  const wan = seed(WAN, VIDEO_CATALOG);
  assert.equal(sdxl.curatedSizeBytes, 8 * 1024 ** 3);
  assert.equal(wan.curatedSizeBytes, 30 * 1024 ** 3);
  assert.equal(hfModelFitsDevice(sdxl, card), true);
  assert.equal(hfModelFitsDevice(wan, card), false);
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds: [sdxl, wan],
        results: [],
        keep: () => true,
        deviceFiltered: true,
        fits: (r: Row) => hfModelFitsDevice(r, card),
      }),
    ),
    [SDXL],
  );
});

test("a listing row still overrides the catalog size it seeded with", () => {
  const card = {
    memoryTotalGb: 24,
    systemRamAvailableGb: 0,
    budgetKnown: true,
  };
  // GGUF groups carry no catalog size, so an unsized GGUF seed stays hidden.
  assert.equal(curatedSizeBytesFor(LTX, VIDEO_CATALOG), undefined);
  assert.equal(hfModelFitsDevice({ id: LTX, isGguf: true }, card), false);
  const listedKlein: Row = {
    id: KLEIN,
    isGguf: true,
    totalParams: 200e9,
    pipelineTag: "text-to-video",
  };
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds: SEEDS,
        results: [listedKlein],
        keep: keepVideo,
        deviceFiltered: true,
        fits: (r: Row) => hfModelFitsDevice(r, card),
      }),
    ),
    [],
  );
});

const BNB = "unsloth/Z-Image-Turbo-unsloth-bnb-4bit";

test("a listing row inherits the curated size of the seed it takes over", () => {
  // 8 GB card -> 5.6 GB budget.
  const card = { memoryTotalGb: 8, systemRamAvailableGb: 0, budgetKnown: true };
  const bnbSeed = seed(BNB);
  assert.equal(bnbSeed.curatedSizeBytes, 8 * 1024 ** 3);
  // The quant guess assumes a future quant, so an already 4-bit repo would wrongly fit.
  const listed: Row = { id: BNB, isGguf: false, totalParams: 6e9 };
  assert.equal(hfModelFitsDevice(listed, card), true);
  assert.equal(hfModelFitsDevice(bnbSeed, card), false);
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds: [bnbSeed],
        results: [listed],
        keep: () => true,
        deviceFiltered: true,
        fits: (r: Row) => hfModelFitsDevice(r, card),
      }),
    ),
    [],
  );
});

test("a task row is sized against the device the load lands on", () => {
  const twoCards = {
    available: true,
    budgetKnown: true,
    memoryTotalGb: 16,
    maxDeviceMemoryGb: 8,
    loadDeviceMemoryGb: 8,
    systemRamAvailableGb: 0,
  };
  // Chat may split across both cards; an image/video pipeline lands on one.
  assert.equal(loadScopedGpu(twoCards, false).memoryTotalGb, 16);
  assert.equal(loadScopedGpu(twoCards, true).memoryTotalGb, 8);
  const sdxl = seed(SDXL);
  assert.equal(hfModelFitsDevice(sdxl, twoCards), true);
  assert.equal(hfModelFitsDevice(sdxl, loadScopedGpu(twoCards, true)), false);
  // classifyGgufFit charges a per-card VRAM reserve, so the count must narrow with capacity.
  assert.equal(loadScopedGpu(twoCards, false).deviceCount, undefined);
  assert.equal(loadScopedGpu(twoCards, true).deviceCount, 1);
  const twoOfThree = { ...twoCards, deviceCount: 3 };
  assert.equal(loadScopedGpu(twoOfThree, false).deviceCount, 3);
  assert.equal(loadScopedGpu(twoOfThree, true).deviceCount, 1);
  assert.equal(loadScopedGpu(twoOfThree, true).memoryTotalGb, 8);
});

test("a dedicated task device keeps RAM reserved by a shared GPU", () => {
  const mixedHost = {
    available: true,
    budgetKnown: true,
    memoryTotalGb: 48,
    maxDeviceMemoryGb: 32,
    loadDeviceMemoryGb: 16,
    loadDeviceSharedMemory: false,
    systemRamAvailableGb: 8,
    systemRamAvailableHostGb: 40,
  };
  const sharedLoadDevice = {
    ...mixedHost,
    loadDeviceMemoryGb: 32,
    loadDeviceSharedMemory: true,
  };

  assert.equal(loadScopedGpu(mixedHost, true).systemRamAvailableGb, 40);
  assert.equal(loadScopedGpu(sharedLoadDevice, true).systemRamAvailableGb, 8);
  assert.equal(loadScopedGpu(mixedHost, false), mixedHost);

  // A Linux ROCm APU reports unified_memory without shared_memory; the folded flag decides.
  const linuxApu = {
    ...mixedHost,
    loadDeviceMemoryGb: 32,
    loadDeviceSharedMemory: false,
    loadDeviceSharesHostMemory: true,
  };
  assert.equal(loadScopedGpu(linuxApu, true).systemRamAvailableGb, 8);
  assert.equal(
    loadScopedGpu({ ...linuxApu, loadDeviceSharesHostMemory: false }, true)
      .systemRamAvailableGb,
    40,
  );
});

test("a unified GPU window is not also offered as system RAM", () => {
  // A GTT window counted as dedicated was added to the RAM budget twice, inventing offload room.
  const apu = {
    available: true,
    budgetKnown: true,
    memoryTotalGb: 32,
    maxDeviceMemoryGb: 32,
    loadDeviceMemoryGb: 32,
    loadDeviceSharedMemory: false,
    loadDeviceSharesHostMemory: true,
    systemRamAvailableHostGb: 48,
    deviceCount: 1,
  };
  const scoped = (systemRamAvailableGb: number) =>
    loadScopedGpu({ ...apu, systemRamAvailableGb }, true);
  const verdict = (g: ReturnType<typeof scoped>) =>
    classifyGgufFit(43 * 1024 ** 3, {
      gpuGb: g.memoryTotalGb,
      systemRamGb: g.systemRamAvailableGb,
      gpuCount: g.deviceCount,
    });
  assert.equal(verdict(scoped(48)), "partial");
  assert.equal(verdict(scoped(48 - 32)), "oom");
});

test("both search lists judge a curated id the same way", () => {
  const oneCard = {
    available: true,
    budgetKnown: true,
    memoryTotalGb: 8,
    maxDeviceMemoryGb: 8,
    loadDeviceMemoryGb: 8,
    systemRamAvailableGb: 0,
  };
  const opts = {
    isGguf: false,
    curatedSizeBytes: curatedSizeBytesFor(BNB, IMAGE_CATALOG),
    gpu: oneCard,
    inferenceGpu: oneCard,
    taskScoped: true,
  };
  // Both lists size to the catalog's 8 GB, so an id one drops cannot return through the other.
  assert.equal(searchRowFitsDevice({ id: BNB }, opts), false);
  assert.equal(searchRowFitsDevice({ id: BNB, totalParams: 6e9 }, opts), false);
});

test("the Hub fit gate judges a media GGUF by the planner that places it", () => {
  // The diffusion planner's budget is below llama.cpp's and cannot offload on a host pool.
  const mac = {
    available: true,
    budgetKnown: true,
    memoryTotalGb: 64,
    maxDeviceMemoryGb: 64,
    loadDeviceMemoryGb: 64,
    loadDeviceSharedMemory: true,
    loadDeviceSharesHostMemory: true,
    systemRamAvailableGb: 0,
    systemRamAvailableHostGb: 64,
    deviceCount: 1,
  };
  const row = {
    id: "unsloth/Some-Video-GGUF",
    isGguf: true,
    estimatedSizeBytes: 52 * 1024 ** 3,
  };
  assert.equal(hfModelFitsDevice(row, mac, { gpuCount: 1 }), true);
  const scoped = loadScopedGpu(mac, true);
  assert.equal(
    hfModelFitsDevice(row, scoped, {
      gpuCount: scoped.deviceCount,
      mediaLoad: true,
      hostPooledMemory: true,
    }),
    false,
  );
});

test("a media GGUF is sized against torch, not the GGUF backend", () => {
  // A Vulkan llama.cpp build sees cards torch cannot, so media rows use the torch inventory.
  const vulkanCard = {
    available: true,
    budgetKnown: true,
    memoryTotalGb: 24,
    maxDeviceMemoryGb: 24,
    loadDeviceMemoryGb: 24,
    systemRamAvailableGb: 0,
  };
  const torchCard = { ...vulkanCard, memoryTotalGb: 12, maxDeviceMemoryGb: 12, loadDeviceMemoryGb: 12 };
  // 14 GiB: inside the 24 GiB card's media budget (16.8), past the 12 GiB one's (8.4).
  const row = { id: "unsloth/Some-Image-GGUF", estimatedSizeBytes: 14 * 1024 ** 3 };
  const opts = {
    isGguf: true,
    gpu: torchCard,
    inferenceGpu: vulkanCard,
    taskScoped: true,
    diffusionLoad: true,
  };
  assert.equal(searchRowFitsDevice(row, opts), false);
  assert.equal(
    searchRowFitsDevice(row, {
      ...opts,
      taskScoped: false,
      diffusionLoad: false,
    }),
    true,
  );
});

test("a search row is sized against the device a task load lands on", () => {
  const twoCards = {
    available: true,
    budgetKnown: true,
    memoryTotalGb: 16,
    maxDeviceMemoryGb: 8,
    loadDeviceMemoryGb: 8,
    systemRamAvailableGb: 0,
  };
  const opts = {
    isGguf: false,
    curatedSizeBytes: curatedSizeBytesFor(SDXL, IMAGE_CATALOG),
    gpu: twoCards,
    inferenceGpu: twoCards,
  };
  assert.equal(
    searchRowFitsDevice({ id: SDXL }, { ...opts, taskScoped: false }),
    true,
  );
  assert.equal(
    searchRowFitsDevice({ id: SDXL }, { ...opts, taskScoped: true }),
    false,
  );
});

test("a GPU-less host keeps its unified-memory budget", () => {
  const mac = {
    available: false,
    budgetKnown: true,
    memoryTotalGb: 0,
    maxDeviceMemoryGb: 0,
    loadDeviceMemoryGb: 0,
    systemRamAvailableGb: 64,
  };
  assert.equal(loadScopedGpu(mac, true), mac);
});

test("a media row is judged by the rule its quant rows use", () => {
  // Diffusion budget on unified 64 GiB is 43.5 GiB (diffusion_memory.py); llama.cpp allows 62.1.
  const mac = { memoryTotalGb: 64, systemRamAvailableGb: 0, budgetKnown: true };
  const row = {
    id: "unsloth/Some-Image-Model-GGUF",
    isGguf: true,
    // 50 > 44.8 for the media rule, but 50 * 1.15 + 1 = 58.5 <= 62.1 for llama.cpp.
    curatedSizeBytes: 50 * 1024 ** 3,
  };
  assert.equal(
    hfModelFitsDevice(row, mac),
    true,
    "chat keeps the llama.cpp rule",
  );
  assert.equal(
    hfModelFitsDevice(row, mac, { mediaLoad: true }),
    false,
    "a media row does not",
  );
  // Applies to every format so the list gate and search gate agree on one row.
  const safetensors = { ...row, id: "unsloth/Some-Image-Model", isGguf: false };
  assert.equal(hfModelFitsDevice(safetensors, mac, { mediaLoad: true }), false);
});

test("with familyOf, curated families follow the listing's sort, artifacts kept together", () => {
  const family = (id: string) =>
    id.toLowerCase().includes("klein") ? "klein" : id.toLowerCase().includes("ltx") ? "ltx" : undefined;
  const kleinBf16: Row = { id: "unsloth/FLUX.2-klein-9B", pipelineTag: "text-to-video" };
  const seeds: Row[] = [...SEEDS, kleinBf16];
  const results: Row[] = [
    { id: KLEIN, isGguf: true, pipelineTag: "text-to-video" },
    { id: OTHER, isGguf: true, pipelineTag: "text-to-video" },
    { id: LTX, isGguf: true, pipelineTag: "image-to-video" },
  ];
  const order = (familyOf?: (id: string) => string | undefined) =>
    ids(
      orderRecommendedRows({
        seeds,
        results,
        keep: keepVideo,
        deviceFiltered: false,
        fits: () => true,
        familyOf,
      }),
    );
  assert.deepEqual(order(), [LTX, KLEIN, kleinBf16.id, OTHER]);
  assert.deepEqual(order(family), [KLEIN, kleinBf16.id, OTHER, LTX]);
  const unlisted = ids(
    orderRecommendedRows({
      seeds,
      results: results.slice(0, 2),
      keep: keepVideo,
      deviceFiltered: false,
      fits: () => true,
      familyOf: family,
    }),
  );
  assert.deepEqual(unlisted, [KLEIN, kleinBf16.id, OTHER, LTX]);
});

test("with familyOf, unsloth rows lead even when a vendor family trends higher", () => {
  const family = (id: string) => (id.toLowerCase().includes("hot") ? "hot" : "cold");
  const vendorHot: Row = { id: "Vendor/Hot-Model", pipelineTag: "text-to-video" };
  const seeds: Row[] = [vendorHot, { id: "unsloth/Hot-Model-GGUF", isGguf: true }];
  const results: Row[] = [
    { id: "unsloth/Hot-Model-GGUF", isGguf: true, pipelineTag: "text-to-video" },
    { id: "unsloth/Cold-Model-GGUF", isGguf: true, pipelineTag: "text-to-video" },
  ];
  assert.deepEqual(
    ids(
      orderRecommendedRows({
        seeds,
        results,
        keep: keepVideo,
        deviceFiltered: false,
        fits: () => true,
        familyOf: family,
      }),
    ),
    ["unsloth/Hot-Model-GGUF", "unsloth/Cold-Model-GGUF", "Vendor/Hot-Model"],
  );
});

test("a vendor id resolves to the unsloth mirror that replaced it", () => {
  for (const [vendor, mirror] of [
    ["Qwen/Qwen-Image-2512", "unsloth/Qwen-Image-2512"],
    ["black-forest-labs/FLUX.1-dev", "unsloth/FLUX.1-dev"],
    ["Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo"],
    ["Qwen/Qwen-Image-2.1", "unsloth/Qwen-Image-2.1"],
  ]) {
    const hit = artifactForRepoId(vendor, IMAGE_CATALOG);
    assert.equal(hit?.artifact.repoId, mirror, vendor);
    assert.equal(hit?.artifact.gated, undefined, vendor);
    assert.equal(loadSpecFor(vendor, IMAGE_CATALOG)?.kind, "pipeline", vendor);
    assert.equal(groupForRepoId(vendor, IMAGE_CATALOG), groupForRepoId(mirror, IMAGE_CATALOG));
  }
});

test("with familyOf, an unslothai family the unsloth listing cannot rank keeps its curated slot", () => {
  const family = (id: string) => id.toLowerCase().replace(/-gguf$/, "");
  const ASR = "unslothai/Qwen3-ASR-0.6B-GGUF";
  const TURBO = "unsloth/whisper-large-v3-turbo";
  const TINY = "unsloth/whisper-tiny";
  const seeds: Row[] = [{ id: ASR, isGguf: true }, { id: TURBO }, { id: TINY }];
  const order = (results: Row[]) =>
    ids(
      orderRecommendedRows({
        seeds,
        results,
        keep: () => true,
        deviceFiltered: false,
        fits: () => true,
        familyOf: family,
      }),
    );
  assert.deepEqual(order([]), [ASR, TURBO, TINY]);
  assert.deepEqual(order([{ id: TINY }, { id: TURBO }]), [ASR, TINY, TURBO]);
});

test("a pinToTop family leads Recommended whatever the listing sort, both artifacts kept together", () => {
  const family = (id: string) => groupForRepoId(id, IMAGE_CATALOG)?.canonicalId.toLowerCase();
  const pinnedFamilies = IMAGE_CATALOG.filter((g) => g.pinToTop).map((g) =>
    g.canonicalId.toLowerCase(),
  );
  assert.deepEqual(pinnedFamilies, ["unsloth/qwen-image-2.1"]);
  const qwen21 = "unsloth/Qwen-Image-2.1";
  const qwen21Gguf = "unsloth/Qwen-Image-2.1-GGUF";
  const zImageTurbo = "unsloth/Z-Image-Turbo-GGUF";
  const qwen2512 = "unsloth/Qwen-Image-2512-GGUF";
  const seeds: Row[] = [
    { id: zImageTurbo, isGguf: true },
    { id: qwen21 },
    { id: qwen21Gguf, isGguf: true },
    { id: qwen2512, isGguf: true },
  ];
  const results: Row[] = [
    { id: qwen2512, isGguf: true },
    { id: zImageTurbo, isGguf: true },
    { id: qwen21Gguf, isGguf: true },
  ];
  const order = (pinned?: readonly string[]) =>
    ids(
      orderRecommendedRows({
        seeds,
        results,
        keep: () => true,
        deviceFiltered: false,
        fits: () => true,
        familyOf: family,
        pinnedFamilies: pinned,
      }),
    );
  assert.deepEqual(order(), [qwen2512, zImageTurbo, qwen21, qwen21Gguf]);
  assert.deepEqual(order(pinnedFamilies), [qwen21, qwen21Gguf, qwen2512, zImageTurbo]);
});

test("an empty Recommended list names a failed search or unreachable hub, not an empty catalog", () => {
  const state = (isLoading: boolean, error: string | null, hubPhase: "available" | "probing" | "unavailable") =>
    recommendedEmptyState({ isLoading, error, hubPhase });
  assert.deepEqual(
    [state(false, null, "available"), state(false, "Failed to fetch", "available"), state(false, null, "unavailable"), state(false, null, "probing"), state(true, "Failed to fetch", "unavailable")],
    ["empty", "failed", "failed", "failed", "loading"],
  );
});
