// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  curatedSizeBytesFor,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { classifyGgufFit } from "../src/lib/gguf-fit.ts";
import {
  hfModelFitsDevice,
  loadScopedGpu,
  orderRecommendedRows,
  searchRowFitsDevice,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

interface Row {
  id: string;
  isGguf?: boolean;
  totalParams?: number;
  curatedSizeBytes?: number;
  pipelineTag?: string;
}

// Two Video catalog seeds, in catalog order.
const LTX = "unsloth/LTX-2.3-GGUF";
const KLEIN = "unsloth/FLUX.2-klein-9B-GGUF";
const SEEDS: Row[] = [
  { id: LTX, isGguf: true },
  { id: KLEIN, isGguf: true },
];
// What the listing reports for those repos (HF `expand=gguf` totals).
const LTX_PARAMS = 21_005_004_544;
const KLEIN_PARAMS = 9_078_581_248;
const OTHER = "unsloth/Wan2.2-T2V-A14B-GGUF";

// The Video picker's task gate: no usable pipeline tag means dropped.
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
  // The listing's row, not the bare seed.
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
  // "LTX-2.3" has no "<n>B" token, so requireKnown hides it; "klein-9B" reads
  // as 9B -> 3.6 GB, inside the 4.2 GB budget.
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

// Neither is unsloth-owned, and Recommended lists `owner: unsloth` only, so
// their seed is the only row they ever get.
const SDXL = "stabilityai/sdxl-turbo";
const WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";
// A seed row as the picker builds it: catalog size, no listing metadata.
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
  // SDXL Turbo is 8 GB but its id has no "<n>B" token to guess from; Wan 2.2
  // TI2V is 30 GB, and its "5B" reads as 2 GB.
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
  // Where a row arrives it is the one measured: klein-9B seeds as a fit, then
  // its listing row cuts it on real metadata.
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

// unsloth owns this one, so the listing DOES report it, and the seed hands off.
const BNB = "unsloth/Z-Image-Turbo-unsloth-bnb-4bit";

test("a listing row inherits the curated size of the seed it takes over", () => {
  // 8 GB card -> 5.6 GB budget.
  const card = { memoryTotalGb: 8, systemRamAvailableGb: 0, budgetKnown: true };
  const bnbSeed = seed(BNB);
  assert.equal(bnbSeed.curatedSizeBytes, 8 * 1024 ** 3);
  // The Hub row carries params only, and the quant guess assumes a quant still to
  // come: 6B -> 2.4 GB for a repo already 4-bit and 8 GB resident, so without the
  // handoff it would flip to fitting.
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
  // SDXL Turbo is 8 GB: inside the 11.2 GB aggregate budget, past the 5.6 GB
  // one card offers, so only the load-scoped budget hides the OOM.
  const sdxl = seed(SDXL);
  assert.equal(hfModelFitsDevice(sdxl, twoCards), true);
  assert.equal(hfModelFitsDevice(sdxl, loadScopedGpu(twoCards, true)), false);
  // The device COUNT narrows with the capacity. classifyGgufFit charges the loader's per-card VRAM
  // reserve once per card, so leaving the host count of 2 on a one-card budget held back two
  // floors against a single device and under-budgeted every scoped quant.
  assert.equal(loadScopedGpu(twoCards, false).deviceCount, undefined);
  assert.equal(loadScopedGpu(twoCards, true).deviceCount, 1);
  const twoOfThree = { ...twoCards, deviceCount: 3 };
  assert.equal(loadScopedGpu(twoOfThree, false).deviceCount, 3);
  assert.equal(loadScopedGpu(twoOfThree, true).deviceCount, 1);
  // An unscoped load keeps the host count, so chat still charges one floor per card.
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

  // A Linux ROCm APU reports unified_memory without shared_memory, so it used to take the
  // raw-host branch above and undo the very subtraction that keeps its GTT window out of the RAM
  // tier. The folded flag is what the reservation question actually turns on.
  const linuxApu = {
    ...mixedHost,
    loadDeviceMemoryGb: 32,
    loadDeviceSharedMemory: false,
    loadDeviceSharesHostMemory: true,
  };
  assert.equal(loadScopedGpu(linuxApu, true).systemRamAvailableGb, 8);
  // The dedicated card still claims it back: this gate narrowed, it did not close.
  assert.equal(
    loadScopedGpu({ ...linuxApu, loadDeviceSharesHostMemory: false }, true)
      .systemRamAvailableGb,
    40,
  );
});

test("a unified GPU window is not also offered as system RAM", () => {
  // 48 GiB host, a 32 GiB GTT window. gpuMemoryTotalsGb counts that window as DEDICATED when
  // shared_memory is false, so systemRamAvailableGb kept the whole 48 and classifyGgufFit added
  // half of it to the window a second time: a ~43 GiB file needing ~50 scored `partial` against
  // an invented ~55 GiB budget, promising an offload into memory the machine does not have.
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
  // Unsubtracted, the way a Linux APU used to arrive.
  assert.equal(verdict(scoped(48)), "partial");
  // Subtracted once, the way Windows and Apple Silicon always were.
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
  // The curated list has the id alone, the Hub list the row with its 6B params.
  // Both size to the catalog's 8 GB, past the 5.6 GB budget, so an id one drops
  // cannot return through the other.
  assert.equal(searchRowFitsDevice({ id: BNB }, opts), false);
  assert.equal(searchRowFitsDevice({ id: BNB, totalParams: 6e9 }, opts), false);
});

test("the Hub fit gate judges a media GGUF by the planner that places it", () => {
  // The reported case: 52 GiB of video GGUF on a 64 GiB Mac. llama.cpp allows 52 * 1.15 + 1 =
  // 60.8 against a 63.5 GiB budget, so the "Fits on device" filter kept it; the diffusion planner
  // allows 44.8 and cannot offload on a host pool, so it never places.
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
  // A Vulkan llama.cpp build sees cards torch cannot, so inferenceGpu is a different inventory
  // (use-gpu-info.ts keeps the torch view for diffusion for exactly this reason). Picking the
  // budget by FILE FORMAT sent an Images GGUF to the Vulkan card's capacity, which the diffusion
  // loader never gets to use.
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
  // Chat is the case the GGUF backend's own inventory is right for, and it still gets it.
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
  // 8 GB fits the 11.2 GB aggregate a chat load may split across, but not the
  // 5.6 GB of the single card an image pipeline lands on.
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
  // Images / Video place a GGUF through the diffusion backend, whose budget on a 64 GiB unified
  // host is (total - 20% reserve) * 0.85 = 43.5 GiB (diffusion_memory.py). llama.cpp allows 62.1.
  // Judging the parent row by one and its quants by the other let a row read as fitting while
  // everything inside it read as oom.
  const mac = { memoryTotalGb: 64, systemRamAvailableGb: 0, budgetKnown: true };
  const row = {
    id: "unsloth/Some-Image-Model-GGUF",
    isGguf: true,
    // 50 GiB, which the two rules disagree about: 50 > 44.8, but 50 * 1.15 + 1 = 58.5 <= 62.1.
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
  // Every format on a task page, not just GGUF: this rule is the budget all of them had before the
  // classifiers were merged, and restricting it to GGUF left the list gate and the search gate
  // answering differently for one row.
  const safetensors = { ...row, id: "unsloth/Some-Image-Model", isGguf: false };
  assert.equal(hfModelFitsDevice(safetensors, mac, { mediaLoad: true }), false);
});
