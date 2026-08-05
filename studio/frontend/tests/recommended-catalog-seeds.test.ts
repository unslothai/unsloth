// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  curatedSizeBytesFor,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import {
  hfModelFitsDevice,
  orderRecommendedRows,
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
  // The listing reports LTX-2.3 with no pipeline tag, so the task gate rejects
  // it; the row that already painted must stay.
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
  const small = { memoryTotalGb: 6, systemRamAvailableGb: 0, budgetKnown: true };
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
  const small = { memoryTotalGb: 6, systemRamAvailableGb: 0, budgetKnown: true };
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
  const card = { memoryTotalGb: 24, systemRamAvailableGb: 0, budgetKnown: true };
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
  const card = { memoryTotalGb: 24, systemRamAvailableGb: 0, budgetKnown: true };
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
