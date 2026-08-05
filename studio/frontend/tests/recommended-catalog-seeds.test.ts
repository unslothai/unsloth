// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  hfModelFitsDevice,
  orderRecommendedRows,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

interface Row {
  id: string;
  isGguf?: boolean;
  totalParams?: number;
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

// The Video picker's task gate: a row with no usable pipeline tag is dropped.
const VIDEO_TASKS = ["text-to-video", "image-to-video"];
const keepVideo = (r: Row) =>
  r.pipelineTag != null && VIDEO_TASKS.includes(r.pipelineTag);

const ids = (rows: Row[]) => rows.map((r) => r.id);

test("a curated row the listing reports but the filters drop keeps its seed", () => {
  // The response carries LTX-2.3, but its row has no pipeline tag, so the task
  // gate rejects it. The catalog knows the model belongs on this page, so the
  // row that already painted must stay instead of vanishing on the response.
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
  // The listing's row (params, size, capability icons), not the bare seed.
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
  // "LTX-2.3" has no "<n>B" token, so the seed is unsizable and requireKnown
  // hides it; "klein-9B" reads as 9B -> 3.6 GB, inside the 4.2 GB budget.
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
