// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { MULTI_GPU_BATCH_NOTE, multiGpuPlacementRow } from "../src/lib/multi-gpu-placement.ts";
import { readSrc } from "./helpers/kit.ts";

test("a one-card load, or an older backend, shows no Multi-GPU row", () => {
  assert.equal(multiGpuPlacementRow(null), null);
  assert.equal(multiGpuPlacementRow(undefined), null);
  assert.equal(multiGpuPlacementRow({ placement: { text_encoder: "cuda:1" } }), null);
});

test("the row names the chosen placement and gives the planner's reason", () => {
  const row = multiGpuPlacementRow({
    decision: {
      chosen: "companion_split",
      summary: "placement: companion split: text_encoder on cuda:1 (denoiser 22.4 GiB fits cuda:0)",
      primary: "cuda:0",
      rejected: [{ kind: "one_card", reason: "everything resident on cuda:0 needs 31 GiB, 23 GiB free" }],
    },
  });
  assert.ok(row);
  assert.equal(row.label, "Text encoder / VAE on a second card");
  assert.match(row.detail, /^Companion split: text_encoder on cuda:1/);
  assert.match(row.detail, /Not chosen: everything resident on cuda:0 needs 31 GiB/);
  assert.ok(row.detail.endsWith(MULTI_GPU_BATCH_NOTE));
});

test("the CFG replica verdict (placement {chosen, reason}) is shown when the summary lacks it", () => {
  const declined = multiGpuPlacementRow({
    decision: { chosen: "offload", summary: "placement: one card with 'group' offload on cuda:0 (no split fits)", rejected: [] },
    cfg_parallel: { armed: false, placement: { chosen: "offload", reason: "a replica needs 30 GiB on cuda:1, 12 GiB free" } },
  });
  assert.ok(declined);
  assert.equal(declined.label, "One card with offload");
  assert.match(declined.detail, /No guidance pass on a second card: a replica needs 30 GiB on cuda:1/);
  const armed = multiGpuPlacementRow({
    decision: { chosen: "cfg_replica", summary: "placement: cfg replica on cuda:1", rejected: [] },
    cfg_parallel: { armed: true, placement: { chosen: "cfg_replica", reason: "placement: cfg replica on cuda:1" } },
  });
  assert.ok(armed);
  assert.equal(armed.label, "Guidance pass on a second card");
  // not repeated when the summary already says it
  assert.equal(armed.detail.match(/cfg replica on cuda:1/gi)?.length, 1);
});

test("a multi-card request that ran on one card names the reason (primary may be unknown)", () => {
  const excluded = multiGpuPlacementRow({
    decision: {
      chosen: "one_card",
      summary:
        "placement: one card: everything on cuda:0 (cuda:1 (Tesla T4, sm 7.5) left out: bf16 compute needs sm 8.0 (this load runs bf16 on cuda:0, sm 8.0))",
      primary: "cuda:0",
      rejected: [],
    },
  });
  assert.ok(excluded);
  assert.equal(excluded.label, "One card");
  assert.match(excluded.detail, /^One card: everything on cuda:0 \(cuda:1 \(Tesla T4, sm 7\.5\) left out: bf16 compute needs sm 8\.0/);
  const kept = multiGpuPlacementRow({
    decision: {
      chosen: "one_card",
      summary: "placement: one card: everything on the selected card (Keep multiple models loaded is Always: the residency arbiter prices and claims one card per image or video load, so a second card would hold memory it never accounted for)",
      primary: null,
    },
  });
  assert.ok(kept);
  assert.match(kept.detail, /Keep multiple models loaded is Always/);
});

test("the batch-split note says only a batch split can differ from one card", () => {
  assert.match(MULTI_GPU_BATCH_NOTE, /identical to one card/);
  assert.match(MULTI_GPU_BATCH_NOTE, /batch split can differ slightly from one card/);
});

for (const page of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  test(`${page}: Loaded build has the Multi-GPU row and the GPU hint the batch note`, () => {
    const source = readSrc(page);
    assert.match(source, /multiGpuPlacementRow\(status\.multi_gpu\)/);
    assert.match(source, /<BuildRow\s+label="Multi-GPU"/);
    assert.match(source, /is not part of this release\. \$\{MULTI_GPU_BATCH_NOTE\}/);
  });
}
