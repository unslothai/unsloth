// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { pickLoadDevice } from "../src/hooks/gpu-selection.ts";

/** The subset of the /api/system device record the load-device pick reads. */
type GpuDevice = {
  index?: number;
  index_kind?: string;
  visible_ordinal?: number;
  memory_total_gb?: number;
};

// CUDA_VISIBLE_DEVICES="3,1": physical GPU 3 becomes CUDA device 0, so a bare "cuda"
// load lands on the 8 GB card even though the lower PHYSICAL index (1) is the 48 GB one.
const reorderedMask: GpuDevice[] = [
  { index: 1, index_kind: "physical", visible_ordinal: 1, memory_total_gb: 48 },
  { index: 3, index_kind: "physical", visible_ordinal: 0, memory_total_gb: 8 },
];

test("a reordering CUDA_VISIBLE_DEVICES sizes against visible ordinal 0, not the lowest physical index", () => {
  assert.equal(pickLoadDevice(reorderedMask)?.memory_total_gb, 8);
});

test("device order in the payload does not change the pick", () => {
  assert.equal(
    pickLoadDevice([...reorderedMask].reverse())?.memory_total_gb,
    8,
  );
});

test("an ascending mask is unchanged: ordinal 0 is still the lowest physical index", () => {
  const ascending: GpuDevice[] = [
    {
      index: 1,
      index_kind: "physical",
      visible_ordinal: 0,
      memory_total_gb: 48,
    },
    {
      index: 3,
      index_kind: "physical",
      visible_ordinal: 1,
      memory_total_gb: 8,
    },
  ];
  assert.equal(pickLoadDevice(ascending)?.memory_total_gb, 48);
});

test("an older backend without visible_ordinal still falls back to the lowest index", () => {
  const legacy: GpuDevice[] = [
    { index: 3, memory_total_gb: 8 },
    { index: 1, memory_total_gb: 48 },
  ];
  assert.equal(pickLoadDevice(legacy)?.memory_total_gb, 48);
});

test("an empty inventory has no load device", () => {
  assert.equal(pickLoadDevice([]), undefined);
});
