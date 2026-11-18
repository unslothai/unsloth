// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { gpuPool } from "../src/features/settings/components/linked-instance-format.ts";

test("mixed cards are named once each and their VRAM is summed", () => {
  const pool = gpuPool([
    { name: "AMD Radeon RX6500 XT", vram_total_gb: 3.98, vram_used_gb: 0.7 },
    { name: "AMD Radeon RX 5700 XT", vram_total_gb: 7.98, vram_used_gb: 0 },
  ]);
  assert.equal(pool.label, "AMD Radeon RX6500 XT + AMD Radeon RX 5700 XT");
  assert.ok(Math.abs((pool.total ?? 0) - 11.96) < 1e-9);
  assert.equal(pool.used, 0.7);
});

test("identical cards collapse to a count", () => {
  const pool = gpuPool([
    { name: "NVIDIA L4", vram_total_gb: 22.5 },
    { name: "NVIDIA L4", vram_total_gb: 22.5 },
  ]);
  assert.equal(pool.label, "2× NVIDIA L4");
  assert.equal(pool.total, 45);
});

test("an unknown reading leaves the sum unknown instead of undercounting", () => {
  const pool = gpuPool([
    { name: "A", vram_total_gb: 8, vram_used_gb: 1 },
    { name: "B", vram_total_gb: 8, vram_used_gb: null },
  ]);
  assert.equal(pool.used, null);
  assert.equal(pool.total, 16);
});

test("a mix of more than two card kinds is labelled by count", () => {
  const card = (name: string) => ({ name, vram_total_gb: 24, vram_used_gb: 1 });
  const pool = gpuPool([card("A"), card("A"), card("B"), card("C")]);
  assert.equal(pool.label, "4 GPUs");
  assert.equal(pool.total, 96);
});
