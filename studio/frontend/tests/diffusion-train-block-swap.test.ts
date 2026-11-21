// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerStoreStubResolver } from "./helpers/kit.ts";

// The helper reuses core's offloadHardwareSupported from the training mappers, whose imports need the resolver.
registerStoreStubResolver();
const { diffusionBlockSwapApplies, diffusionBlockSwapPayload, diffusionBlockSwapSummary } = await import(
  "../src/features/images/train/diffusion-block-swap.ts"
);

const cuda = {
  status: "ready",
  device_backend: "cuda",
  gpu: { devices: [{ unified_memory: false }] },
} as never;
const apu = {
  status: "ready",
  device_backend: "cuda",
  gpu: { devices: [{ unified_memory: true }] },
} as never;
const mac = { status: "ready", device_backend: "mlx", gpu: { devices: [] } } as never;

const on = { supportsBlockSwap: true, basePrecision: "bf16", gradientCheckpointing: true, system: cuda };

test("block swap is offered only for a flagged family on bf16 with checkpointing on a discrete card", () => {
  assert.equal(diffusionBlockSwapApplies(on), true);
  assert.equal(diffusionBlockSwapApplies({ ...on, supportsBlockSwap: undefined }), false);
  assert.equal(diffusionBlockSwapApplies({ ...on, supportsBlockSwap: false }), false);
  for (const p of ["auto", "nf4", "int8", "fp8", "mxfp8"]) {
    assert.equal(diffusionBlockSwapApplies({ ...on, basePrecision: p }), false, p);
  }
  assert.equal(diffusionBlockSwapApplies({ ...on, gradientCheckpointing: false }), false);
  assert.equal(diffusionBlockSwapApplies({ ...on, system: apu }), false);
  assert.equal(diffusionBlockSwapApplies({ ...on, system: mac }), false);
  // Before /api/system answers, as core's offload rule: shown, the backend still refuses.
  assert.equal(diffusionBlockSwapApplies({ ...on, system: null }), true);
});

test("the payload is omitted when hidden or off, so the default request is unchanged", () => {
  assert.equal(diffusionBlockSwapPayload(false, "auto", 4), undefined);
  assert.equal(diffusionBlockSwapPayload(true, "off", 4), undefined);
  assert.equal(diffusionBlockSwapPayload(true, "auto", 4), "auto");
  assert.equal(diffusionBlockSwapPayload(true, "count", 14.7), 14);
  assert.equal(diffusionBlockSwapPayload(true, "count", 0), undefined);
  assert.equal(diffusionBlockSwapPayload(true, "count", 99999), 1024);
});

test("the status line names blocks and pinned RAM", () => {
  assert.equal(diffusionBlockSwapSummary(null), null);
  assert.equal(
    diffusionBlockSwapSummary({ swapped_blocks: 14, total_blocks: 57, pinned_gib: 5.5 }),
    "14 of 57 blocks from host RAM (5.5 GiB pinned)",
  );
  assert.equal(
    diffusionBlockSwapSummary({ swapped_blocks: 0, total_blocks: 30, pinned_gib: 0 }),
    "none needed (0 of 30 blocks)",
  );
});

test("the panel wires the control, the request field and the status line", async () => {
  const src = await readSrcAsync("features/images/train/diffusion-train-panel.tsx");
  assert.match(src, /block_swap: diffusionBlockSwapPayload\(blockSwapApplies, blockSwapMode, blockSwapCount\)/);
  assert.match(src, /\{blockSwapApplies && \(/);
  assert.match(src, /supportsBlockSwap: reportedFamily\?\.supports_block_swap/);
  assert.match(src, /diffusionBlockSwapSummary\(status\?\.block_swap\)/);
});
