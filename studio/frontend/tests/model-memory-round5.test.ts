// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { computeModelMemory, extraArgsShapeKvCache } = await import(
  "../src/lib/model-memory.ts"
);

const GB = 1024 ** 3;

test("a drafter's fixed weights cannot be auto-fitted away", () => {
  // Drafter weights are resident at any context length, so no shorter context helps.
  const segments = computeModelMemory({
    weightsBytes: 14 * GB,
    specBytes: 9 * GB,
    specFixedBytes: 8 * GB,
    kvBytes: 4 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
    contextIsAutoFitted: true,
  });
  assert.equal(
    segments.status,
    "model-exceeds",
    "auto-fit softening swallowed an overage no context change can fix",
  );
});

test("auto-fit still softens a purely context-driven overage", () => {
  const segments = computeModelMemory({
    weightsBytes: 14 * GB,
    kvBytes: 20 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
    contextIsAutoFitted: true,
  });
  assert.equal(segments.status, "fits");
});

test("context checkpoints are not charged against the card", () => {
  // llama.cpp keeps SWA checkpoints in host heap, so they are not VRAM.
  const kvBytes = 18 * GB;
  const kvCheckpointBytes = 12 * GB;
  const onCard = computeModelMemory({
    weightsBytes: 6 * GB,
    kvBytes: kvBytes - kvCheckpointBytes,
    gpuGb: 24,
    budgetFraction: 0.9,
    nCtx: 32768,
  });
  const everythingCharged = computeModelMemory({
    weightsBytes: 6 * GB,
    kvBytes,
    gpuGb: 24,
    budgetFraction: 0.9,
    nCtx: 32768,
  });
  assert.equal(onCard.status, "fits");
  assert.equal(
    everythingCharged.status,
    "context-exceeds",
    "test is not exercising the difference it claims to",
  );
});

test("KV-shaping pass-through args are recognised", () => {
  assert.equal(extraArgsShapeKvCache(["--swa-full"]), true);
  assert.equal(extraArgsShapeKvCache(["--ctx-size=131072"]), true);
  assert.equal(extraArgsShapeKvCache(["-ub", "2048"]), true);
  assert.equal(extraArgsShapeKvCache(["--verbose"]), false);
  assert.equal(extraArgsShapeKvCache([]), false);
  assert.equal(extraArgsShapeKvCache(null), false);
});

test("a mixed shared-memory host is judged on dedicated VRAM only", () => {
  // sharedMemory is every(), so false here: 26 GiB fits the 36 GiB combined figure, not the card.
  const combined = computeModelMemory({
    weightsBytes: 26 * GB,
    gpuGb: 36,
    budgetFraction: 0.9,
  });
  const dedicated = computeModelMemory({
    weightsBytes: 26 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
  });
  assert.equal(combined.status, "fits");
  assert.equal(
    dedicated.status,
    "model-exceeds",
    "the dedicated-only budget must still refuse a model larger than the card",
  );
});

test("a CPU-resident launch draws no VRAM bar", () => {
  // Zero planner GPU bytes is an answer; a `||` fallback would replace it.
  const segments = computeModelMemory({
    weightsBytes: 8 * GB,
    kvBytes: 2 * GB,
    gpuTotalBytes: 0,
    gpuGb: 24,
    budgetFraction: 0.9,
  });
  assert.equal(segments.status, "unknown");
});

test("the planner's total wins over the segment sum", () => {
  // The planner total counts terms the segments cannot, so a lower total must still win.
  const segments = computeModelMemory({
    weightsBytes: 10 * GB,
    kvBytes: 4 * GB,
    gpuTotalBytes: 20 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
  });
  assert.equal(Math.round(segments.totalGb), 20);
});

test("KV-shaping recognises the flags that override structured settings", () => {
  assert.equal(extraArgsShapeKvCache(["--flash-attn", "off"]), true);
  assert.equal(extraArgsShapeKvCache(["-fa", "off"]), true);
  assert.equal(extraArgsShapeKvCache(["--spec-type", "draft-mtp"]), true);
  assert.equal(extraArgsShapeKvCache(["--spec-draft-n-max=8"]), true);
});

test("an auto-fitted row does not paint red for a context it will not open", () => {
  const segments = computeModelMemory({
    weightsBytes: 8 * GB,
    kvBytes: 40 * GB,
    gpuFloorBytes: 9 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
    contextIsAutoFitted: true,
  });
  assert.equal(segments.status, "fits");
  assert.ok(
    segments.fillPct <= 100,
    `auto-fitted pressure read ${segments.fillPct}% of budget`,
  );
  assert.notEqual(segments.pressure, "critical");
});

test("a pinned row still reports the pressure it really has", () => {
  const segments = computeModelMemory({
    weightsBytes: 8 * GB,
    kvBytes: 40 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
    contextIsAutoFitted: false,
  });
  assert.equal(segments.status, "context-exceeds");
  assert.ok(segments.fillPct > 100);
  assert.equal(segments.pressure, "critical");
});

test("a pinned context still warns even when nothing was pinned in the UI", () => {
  // An inherited LLAMA_ARG_CTX_SIZE is kept by the loader, so it is reported as pinned.
  const inherited = computeModelMemory({
    weightsBytes: 8 * GB,
    kvBytes: 40 * GB,
    gpuFloorBytes: 9 * GB,
    gpuGb: 24,
    budgetFraction: 0.9,
    contextIsAutoFitted: false,
  });
  assert.equal(inherited.status, "context-exceeds");
  assert.ok(inherited.fillPct > 100);
});
