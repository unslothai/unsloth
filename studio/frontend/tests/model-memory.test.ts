// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { computeModelMemory, formatKvRate, formatMemoryGb } = await import(
  "../src/lib/model-memory.ts"
);

const GB = 1024 ** 3;

// 16 GiB card -> 15.52 GiB usable at the shared 0.97 fraction, which is what
// the loader admits at (_CTX_FIT_VRAM_FRACTION). This was 0.90 / 14.4, a value
// llama_cpp.py records as already tried and reverted: "0.90 dropped 91-94% fits
// to CPU offload, #5106".
const GPU_GB = 16;
const BUDGET_GB = 15.52;

test("no GPU or no weights cannot be charted", () => {
  assert.equal(
    computeModelMemory({ weightsBytes: 8 * GB, gpuGb: 0 }).status,
    "unknown",
  );
  assert.equal(
    computeModelMemory({ weightsBytes: null, gpuGb: GPU_GB }).status,
    "unknown",
  );
});

test("weights and context inside the budget report fits", () => {
  const result = computeModelMemory({
    weightsBytes: 6 * GB,
    kvBytes: 2 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(result.status, "fits");
  assert.equal(result.modelGb, 6);
  assert.equal(result.kvGb, 2);
  assert.equal(result.totalGb, 8);
  assert.ok(Math.abs(result.budgetGb - BUDGET_GB) < 1e-9);
});

// The case the warning exists for: nothing wrong with the model, everything
// wrong with the settings on top of it.
test("weights fit but context pushes past the budget", () => {
  const result = computeModelMemory({
    weightsBytes: 12 * GB,
    kvBytes: 4 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(result.status, "context-exceeds");
});

test("oversized weights are not blamed on the context", () => {
  const result = computeModelMemory({
    weightsBytes: 20 * GB,
    kvBytes: 1 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(result.status, "model-exceeds");
});

test("speculative reserve counts toward the context segment", () => {
  const withoutSpec = computeModelMemory({
    weightsBytes: 10 * GB,
    kvBytes: 3 * GB,
    gpuGb: GPU_GB,
  });
  const withSpec = computeModelMemory({
    weightsBytes: 10 * GB,
    kvBytes: 3 * GB,
    // 3, not 2: at the 0.97 budget the old figure no longer tipped this over,
    // so the test would have passed while measuring nothing.
    specBytes: 3 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(withoutSpec.status, "fits");
  assert.equal(withSpec.kvGb + withSpec.specGb, 6);
  // Turning on speculative decoding is what tips this one over.
  assert.equal(withSpec.status, "context-exceeds");
});

test("segments never overflow the track", () => {
  const result = computeModelMemory({
    weightsBytes: 13 * GB,
    kvBytes: 40 * GB,
    gpuGb: GPU_GB,
  });
  assert.ok(result.modelPct + result.kvPct + result.specPct <= 100 + 1e-9);
  assert.ok(result.kvPct >= 0);
});

test("an oversized model alone fills the track without a negative context", () => {
  const result = computeModelMemory({
    weightsBytes: 50 * GB,
    kvBytes: 5 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(result.modelPct, 100);
  assert.equal(result.kvPct, 0);
});

test("a missing context estimate still charts the weights", () => {
  const result = computeModelMemory({
    weightsBytes: 6 * GB,
    kvBytes: null,
    gpuGb: GPU_GB,
  });
  assert.equal(result.status, "fits");
  assert.equal(result.kvGb, 0);
  assert.ok(result.modelPct > 0);
});

test("memory labels stay compact", () => {
  assert.equal(formatMemoryGb(0), "0 GiB");
  assert.equal(formatMemoryGb(7.24), "7.2 GiB");
  assert.equal(formatMemoryGb(23.6), "24 GiB");
});

test("KV and speculative reserve are separate segments", () => {
  const result = computeModelMemory({
    weightsBytes: 6 * GB,
    kvBytes: 2 * GB,
    specBytes: 1 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(result.kvGb, 2);
  assert.equal(result.specGb, 1);
  // Still summed for callers that draw context as one block.
  assert.equal(result.kvGb + result.specGb, 3);
});

test("three segments never overflow the track", () => {
  const result = computeModelMemory({
    weightsBytes: 13 * GB,
    kvBytes: 40 * GB,
    specBytes: 20 * GB,
    gpuGb: GPU_GB,
  });
  const sum = result.modelPct + result.kvPct + result.specPct;
  assert.ok(sum <= 100 + 1e-9, `segments summed to ${sum}`);
  assert.ok(result.specPct >= 0);
});

test("per-token KV rate derives from the context it was measured at", () => {
  const result = computeModelMemory({
    weightsBytes: 6 * GB,
    kvBytes: 1024 * 1024 * 1024,
    nCtx: 131072,
    gpuGb: GPU_GB,
  });
  // 1 GiB over 131072 tokens is exactly 8 KiB/token.
  assert.equal(result.kvBytesPerToken, 8192);
});

test("no context length means no rate rather than a wrong one", () => {
  const result = computeModelMemory({
    weightsBytes: 6 * GB,
    kvBytes: 2 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(result.kvBytesPerToken, 0);
});

test("KV rate labels pick sane units", () => {
  // KiB/MiB, not KB/MB. The divides are by 1024, so the old labels were the
  // same mislabel as the GB/GiB one a scale up.
  assert.equal(formatKvRate(0), "0 KiB");
  assert.equal(formatKvRate(6234), "6.1 KiB");
  assert.equal(formatKvRate(1024 * 1024 * 3), "3.0 MiB");
  // Non-finite input has no honest rendering and must not print "NaN KiB".
  assert.equal(formatKvRate(Number.NaN), "0 KiB");
  assert.equal(formatKvRate(-1), "0 KiB");
});

test("an unloadable model is flagged, not silently drawn as full", () => {
  // 184 GB of weights on a 128 GB host: no context setting can rescue this,
  // so it must still say something rather than rely on a fit badge elsewhere.
  const result = computeModelMemory({
    weightsBytes: 184 * GB,
    kvBytes: 4 * GB,
    gpuGb: 128,
  });
  assert.equal(result.status, "model-exceeds");
  assert.equal(result.modelPct, 100);
  assert.equal(result.kvPct, 0);
  assert.ok(result.totalGb > result.budgetGb);
});

test("bar holds the accent below 80% of budget", () => {
  // 14.4 GB budget; 11 GB total is ~76%.
  const result = computeModelMemory({
    weightsBytes: 9 * GB,
    kvBytes: 2 * GB,
    gpuGb: GPU_GB,
  });
  assert.equal(result.pressure, "normal");
  assert.ok(result.fillPct < 80);
});

test("bar warns from 80% and turns critical from 90%", () => {
  // Figures rescaled for the 15.52 GiB budget: 13/15.52 = 83.8%, 14.5/15.52 =
  // 93.4%. The bands themselves are unchanged.
  const high = computeModelMemory({
    weightsBytes: 11 * GB,
    kvBytes: 2 * GB,
    gpuGb: GPU_GB,
  });
  assert.ok(high.fillPct >= 80 && high.fillPct < 90, `got ${high.fillPct}`);
  assert.equal(high.pressure, "high");

  const critical = computeModelMemory({
    weightsBytes: 13 * GB,
    kvBytes: 1.5 * GB,
    gpuGb: GPU_GB,
  });
  assert.ok(critical.fillPct >= 90, `got ${critical.fillPct}`);
  assert.equal(critical.pressure, "critical");
});

test("fill percentage is uncapped so over-budget stays distinguishable", () => {
  const result = computeModelMemory({
    weightsBytes: 184 * GB,
    kvBytes: 4 * GB,
    gpuGb: 128,
  });
  // Widths clamp to the track, but the pressure read must not.
  assert.ok(result.fillPct > 100, `got ${result.fillPct}`);
  assert.equal(result.pressure, "critical");
  assert.equal(result.modelPct, 100);
});
