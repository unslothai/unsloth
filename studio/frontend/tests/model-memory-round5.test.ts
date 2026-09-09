// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The frontend half of the round-5 review: three ways the bar reported a
// confident "fits" for a load that would not fit.
//
// Each case here failed before its fix, and each is a false NEGATIVE -- the bar
// staying quiet when it should warn. That direction matters more than the
// opposite one: a spurious warning is an annoyance, while a missing one is the
// whole feature failing silently at the moment it was supposed to earn its keep.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { computeModelMemory, extraArgsShapeKvCache } = await import(
  "../src/lib/model-memory.ts"
);

const GB = 1024 ** 3;

test("a drafter's fixed weights cannot be auto-fitted away", () => {
  // 24 GiB card at the default fraction. Target weights fit alone; target plus
  // an 8 GiB drafter do not, and no shorter context can recover that -- the
  // drafter's weights are resident whatever the context length is.
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
  // The counterpart, so the fix above does not simply warn on everything: with
  // no fixed speculative cost the KV term alone is reducible, and an unpinned
  // row must stay quiet exactly as it did before.
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
  // llama.cpp keeps SWA checkpoint snapshots in host heap, so a VRAM bar that
  // counts them warns OOM over memory that never reaches the GPU. Modelled the
  // way the hook does it: the host share subtracted from the cache figure.
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
  // --swa-full replaces a sliding window with a full-context cache, so a bar
  // priced from the structured controls alone is describing a different load.
  assert.equal(extraArgsShapeKvCache(["--swa-full"]), true);
  assert.equal(extraArgsShapeKvCache(["--ctx-size=131072"]), true);
  assert.equal(extraArgsShapeKvCache(["-ub", "2048"]), true);
  // Placement flags are a separate category with its own guard; this one must
  // not claim them, or the two abstention reasons become indistinguishable.
  assert.equal(extraArgsShapeKvCache(["--verbose"]), false);
  assert.equal(extraArgsShapeKvCache([]), false);
  assert.equal(extraArgsShapeKvCache(null), false);
});

test("a mixed shared-memory host is judged on dedicated VRAM only", () => {
  // A 24 GiB discrete card beside a Vulkan iGPU reporting 12 GiB of free system
  // RAM. `sharedMemory` is every(), so it reads false here and the dedicated-vs-
  // combined choice is the only thing standing between this model and a wrong
  // verdict: 26 GiB fits the 36 GiB combined figure and does not fit the card.
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
  // Inherited placement (LLAMA_ARG_DEVICE=none) makes the planner report zero
  // GPU bytes. That is an answer, not a missing one, and a `||` fallback used to
  // swap it for the segment sum and draw pressure for a load that touches no
  // card at all.
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
  // The segments are assembled from separate fields and can only include what
  // this file knows to ask for; the planner's figure already counts the terms it
  // does not. A total below the sum still has to be taken, or the delegation is
  // decorative.
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
  // --flash-attn off changes the cache LAYOUT, and an extras --spec-type beats
  // the structured speculative mode outright, so both make the priced figure
  // describe a different launch.
  assert.equal(extraArgsShapeKvCache(["--flash-attn", "off"]), true);
  assert.equal(extraArgsShapeKvCache(["-fa", "off"]), true);
  assert.equal(extraArgsShapeKvCache(["--spec-type", "draft-mtp"]), true);
  assert.equal(extraArgsShapeKvCache(["--spec-draft-n-max=8"]), true);
});

test("an auto-fitted row does not paint red for a context it will not open", () => {
  // Priced at the native context, which the loader will reduce. The textual
  // verdict was already suppressed; the bar itself was not, so a model that
  // loads fine showed a full destructive bar and an over-budget readout.
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
  // The counterpart: with a context the user pinned there is no fitting to come,
  // so an over-budget total must still read as over budget.
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
  // An inherited LLAMA_ARG_CTX_SIZE is kept by the loader, not fitted, so the
  // route reports it as pinned. Before that flag existed the frontend read
  // "auto-fitted" from the absence of a saved context, which both suppressed the
  // overage and drew only the floor: a comfortable fit for a launch that OOMs.
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
