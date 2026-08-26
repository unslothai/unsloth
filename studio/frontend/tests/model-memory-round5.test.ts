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
