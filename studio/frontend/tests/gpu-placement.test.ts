// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  GPU_LAYERS_AUTO,
  resolveComparePlacement,
  recoverDroppedDiffusionSplit,
  resolveStagedDiffusionClassification,
  shouldHydrateGpuPlacementControls,
  shouldPinDiffusionPlacement,
} from "../src/features/chat/lib/gpu-placement.ts";

// The Send-time snapshot: Manual with 12 of another chat GGUF's layers on GPU.
const shared = { gpuMemoryMode: "manual" as const, gpuLayers: 12 };

test("a Vulkan CPU fallback does not replace the standing GPU intent", () => {
  assert.equal(
    shouldHydrateGpuPlacementControls("vulkan_startup_crash"),
    false,
  );
  assert.equal(shouldHydrateGpuPlacementControls(null), true);
});

test("a diffusion pane never inherits another model's layer split", () => {
  // An unremembered pane carries neither key, so `??` would fall through to shared.
  assert.deepEqual(resolveComparePlacement({}, shared, true), {
    gpuMemoryMode: "auto",
    gpuLayers: GPU_LAYERS_AUTO,
  });
});

test("a leaked zero cannot mask a diffusion pane's devices", () => {
  assert.deepEqual(
    resolveComparePlacement(
      {},
      { gpuMemoryMode: "manual", gpuLayers: 0 },
      true,
    ),
    { gpuMemoryMode: "auto", gpuLayers: GPU_LAYERS_AUTO },
  );
});

test("a diffusion pane's OWN split is honoured (#7574)", () => {
  assert.deepEqual(
    resolveComparePlacement(
      { gpuMemoryMode: "manual", gpuLayers: 0 },
      shared,
      true,
    ),
    { gpuMemoryMode: "manual", gpuLayers: 0 },
  );
});

test("a chat GGUF pane still inherits the Send-time snapshot", () => {
  assert.deepEqual(resolveComparePlacement({}, shared, false), {
    gpuMemoryMode: "manual",
    gpuLayers: 12,
  });
});

test("an own value wins over the snapshot for a chat GGUF too", () => {
  assert.deepEqual(
    resolveComparePlacement(
      { gpuMemoryMode: "auto", gpuLayers: 3 },
      shared,
      false,
    ),
    { gpuMemoryMode: "auto", gpuLayers: 3 },
  );
});

// ── an unclassified GGUF must not inherit the split either ──
// An undownloaded GGUF with no "DiffusionGemma" in its name comes back
// is_diffusion=false + diffusion_unknown=true; /load may then read a diffusion
// header and apply whatever split the request carried.

test("an unclassified GGUF pane gets the diffusion-safe placement", () => {
  assert.equal(shouldPinDiffusionPlacement(true, false, true), true);
  assert.deepEqual(
    resolveComparePlacement(
      {},
      shared,
      shouldPinDiffusionPlacement(true, false, true),
    ),
    { gpuMemoryMode: "auto", gpuLayers: GPU_LAYERS_AUTO },
  );
});

test("an unclassified GGUF pane cannot inherit a CPU-masking zero", () => {
  assert.deepEqual(
    resolveComparePlacement(
      {},
      { gpuMemoryMode: "manual", gpuLayers: 0 },
      shouldPinDiffusionPlacement(true, false, true),
    ),
    { gpuMemoryMode: "auto", gpuLayers: GPU_LAYERS_AUTO },
  );
});

test("a CLASSIFIED ordinary GGUF still inherits the snapshot", () => {
  // The point of the tri-state: the common path keeps its existing inheritance.
  assert.equal(shouldPinDiffusionPlacement(true, false, false), false);
  assert.deepEqual(
    resolveComparePlacement(
      {},
      shared,
      shouldPinDiffusionPlacement(true, false, false),
    ),
    { gpuMemoryMode: "manual", gpuLayers: 12 },
  );
});

test("a confirmed diffusion GGUF is pinned however it was classified", () => {
  assert.equal(shouldPinDiffusionPlacement(true, true, false), true);
  assert.equal(shouldPinDiffusionPlacement(true, true, true), true);
});

test("a non-GGUF pane keeps inheriting the snapshot", () => {
  // It sends no placement at all, so an unknown flag must not flip it to Auto.
  assert.equal(shouldPinDiffusionPlacement(false, undefined, false), false);
  assert.equal(shouldPinDiffusionPlacement(false, undefined, true), false);
});

test("an own split still wins for an unclassified GGUF", () => {
  assert.deepEqual(
    resolveComparePlacement(
      { gpuMemoryMode: "manual", gpuLayers: 6 },
      shared,
      shouldPinDiffusionPlacement(true, false, true),
    ),
    { gpuMemoryMode: "manual", gpuLayers: 6 },
  );
});

// -- the config-picker path must hand on "unknown", not a definite false --
// onRun passes the probe answer into the selection, which becomes
// sel.isDiffusion in the compare flow. A definite false there skips the pane's
// re-probe, so the unknown state has to survive this hop.

test("an inconclusive staged probe stays unknown", () => {
  assert.equal(
    resolveStagedDiffusionClassification(undefined, {
      isDiffusion: false,
      diffusionUnknown: true,
    }),
    undefined,
  );
});

test("a confirmed ordinary GGUF stays a definite false", () => {
  assert.equal(
    resolveStagedDiffusionClassification(undefined, {
      isDiffusion: false,
      diffusionUnknown: false,
    }),
    false,
  );
});

test("a confirmed diffusion GGUF stays a definite true", () => {
  assert.equal(
    resolveStagedDiffusionClassification(undefined, {
      isDiffusion: true,
      diffusionUnknown: false,
    }),
    true,
  );
});

test("an already-known diffusion target short-circuits the probe", () => {
  assert.equal(resolveStagedDiffusionClassification(true, null), true);
});

test("a pending probe is unknown, not ordinary", () => {
  assert.equal(resolveStagedDiffusionClassification(undefined, null), undefined);
  assert.equal(
    resolveStagedDiffusionClassification(undefined, undefined),
    undefined,
  );
});

test("the unknown verdict re-probes and reaches diffusion-safe placement", () => {
  // End to end: unknown -> undefined -> the compare preflight re-probes.
  const handedOn = resolveStagedDiffusionClassification(undefined, {
    isDiffusion: false,
    diffusionUnknown: true,
  });
  assert.equal(handedOn, undefined);
  assert.deepEqual(
    resolveComparePlacement({}, shared, shouldPinDiffusionPlacement(true, handedOn, true)),
    { gpuMemoryMode: "auto", gpuLayers: GPU_LAYERS_AUTO },
  );
});

// -- a dropped split must survive a browser refresh --
// A shim without --ngl runs Auto and the backend keeps the ask in
// diffusion_requested_ngl. A refresh starts the store at Auto, so the response
// has to carry it back or the next Apply sends manual/-1 and the post-upgrade
// retry has nothing to apply.

test("an auto diffusion response recovers the standing ask", () => {
  assert.equal(recoverDroppedDiffusionSplit(true, "auto", 20), 20);
});

test("a zero-layer ask is recovered, not treated as absent", () => {
  assert.equal(recoverDroppedDiffusionSplit(true, "auto", 0), 0);
});

test("an applied manual split is authoritative, so nothing is recovered", () => {
  // mode "manual" means the shim honoured the split: gpu_layers is the truth.
  assert.equal(recoverDroppedDiffusionSplit(true, "manual", 20), null);
});

test("no standing ask means nothing to recover", () => {
  assert.equal(recoverDroppedDiffusionSplit(true, "auto", null), null);
  assert.equal(recoverDroppedDiffusionSplit(true, "auto", undefined), null);
});

test("a non-diffusion response never recovers a split", () => {
  assert.equal(recoverDroppedDiffusionSplit(false, "auto", 20), null);
  assert.equal(recoverDroppedDiffusionSplit(undefined, "auto", 20), null);
});

test("an older backend without the field leaves the response unchanged", () => {
  // Absent field -> undefined -> nothing recovered.
  assert.equal(recoverDroppedDiffusionSplit(true, "auto", undefined), null);
});
