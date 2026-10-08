// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The YAML export writes every offload field, so an import has to read every one back.

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { mapBackendModelConfigToTrainingPatch } = await import(
  "../src/features/training/lib/model-defaults.ts"
);

test("importing an exported config restores every offload field", () => {
  const patch = mapBackendModelConfigToTrainingPatch({
    training: { offload_layers: "auto", offload_vram_gb: 12.5, prefetch_depth: "auto" },
  });
  assert.equal(patch.offloadLayers, "auto");
  assert.equal(patch.offloadVramGb, 12.5);
  assert.equal(patch.prefetchDepth, "auto");
  const off = mapBackendModelConfigToTrainingPatch({
    training: { offload_layers: 14, offload_vram_gb: null, prefetch_depth: 3 },
  });
  assert.equal(off.offloadVramGb, null);
  assert.equal(off.prefetchDepth, 3);
  // Configs exported before these fields existed keep the current values.
  assert.equal("offloadVramGb" in mapBackendModelConfigToTrainingPatch({ training: {} }), false);
});
