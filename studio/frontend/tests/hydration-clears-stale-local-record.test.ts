// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hydration must be able to CLEAR a local record, not only overwrite one.
//
// The panel writes the resolved server row into browser storage so the load paths
// that never open the panel see the shared settings. savePerModelConfig expresses
// "no settings" by deleting the entry, so a merge that comes out default is the
// only way a clear travels. Skipping the write in exactly that case leaves the
// stale local record in place, and model-selector's quick select reads it through
// resolveInitialConfig without any hydration, so a flag cleared on another origin
// is handed straight back to the next launch.
//
// The reachable shape is an arguments-only record: clearing the one setting a model
// had leaves the row as an explicit empty list, which is a default config.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { installLocalStorageFake, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const { fromApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const {
  DEFAULT_PER_MODEL_CONFIG,
  isDefaultConfig,
  resolveInitialConfig,
  savePerModelConfig,
} = await import("../src/features/model-picker/model-config/per-model-config.ts");

const MODEL = "unsloth/Model-GGUF";
const VARIANT = "q4_k_m";

test("an explicit server clear leaves a config the panel must still persist", () => {
  // What this browser remembers: one flag, nothing else.
  const stored = { ...DEFAULT_PER_MODEL_CONFIG, llamaExtraArgs: ["--flash-attn"] };
  // What another origin left behind after clearing it. [] is the tombstone the
  // route uses for an explicit clear, distinct from an absent field.
  const merged = fromApiOverride({ llama_extra_args: [] }, stored);

  assert.deepEqual(merged.llamaExtraArgs, [], "the clear must survive the merge");
  // And the merged config is default, which is what makes the write conditional
  // on !isDefaultConfig skip precisely the case that needs to travel.
  assert.equal(isDefaultConfig(merged), true);
});

test("persisting that config removes the stale entry rather than keeping it", () => {
  savePerModelConfig(MODEL, VARIANT, {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaExtraArgs: ["--flash-attn"],
  });
  assert.deepEqual(
    resolveInitialConfig(MODEL, VARIANT).config.llamaExtraArgs,
    ["--flash-attn"],
    "precondition: the stale record exists",
  );

  const merged = fromApiOverride(
    { llama_extra_args: [] },
    resolveInitialConfig(MODEL, VARIANT).config,
  );
  savePerModelConfig(MODEL, VARIANT, merged);

  const after = resolveInitialConfig(MODEL, VARIANT);
  assert.equal(
    after.remembered,
    false,
    "the cleared flag must not survive for quick select to reload",
  );
  assert.ok(
    after.config.llamaExtraArgs == null ||
      after.config.llamaExtraArgs.length === 0,
  );
});

test("the hydration write is not gated on the merge being non-default", () => {
  // The guard is what decides whether the clear reaches storage at all, so assert
  // it at the source. savePerModelConfig already no-ops when there is nothing to
  // delete, which is why it does not need a caller-side default check.
  const panel = new URL(
    "../src/features/model-picker/components/model-config-page.tsx",
    import.meta.url,
  );
  const src = readFileSync(panel, "utf8").replace(/\s+/g, " ");
  assert.doesNotMatch(
    src,
    /if \(!isDefaultConfig\(rememberedConfig\)\) \{ savePerModelConfig\(/,
    "a default merge is exactly the clear that has to be written",
  );
});
