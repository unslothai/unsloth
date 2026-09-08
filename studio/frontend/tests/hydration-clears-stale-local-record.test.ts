// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hydration must be able to CLEAR a local record, not only overwrite one.
//
// savePerModelConfig says "no settings" by deleting the entry, so a merge that comes
// out default IS the clear. Skipping the write there strands the old value, and
// model-selector's quick select reads it via resolveInitialConfig without opening the
// panel, handing a flag cleared on another origin back to the next launch.
//
// Reachable shape: a model whose only setting is its extra arguments. Clearing them
// leaves an explicit empty list, which is a default config.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { installLocalStorageFake, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { storage } = installLocalStorageFake();

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

test("a hydration write can fail, and says so in its return rather than throwing", () => {
  // The failure the panel has to notice. savePerModelConfig returns false for a full
  // or unavailable store and for a record from a newer build it must not replace; it
  // does not throw, so an unchecked call is indistinguishable from a successful one.
  savePerModelConfig(MODEL, VARIANT, {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaExtraArgs: ["--flash-attn"],
  });
  const original = storage.setItem;
  storage.setItem = () => {
    // What a browser raises at the quota, and Safari in private mode on any write.
    throw new DOMException("QuotaExceededError", "QuotaExceededError");
  };
  try {
    assert.equal(
      savePerModelConfig(MODEL, VARIANT, {
        ...DEFAULT_PER_MODEL_CONFIG,
        llamaExtraArgs: ["--numa", "distribute"],
      }),
      false,
      "a failed write has to be reported, not swallowed",
    );
  } finally {
    storage.setItem = original;
  }
  // And the record the panel would be claiming to have replaced is still the old one.
  assert.deepEqual(resolveInitialConfig(MODEL, VARIANT).config.llamaExtraArgs, [
    "--flash-attn",
  ]);
});

test("hydration does not mark itself saved when the write failed", () => {
  // setRemember/setSavedRemember run before the write, so an ignored false left the
  // panel claiming the server settings were remembered while quick select and
  // background loads -- which read resolveInitialConfig and never open this panel --
  // still saw the stale record or none at all. Feeding the result back in makes it a
  // pending change instead, which is what puts Save (and its error toast) in reach.
  const src = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
    "utf8",
  ).replace(/\s+/g, " ");

  assert.match(
    src,
    /const hydrationSaved = savePerModelConfig\( configId, target\.ggufVariant, rememberedConfig, hydrationEvicted, \); setSavedRemember\(hydrationSaved\);/,
  );
});

test("hydration propagates what its own write evicted", () => {
  // Storage is capped at 500 entries and 1 MiB, and savePerModelConfig evicts to stay
  // inside it, silently, still reporting success. The save handler collects those and
  // clears their mirrored fields; hydration writes through the same budget, so a model
  // dropped here would keep applying its server row to API loads while quick select
  // read defaults for it, with nothing in the UI able to forget it.
  const src = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
    "utf8",
  ).replace(/\s+/g, " ");

  // The write hands savePerModelConfig somewhere to report evictions.
  assert.match(
    src,
    /savePerModelConfig\( configId, target\.ggufVariant, rememberedConfig, hydrationEvicted, \)/,
  );
  // And they are cleared the way the save path clears them: mirrored fields only.
  assert.match(
    src,
    /for \(const dropped of hydrationEvicted\) \{ syncModelOverride\(dropped\.modelId, dropped\.ggufVariant, null, \{ keepLaunchFlags: true, \}\); \}/,
  );
});

test("hydration keeps a moved context pin in one field", () => {
  // The picker reads customContextLength first and the auto-switch load max_seq_length
  // first, so a record holding both loads the same model at two lengths.
  const legacy = { maxSeqLength: 8192, customContextLength: null };
  const moved = fromApiOverride({ custom_context_length: 32768 }, legacy as any);
  assert.equal(moved.customContextLength, 32768);
  assert.equal(moved.maxSeqLength, null, "the stale legacy field must not survive");

  // The other direction: a row pinning in the pre-move field owns both too.
  const back = fromApiOverride(
    { max_seq_length: 8192 },
    { customContextLength: 32768, maxSeqLength: null } as any,
  );
  assert.equal(back.customContextLength, null);
  assert.equal(back.maxSeqLength, 8192);

  // A row stating no pin falls back to this browser's, or opening the panel deletes it.
  const kept = fromApiOverride({}, legacy as any);
  assert.equal(kept.maxSeqLength, 8192);
});
