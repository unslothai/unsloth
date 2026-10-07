// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hydration must be able to CLEAR a local record: a merge that comes out default IS the clear,
// and quick select reads the record via resolveInitialConfig without opening the panel.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { storage } = installLocalStorageFake();

const { fromApiOverride, toApiOverride } = await import(
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

test("int8 prefill reaches the server row only when on", () => {
  const on = { ...DEFAULT_PER_MODEL_CONFIG, mlxInt8Prefill: true };
  assert.equal(toApiOverride(on).mlx_int8_prefill, true);
  assert.equal("mlx_int8_prefill" in toApiOverride(DEFAULT_PER_MODEL_CONFIG), false);
  assert.equal(fromApiOverride({ mlx_int8_prefill: true }).mlxInt8Prefill, true);
  assert.equal(fromApiOverride({}, on).mlxInt8Prefill, true);
});

test("a server row holding the superseded cache width hydrates as the single choice", () => {
  assert.equal(fromApiOverride({ mlx_kv_bits: 8 }).mlxKvQuant, "8");
  assert.equal(fromApiOverride({ mlx_kv_quant: "tq-4", mlx_kv_bits: 8 }).mlxKvQuant, "tq-4");
  const local = { ...DEFAULT_PER_MODEL_CONFIG, mlxKvQuant: "tq-3.5" as const };
  assert.equal(fromApiOverride({ mlx_kv_quant: "auto", mlx_kv_bits: 4 }, local).mlxKvQuant, null);
  assert.equal(
    fromApiOverride({ mlx_kv_quant: null } as never, local).mlxKvQuant,
    null,
  );
  assert.equal(fromApiOverride({}, local).mlxKvQuant, "tq-3.5");
});

test("an explicit server clear leaves a config the panel must still persist", () => {
  const stored = { ...DEFAULT_PER_MODEL_CONFIG, llamaExtraArgs: ["--flash-attn"] };
  // [] is the route's tombstone for an explicit clear, distinct from an absent field.
  const merged = fromApiOverride({ llama_extra_args: [] }, stored);

  assert.deepEqual(merged.llamaExtraArgs, [], "the clear must survive the merge");
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
  // savePerModelConfig returns false (never throws) for a full store or a newer-build record.
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
  assert.deepEqual(resolveInitialConfig(MODEL, VARIANT).config.llamaExtraArgs, [
    "--flash-attn",
  ]);
});

test("hydration does not mark itself saved when the write failed", () => {
  // A failed write must stay a pending change so Save and its error toast remain in reach.
  const src = readSrc("features/model-picker/components/model-config-page.tsx").replace(/\s+/g, " ");

  assert.match(
    src,
    /const hydrationSaved = savePerModelConfig\( configId, target\.ggufVariant, rememberedConfig, hydrationEvicted, \); setSavedRemember\(hydrationSaved\);/,
  );
});

test("hydration propagates what its own write evicted", () => {
  // savePerModelConfig silently evicts to stay within 500 entries / 1 MiB; clear the evicted.
  const src = readSrc("features/model-picker/components/model-config-page.tsx").replace(/\s+/g, " ");

  assert.match(
    src,
    /savePerModelConfig\( configId, target\.ggufVariant, rememberedConfig, hydrationEvicted, \)/,
  );
  assert.match(
    src,
    /for \(const dropped of hydrationEvicted\) \{ syncModelOverride\(dropped\.modelId, dropped\.ggufVariant, null, \{ keepLaunchFlags: true, \}\); \}/,
  );
});

test("hydration keeps a moved context pin in one field", () => {
  // The picker reads customContextLength first and auto-switch reads max_seq_length first.
  const legacy = { maxSeqLength: 8192, customContextLength: null };
  const moved = fromApiOverride({ custom_context_length: 32768 }, legacy as any);
  assert.equal(moved.customContextLength, 32768);
  assert.equal(moved.maxSeqLength, null, "the stale legacy field must not survive");

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
