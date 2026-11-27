// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

/**
 * One-time import of the legacy `unsloth_load_settings` store into `unsloth_model_configs`.
 * Asserted only through resolveInitialConfig, so re-keying or re-versioning records is free.
 */

const MODULE_PATH = fileURLToPath(
  new URL(
    "../src/features/model-picker/model-config/per-model-config.ts",
    import.meta.url,
  ),
);

const MODEL_ID = "unsloth/gemma-3-270m-it-GGUF";
const VARIANT = "UD-Q4_K_XL";
const LEGACY_KEY = `${MODEL_ID}::${VARIANT}`;
const CTX = 4096;

/**
 * A fresh module copy: the migration is latched by both a persistent flag and a module-level
 * `legacyMigrationChecked`. The query must sit on an absolute `file:` URL because
 * bundler-resolver.mjs round-trips relative specifiers through fileURLToPath, dropping it.
 */
async function freshModule() {
  const url = pathToFileURL(MODULE_PATH);
  url.search = `?fresh=${Math.random()}`;
  return (await import(url.href)) as typeof import(
    "../src/features/model-picker/model-config/per-model-config.ts"
  );
}

function seedLegacy(entry: Record<string, unknown>, key = LEGACY_KEY): void {
  store.set("unsloth_load_settings", JSON.stringify({ [key]: entry }));
}

const FULL_LEGACY_ENTRY = {
  contextLength: CTX,
  kvCacheDtype: "q8_0",
  tensorParallel: true,
};

test("a legacy entry is remembered with the values it carried", async () => {
  store.clear();
  seedLegacy(FULL_LEGACY_ENTRY);
  const { resolveInitialConfig } = await freshModule();

  const { config, remembered } = resolveInitialConfig(MODEL_ID, VARIANT);

  assert.equal(remembered, true);
  assert.equal(config.customContextLength, CTX);
  assert.equal(config.kvCacheDtype, "q8_0");
  assert.equal(config.tensorParallel, true);
});

test("the model is remembered under the id and quant the picker asks by, whatever the legacy key spelled", async () => {
  store.clear();
  // The legacy key folds repo id and quant on the last "::"; lookup must be case-insensitive.
  seedLegacy(FULL_LEGACY_ENTRY);
  const { resolveInitialConfig } = await freshModule();

  const asked = resolveInitialConfig(MODEL_ID.toLowerCase(), VARIANT.toLowerCase());

  assert.equal(asked.remembered, true);
  assert.equal(asked.config.customContextLength, CTX);
});

test("migrating once is enough: a later legacy store is not imported again", async () => {
  store.clear();
  seedLegacy(FULL_LEGACY_ENTRY);
  const first = await freshModule();
  assert.equal(first.resolveInitialConfig(MODEL_ID, VARIANT).config.customContextLength, CTX);

  // Re-running the import on every reload would resurrect models the user has since forgotten.
  seedLegacy({ contextLength: CTX + 2048, tensorParallel: true }, "unsloth/other-model::Q4_K_M");
  const second = await freshModule();

  assert.equal(
    second.resolveInitialConfig("unsloth/other-model", "Q4_K_M").remembered,
    false,
    "a second document re-ran the one-time legacy import",
  );
  assert.equal(second.resolveInitialConfig(MODEL_ID, VARIANT).config.customContextLength, CTX);
});

test("settings saved in this build outrank a legacy blob for the same model", async () => {
  store.clear();
  // Saved FIRST: seed-then-save would trigger the import on save and skip the precedence branch.
  const saver = await freshModule();
  saver.savePerModelConfig(MODEL_ID, VARIANT, {
    ...saver.DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 16384,
  });

  // Any versioned record is newer than the legacy blob, so the legacy value must not win.
  store.delete("unsloth_model_configs_migrated");
  seedLegacy(FULL_LEGACY_ENTRY);

  const fresh = await freshModule();
  const { config } = fresh.resolveInitialConfig(MODEL_ID, VARIANT);

  assert.equal(config.customContextLength, 16384);
  assert.equal(config.kvCacheDtype, null);
});

test("a legacy blob carrying nothing but defaults does not make a model look remembered", async () => {
  store.clear();
  // An all-defaults record would tick "Remember for this model" for an unconfigured model.
  seedLegacy({ tensorParallel: false });
  const { resolveInitialConfig } = await freshModule();

  assert.equal(resolveInitialConfig(MODEL_ID, VARIANT).remembered, false);
});

test("no legacy store at all leaves the model unremembered rather than throwing", async () => {
  store.clear();
  const { resolveInitialConfig } = await freshModule();

  assert.equal(resolveInitialConfig(MODEL_ID, VARIANT).remembered, false);
});

test("an unreadable legacy store is survived, not propagated", async () => {
  store.clear();
  store.set("unsloth_load_settings", "{not json");
  const { resolveInitialConfig } = await freshModule();

  assert.equal(resolveInitialConfig(MODEL_ID, VARIANT).remembered, false);
});
