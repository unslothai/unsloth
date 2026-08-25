// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

/**
 * The one-time import of a pre-feature `unsloth_load_settings` store into the versioned
 * `unsloth_model_configs` map.
 *
 * Until now this had no test at all. Its only coverage was one step of
 * tests/studio/playwright_model_config.py, which needs a browser, a booted Studio and a
 * downloaded GGUF, and which races that Studio's own network traffic -- so a defect here
 * surfaced as an intermittent red on a job nobody could run locally, and the assertion it
 * failed ("not migrated into unsloth_model_configs") named the migration for damage the
 * migration had not done. These pin the contract directly, in milliseconds.
 *
 * Behaviour, not storage shape. Every assertion below goes through the module's own
 * public read path (`resolveInitialConfig`), so re-keying or re-versioning the records is
 * free; only a change to what a user's remembered settings DO can turn these red.
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
 * A fresh copy of the module, because the migration is latched twice over: a persistent
 * `unsloth_model_configs_migrated` flag AND a module-level `legacyMigrationChecked`. A
 * second case sharing one instance would exercise the latch, not the migration.
 *
 * The query has to go on an absolute `file:` URL. tests/bundler-resolver.mjs round-trips a
 * relative specifier through `fileURLToPath`, which drops it, and every case would then
 * silently share the first instance.
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

  // The context is the field that went missing in CI, but asserting only it would let a
  // migration that carried nothing else pass.
  assert.equal(remembered, true);
  assert.equal(config.customContextLength, CTX);
  assert.equal(config.kvCacheDtype, "q8_0");
  assert.equal(config.tensorParallel, true);
});

test("the model is remembered under the id and quant the picker asks by, whatever the legacy key spelled", async () => {
  store.clear();
  // The legacy key folds a repo id and a quant into one string on the last "::". Case is
  // the picker's to normalise, so asking in a different one must still find it.
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

  // A fresh document (new module instance) with a DIFFERENT legacy store present. Re-running
  // the import on every reload is the regression that reverted the predecessor of #7207: a
  // model the user has since forgotten would come back from the legacy blob for ever.
  seedLegacy({ contextLength: CTX + 2048, tensorParallel: true }, "unsloth/other-model::Q4_K_M");
  const second = await freshModule();

  assert.equal(
    second.resolveInitialConfig("unsloth/other-model", "Q4_K_M").remembered,
    false,
    "a second document re-ran the one-time legacy import",
  );
  // and the first import is untouched.
  assert.equal(second.resolveInitialConfig(MODEL_ID, VARIANT).config.customContextLength, CTX);
});

test("settings saved in this build outrank a legacy blob for the same model", async () => {
  store.clear();
  // Saved FIRST, with no legacy store in sight, so this record exists before the import
  // has ever looked at the model. Ordering it the other way round -- seed, then save --
  // is the trap: the save itself triggers the import, so the precedence branch never
  // runs and the case passes no matter what that branch does. Verified by deleting the
  // `Object.hasOwn(map, key)` guard in mergeLegacyEntries, which this catches and the
  // seed-then-save ordering did not.
  const saver = await freshModule();
  saver.savePerModelConfig(MODEL_ID, VARIANT, {
    ...saver.DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 16384,
  });

  // The flag postdates the versioned store, so an install genuinely reaches this shape:
  // records written by this build, a legacy blob still lying about from before the
  // upgrade, and no record of the import having run. Whatever wrote the record knew
  // about the versioned store, so it is strictly newer than a blob no build has written
  // since. Letting the legacy value win here would be the real data loss.
  store.delete("unsloth_model_configs_migrated");
  seedLegacy(FULL_LEGACY_ENTRY);

  const fresh = await freshModule();
  const { config } = fresh.resolveInitialConfig(MODEL_ID, VARIANT);

  assert.equal(config.customContextLength, 16384);
  // and the legacy blob does not get to half-apply itself over the saved record either.
  assert.equal(config.kvCacheDtype, null);
});

test("a legacy blob carrying nothing but defaults does not make a model look remembered", async () => {
  store.clear();
  // An all-defaults record would show "Remember for this model" ticked for a model the
  // user never configured, and pin it against later changes to the app defaults.
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
