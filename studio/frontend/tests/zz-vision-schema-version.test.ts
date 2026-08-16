// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * S2, from batch-upgrade-downgrade-compat.test.ts: no user setting may be lost
 * on any upgrade or downgrade path. Hidden is acceptable, lost is not.
 *
 * A config whose only non-default field is `disableVision` must therefore be
 * stamped ABOVE the last pre-feature build's ceiling, so that build quarantines
 * the record instead of accepting it, dropping the unknown key, and destroying
 * the setting on its next rewrite.
 */

import assert from "node:assert/strict";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const { DEFAULT_PER_MODEL_CONFIG, resolveInitialConfig, savePerModelConfig } =
  await import("../src/features/model-picker/model-config/per-model-config.ts");

const STORAGE_KEY = "unsloth_model_configs";

// The ceiling shipped by the last build BEFORE the vision toggle. A record at
// or below this is readable — and therefore erasable — by that build.
const PRE_VISION_CEILING = 3;

function installLocalStorage(): void {
  store.clear();
}

function config(overrides: Partial<PerModelConfig> = {}): PerModelConfig {
  return { ...DEFAULT_PER_MODEL_CONFIG, ...overrides };
}

function storedRecord(): Record<string, unknown> {
  const raw = (store.get(STORAGE_KEY) ?? null);
  assert.ok(raw, "nothing was persisted");
  const map = JSON.parse(raw) as Record<string, Record<string, unknown>>;
  const keys = Object.keys(map);
  assert.equal(keys.length, 1, `expected one record, got ${keys.length}`);
  return map[keys[0]];
}

test("a vision-off record is stamped above the last pre-feature ceiling", () => {
  installLocalStorage();

  savePerModelConfig("some/vision-gguf", "Q4_K_M", config({ disableVision: true }));

  const record = storedRecord();
  assert.equal(record.disableVision, true);
  assert.ok(
    (record.version as number) > PRE_VISION_CEILING,
    `stamped v${record.version}; a pre-feature build (ceiling ${PRE_VISION_CEILING}) ` +
      "would accept this record, drop disableVision, and erase it on the next save",
  );
});

test("an old client's rewrite cannot reach a vision-off record", () => {
  installLocalStorage();
  savePerModelConfig("some/vision-gguf", "Q4_K_M", config({ disableVision: true }));

  // Replay exactly what a pre-feature build does: read every record it is
  // allowed to interpret, normalize it through a schema that has never heard of
  // disableVision, and write the result back.
  const map = JSON.parse((store.get(STORAGE_KEY) ?? null) as string) as Record<
    string,
    Record<string, unknown>
  >;
  let rewrote = false;
  for (const key of Object.keys(map)) {
    const version = typeof map[key].version === "number" ? (map[key].version as number) : 0;
    if (version > PRE_VISION_CEILING) continue; // quarantined, left intact
    const { disableVision: _dropped, ...knownFields } = map[key];
    map[key] = knownFields;
    rewrote = true;
  }
  store.set(STORAGE_KEY, JSON.stringify(map));

  assert.equal(rewrote, false, "the old client was able to rewrite the record");
  assert.equal(resolveInitialConfig("some/vision-gguf", "Q4_K_M").config.disableVision, true);
});

test("a config with no new field is still stamped low enough for an old client", () => {
  // The other half of the rule: only a record carrying a NEWER field is put out
  // of an older client's reach. A plain kv-cache change must stay readable.
  installLocalStorage();

  savePerModelConfig("some/gguf", "Q4_K_M", config({ kvCacheDtype: "q8_0" }));

  assert.ok((storedRecord().version as number) <= PRE_VISION_CEILING);
});

test("the toggle still round-trips within this build", () => {
  installLocalStorage();
  savePerModelConfig("some/vision-gguf", "Q4_K_M", config({ disableVision: true }));

  assert.equal(resolveInitialConfig("some/vision-gguf", "Q4_K_M").config.disableVision, true);
});
