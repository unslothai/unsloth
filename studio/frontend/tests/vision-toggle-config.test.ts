// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The per-model config store has no migration step, so a new field must survive old and new builds.

import assert from "node:assert/strict";
import test from "node:test";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  DEFAULT_PER_MODEL_CONFIG,
  normalizePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { loadedConfigSignature } = await import(
  "../src/features/model-picker/model-config/config-signature.ts"
);

const STORAGE_KEY = "unsloth_model_configs";
// Ceiling of the last build before the toggle; records at or below it can be erased by it.
const PRE_VISION_CEILING = 3;

function config(overrides: Partial<PerModelConfig> = {}): PerModelConfig {
  return { ...DEFAULT_PER_MODEL_CONFIG, ...overrides };
}

function storedRecord(): Record<string, unknown> {
  const raw = store.get(STORAGE_KEY);
  assert.ok(raw, "nothing was persisted");
  const map = JSON.parse(raw) as Record<string, Record<string, unknown>>;
  const keys = Object.keys(map);
  assert.equal(keys.length, 1, `expected one record, got ${keys.length}`);
  return map[keys[0]];
}

test("vision is on by default", () => {
  assert.equal(DEFAULT_PER_MODEL_CONFIG.disableVision, false);
});

test("a blob written before the toggle existed still loads", () => {
  // Genuinely absent, as an older build's JSON has it.
  const { disableVision: _omitted, ...legacy } = DEFAULT_PER_MODEL_CONFIG;
  assert.equal(
    normalizePerModelConfig(legacy as Record<string, unknown>).disableVision,
    false,
  );
  for (const bad of ["true", 1, null, {}]) {
    assert.equal(
      normalizePerModelConfig({
        ...DEFAULT_PER_MODEL_CONFIG,
        disableVision: bad,
      }).disableVision,
      false,
      `for ${JSON.stringify(bad)}`,
    );
  }
});

test("the toggle round-trips through save and load", () => {
  store.clear();
  savePerModelConfig(
    "some/vision-gguf",
    "Q4_K_M",
    config({ disableVision: true }),
  );
  assert.equal(
    resolveInitialConfig("some/vision-gguf", "Q4_K_M").config.disableVision,
    true,
  );
});

test("an older build cannot reach a vision-off record and erase it", () => {
  store.clear();
  savePerModelConfig(
    "some/vision-gguf",
    "Q4_K_M",
    config({ disableVision: true }),
  );
  const record = storedRecord();
  assert.equal(record.disableVision, true);
  assert.ok(
    (record.version as number) > PRE_VISION_CEILING,
    `stamped v${record.version}; a pre-feature build (ceiling ${PRE_VISION_CEILING}) ` +
      "would accept this record, drop disableVision, and erase it on its next save",
  );

  const map = JSON.parse(store.get(STORAGE_KEY) as string) as Record<
    string,
    Record<string, unknown>
  >;
  let rewrote = false;
  for (const key of Object.keys(map)) {
    const version =
      typeof map[key].version === "number" ? (map[key].version as number) : 0;
    if (version > PRE_VISION_CEILING) continue;
    const { disableVision: _dropped, ...known } = map[key];
    map[key] = known;
    rewrote = true;
  }
  store.set(STORAGE_KEY, JSON.stringify(map));
  assert.equal(rewrote, false, "the old build was able to rewrite the record");
  assert.equal(
    resolveInitialConfig("some/vision-gguf", "Q4_K_M").config.disableVision,
    true,
  );
});

test("a record with no new field stays inside an older build's reach", () => {
  // Only records carrying a NEWER field are stamped out of reach, not the whole store.
  store.clear();
  savePerModelConfig("some/gguf", "Q4_K_M", config({ kvCacheDtype: "q8_0" }));
  assert.ok((storedRecord().version as number) <= PRE_VISION_CEILING);
});

test("flipping the toggle changes the signature, so a reload is not deduped away", () => {
  assert.notEqual(
    loadedConfigSignature(config({ disableVision: false })),
    loadedConfigSignature(config({ disableVision: true })),
  );
});
