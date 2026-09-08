// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Compatibility of the stored MLX context pin, which moved from `maxSeqLength` into
// `customContextLength` without a schema bump: `toStoredConfig` stamps both shapes
// version 1, so neither direction can tell them apart.
//
// Backwards works: `savedContextPin` reads either field.
//
// Forwards does not, and the last section demonstrates it rather than asserting it away.
// An old client's model-config page resolves a non-GGUF context from `config.maxSeqLength`
// alone (model-config-page.tsx on main, ~2711), so a new record reads as 4096 and as "at
// default settings". Only that page: main's `resolveLoadMaxSeqLength` and its compare-pane
// rule both read `customContextLength` first, so auto-load and compare still honour it.

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
  CONTEXT_LENGTH_MIN,
  DEFAULT_MAX_SEQ_LENGTH,
  DEFAULT_PER_MODEL_CONFIG,
  MAX_SEQ_LENGTH_MIN,
  contextPinPatch,
  isDefaultConfig,
  normalizeMaxSeqLength,
  normalizePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
  savedContextPin,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { resolveLoadMaxSeqLength } = await import(
  "../src/features/chat/presets/preset-policy.ts"
);

const STORAGE_KEY = "unsloth_model_configs";
const MODEL = "mlx-community/Qwen3-8B-4bit";
/** The window an MLX model of this size reports; only used as the "native" input. */
const NATIVE = 262144;


/** The storage key `savePerModelConfig` uses for MODEL, discovered rather than guessed. */
function storageKeyForModel(): string {
  store.clear();
  assert.ok(
    savePerModelConfig(MODEL, null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      customContextLength: 4096,
    }),
  );
  const map = JSON.parse(store.get(STORAGE_KEY) ?? "{}") as Record<
    string,
    unknown
  >;
  const keys = Object.keys(map);
  assert.equal(
    keys.length,
    1,
    `expected one stored key, got ${JSON.stringify(keys)}`,
  );
  return keys[0];
}

const MODEL_KEY = storageKeyForModel();

/** Put `raw` in storage verbatim, bypassing normalization, and read it back the way the app does. */
function stage(raw: Record<string, unknown>) {
  store.clear();
  // The legacy migration runs on the first read and would merge an unrelated blob in.
  store.set("unsloth_model_configs_migrated", "1");
  store.set(STORAGE_KEY, JSON.stringify({ [MODEL_KEY]: raw }));
  return resolveInitialConfig(MODEL, null);
}

/** The version `toStoredConfig` stamps on a record carrying this pin and nothing else. */
function stampedVersion(patch: Record<string, unknown>): number {
  store.clear();
  assert.ok(
    savePerModelConfig(MODEL, null, { ...DEFAULT_PER_MODEL_CONFIG, ...patch }),
  );
  const map = JSON.parse(store.get(STORAGE_KEY) ?? "{}") as Record<
    string,
    { version: number }
  >;
  return map[MODEL_KEY].version;
}


type Row = {
  name: string;
  raw: Record<string, unknown>;
  /** After `normalize`, which is what every reader outside a raw preset sees. */
  normalizedPin: number | null;
  /** `savedContextPin` applied to the RAW record, which preset code does reach. */
  rawPin: number | null;
  isDefault: boolean;
  /** `readable: false` means the future-schema guard refuses the record entirely. */
  readable: boolean;
  /** What /load is asked for on each backend, with the control untouched. */
  mlxRequest: number;
  transformersRequest: number;
  note?: string;
};

const ROWS: Row[] = [
  {
    name: "old record: pin only in maxSeqLength",
    raw: { version: 1, maxSeqLength: 8192 },
    normalizedPin: 8192,
    rawPin: 8192,
    isDefault: false,
    readable: true,
    mlxRequest: 8192,
    transformersRequest: 8192,
    note: "backwards compatibility: the pre-move field still pins, on either backend.",
  },
  {
    name: "new record: pin only in customContextLength",
    raw: { version: 1, customContextLength: 32768 },
    normalizedPin: 32768,
    rawPin: 32768,
    isDefault: false,
    readable: true,
    mlxRequest: 32768,
    transformersRequest: 32768,
  },
  {
    name: "both fields set",
    raw: { version: 1, customContextLength: 32768, maxSeqLength: 8192 },
    normalizedPin: 32768,
    rawPin: 32768,
    isDefault: false,
    readable: true,
    mlxRequest: 32768,
    transformersRequest: 32768,
    note:
      "customContextLength wins in both the pin reader and the load resolver, so the " +
      "two agree. No edit can author this state -- contextPinPatch clears the other " +
      "field -- but a hand-edited or merged store can hold it.",
  },
  {
    name: "neither field set",
    raw: { version: 1 },
    normalizedPin: null,
    rawPin: null,
    isDefault: true,
    readable: true,
    // MLX sizes its own window (0 asks for nothing); transformers gets the app default.
    mlxRequest: 0,
    transformersRequest: DEFAULT_MAX_SEQ_LENGTH,
  },
  {
    name: "zero in both fields",
    raw: { version: 1, customContextLength: 0, maxSeqLength: 0 },
    normalizedPin: null,
    // 0 is neither null nor undefined, so `??` keeps it: an unnormalized record yields a
    // falsy 0 "pin". Reachable because preset code calls savedContextPin on raw shapes.
    rawPin: 0,
    isDefault: true,
    readable: true,
    mlxRequest: 0,
    transformersRequest: DEFAULT_MAX_SEQ_LENGTH,
  },
  {
    name: "negative in both fields",
    raw: { version: 1, customContextLength: -1, maxSeqLength: -4096 },
    normalizedPin: null,
    rawPin: -1,
    isDefault: true,
    readable: true,
    mlxRequest: 0,
    transformersRequest: DEFAULT_MAX_SEQ_LENGTH,
  },
  {
    name: "non-integer in both fields",
    raw: { version: 1, customContextLength: 8192.7, maxSeqLength: 8192.7 },
    // Floors vs snaps to 128: both land on 8192 by different arithmetic.
    normalizedPin: 8192,
    rawPin: 8192.7,
    isDefault: false,
    readable: true,
    mlxRequest: 8192,
    transformersRequest: 8192,
  },
  {
    name: "version 4 (below the current schema version)",
    raw: { version: 4, customContextLength: 32768 },
    normalizedPin: 32768,
    rawPin: 32768,
    isDefault: false,
    readable: true,
    mlxRequest: 32768,
    transformersRequest: 32768,
    note:
      "The task called v4 the 'future' record. It is not: STORAGE_SCHEMA_VERSION is 5 " +
      "in this tree and in main, not 3, so v4 is a v4-client record and reads normally.",
  },
  {
    name: "version 6 (genuinely future)",
    raw: { version: 6, customContextLength: 32768 },
    normalizedPin: null,
    rawPin: 32768,
    isDefault: true,
    // Above STORAGE_SCHEMA_VERSION: loadPerModelConfig refuses it, so the pin is unread
    // and the record is safe from being overwritten.
    readable: false,
    mlxRequest: 0,
    transformersRequest: DEFAULT_MAX_SEQ_LENGTH,
  },
];

/** The n_ctx `/load` is asked for, through the shipped resolver, control untouched. */
function loadRequest(
  config: { customContextLength: number | null; maxSeqLength: number | null },
  isMlx: boolean,
): number {
  return resolveLoadMaxSeqLength({
    modelId: MODEL,
    ggufVariant: null,
    isGguf: false,
    customContextLength: config.customContextLength,
    loadedContextLength: null,
    currentCheckpoint: "",
    activeGgufVariant: null,
    isMlx,
    pinnedMaxSeqLength: normalizeMaxSeqLength(config.maxSeqLength),
    defaultMaxSeqLength: DEFAULT_MAX_SEQ_LENGTH,
    presetSource: "builtin-default",
  });
}

for (const row of ROWS) {
  test(`record matrix — ${row.name}`, () => {
    const { config, remembered } = stage(row.raw);
    assert.equal(remembered, row.readable, "remembered");
    assert.equal(
      savedContextPin(config),
      row.normalizedPin,
      "savedContextPin(normalized)",
    );
    assert.equal(
      savedContextPin(
        row.raw as { customContextLength?: number; maxSeqLength?: number },
      ),
      row.rawPin,
      "savedContextPin(raw)",
    );
    assert.equal(isDefaultConfig(config), row.isDefault, "isDefaultConfig");
    assert.equal(loadRequest(config, true), row.mlxRequest, "MLX /load n_ctx");
    assert.equal(
      loadRequest(config, false),
      row.transformersRequest,
      "transformers /load n_ctx",
    );
  });
}


test("contextPinPatch writes the pin in exactly one field and clears the other", () => {
  assert.deepEqual(contextPinPatch(32768, true), {
    customContextLength: 32768,
    maxSeqLength: null,
  });
  assert.deepEqual(contextPinPatch(32768, false), {
    customContextLength: null,
    maxSeqLength: 32768,
  });
});

test("contextPinPatch bounds without snapping, and never writes a blank", () => {
  // Bounded to what /load accepts, not snapped to the control's 128 step.
  assert.equal(contextPinPatch(8193, true).customContextLength, 8193);
  assert.equal(contextPinPatch(8192.7, true).customContextLength, 8192);
  assert.equal(
    contextPinPatch(1, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  // 0 and negatives become the floor, not a cleared pin: "unpin" must not route here.
  assert.equal(
    contextPinPatch(0, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  assert.equal(contextPinPatch(-1, false).maxSeqLength, MAX_SEQ_LENGTH_MIN);
  assert.equal(
    contextPinPatch(Number.NaN, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  assert.equal(
    contextPinPatch(Number.POSITIVE_INFINITY, true).customContextLength,
    MAX_SEQ_LENGTH_MIN,
  );
  assert.equal(MAX_SEQ_LENGTH_MIN, CONTEXT_LENGTH_MIN);
});

test("a patched pin round-trips through storage on both backends", () => {
  for (const isMlx of [true, false]) {
    store.clear();
    const patched = {
      ...DEFAULT_PER_MODEL_CONFIG,
      ...contextPinPatch(32768, isMlx),
    };
    assert.equal(isDefaultConfig(normalizePerModelConfig(patched)), false);
    assert.ok(savePerModelConfig(MODEL, null, patched));
    const { config, remembered } = resolveInitialConfig(MODEL, null);
    assert.ok(remembered);
    assert.equal(savedContextPin(config), 32768, `isMlx=${isMlx}`);
    assert.equal(loadRequest(config, isMlx), 32768, `isMlx=${isMlx}`);
    // And the cross-backend read: the same record on a host serving the OTHER backend.
    assert.equal(
      loadRequest(config, !isMlx),
      32768,
      `cross-read isMlx=${!isMlx}`,
    );
  }
});


test("both pin shapes are stamped version 1, so neither is distinguishable by version", () => {
  assert.equal(stampedVersion({ customContextLength: 32768 }), 1);
  assert.equal(stampedVersion({ maxSeqLength: 32768 }), 1);
  // The old client's only forwards guard is `version > 5`, and v1 invites any client
  // back to v1 to rewrite the record.
  assert.equal(
    stage({ version: 1, customContextLength: 32768 }).remembered,
    true,
  );
  assert.equal(
    stage({ version: 5, customContextLength: 32768 }).remembered,
    true,
  );
  assert.equal(
    stage({ version: 6, customContextLength: 32768 }).remembered,
    false,
  );
});


/**
 * The old model-config page's read rule for a NON-GGUF target, transcribed from
 * `studio/frontend/src/features/model-picker/components/model-config-page.tsx` on main:
 *
 *   ~2699  const contextAtDefault = !target.isGguf || config.customContextLength == null;
 *   ~2700  const atDefault = contextAtDefault && perModelConfigsEqual(
 *              { ...config, customContextLength: null }, DEFAULT_PER_MODEL_CONFIG);
 *   ~2711  const maxSeqLengthValue =
 *              normalizeMaxSeqLength(config.maxSeqLength) ??
 *              clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength);
 *   ~3126  const effectiveLoadConfig = target.isGguf
 *              ? effectiveRuntimeConfig
 *              : { ...effectiveRuntimeConfig, maxSeqLength: effectiveMaxSeqLengthValue };
 *
 * `customContextLength` appears in none of the three that decide the number, and the one
 * place it does appear deletes it before comparing.
 */
function oldClientConfigPage(
  config: PerModelConfig,
  native: number,
): { shownContext: number; loadRequests: number; showsAsDefault: boolean } {
  const clamp = (value: number, max: number) => Math.min(value, max);
  const shown =
    normalizeMaxSeqLength(config.maxSeqLength) ??
    clamp(DEFAULT_MAX_SEQ_LENGTH, native);
  // contextAtDefault is unconditional for non-GGUF and nulls customContextLength out.
  const showsAsDefault = isDefaultConfig({
    ...config,
    customContextLength: null,
  });
  return { shownContext: shown, loadRequests: shown, showsAsDefault };
}

test("FORWARDS-COMPAT GAP: an old client's config page silently drops a new record's pin", () => {
  store.clear();
  const patched = {
    ...DEFAULT_PER_MODEL_CONFIG,
    ...contextPinPatch(32768, true),
  };
  assert.ok(savePerModelConfig(MODEL, null, patched));
  const stored = JSON.parse(store.get(STORAGE_KEY) ?? "{}") as Record<
    string,
    Record<string, unknown>
  >;
  const record = stored[MODEL_KEY];

  assert.equal(record.customContextLength, 32768);
  assert.equal(record.maxSeqLength, null);
  assert.equal(record.version, 1);

  const { config } = resolveInitialConfig(MODEL, null);

  assert.equal(savedContextPin(config), 32768);
  assert.equal(loadRequest(config, true), 32768);

  const old = oldClientConfigPage(config, NATIVE);
  assert.equal(old.shownContext, DEFAULT_MAX_SEQ_LENGTH);
  assert.equal(old.loadRequests, DEFAULT_MAX_SEQ_LENGTH);
  assert.equal(old.showsAsDefault, true);

  // Nothing in the record could warn it: the stamp is 1 and its guard is `version > 5`.
  assert.ok((record.version as number) <= 5);

  const table = [
    "| reader                                   | context requested | shows as |",
    "| ---------------------------------------- | ----------------- | -------- |",
    `| new code (savedContextPin)               | ${loadRequest(config, true)}             | pinned   |`,
    `| old client, model-config page (non-GGUF) | ${old.loadRequests}              | default  |`,
  ].join("\n");
  assert.ok(table.includes("32768") && table.includes("4096"));
  console.log(
    [
      "",
      "Record on disk (written by the new code):",
      `  ${JSON.stringify({
        version: record.version,
        customContextLength: record.customContextLength,
        maxSeqLength: record.maxSeqLength,
      })}`,
      "",
      table,
      "",
    ].join("\n"),
  );
});

test("the gap is the config page's, not resolveLoadMaxSeqLength's", () => {
  // main's resolveLoadMaxSeqLength reads customContextLength first, unconditionally, for
  // every backend -- the signature changed in this PR but that first branch did not. So
  // the auto-load path in chat-adapter still honours a new record on an old client, and
  // "the old client loses the pin" would be too strong a claim to make unqualified.
  store.clear();
  assert.ok(
    savePerModelConfig(MODEL, null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      ...contextPinPatch(32768, true),
    }),
  );
  const { config } = resolveInitialConfig(MODEL, null);
  // The old signature, with the old argument names, on the same record.
  const oldResolverAnswer =
    config.customContextLength != null
      ? config.customContextLength
      : (normalizeMaxSeqLength(config.maxSeqLength) ?? DEFAULT_MAX_SEQ_LENGTH);
  assert.equal(oldResolverAnswer, 32768);
  // And main's compare-pane rule (shared-composer.tsx ~1404), likewise.
  const oldComparePaneAnswer =
    config.customContextLength ??
    normalizeMaxSeqLength(config.maxSeqLength) ??
    DEFAULT_MAX_SEQ_LENGTH;
  assert.equal(oldComparePaneAnswer, 32768);
});

test("BACKWARDS COMPAT: an old record still pins under the new code, on either backend", () => {
  const { config } = stage({ version: 1, maxSeqLength: 8192 });
  assert.equal(savedContextPin(config), 8192);
  assert.equal(loadRequest(config, true), 8192);
  assert.equal(loadRequest(config, false), 8192);
  // And it is not reported as default, so it survives savePerModelConfig's delete-if-default.
  assert.equal(isDefaultConfig(config), false);
});
