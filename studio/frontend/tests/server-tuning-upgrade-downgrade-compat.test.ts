// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No user setting may be lost on upgrade/downgrade; hidden is acceptable, lost is not.
// The version stamp is a downgrade lock: toStoredConfig must stamp the oldest capable version.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";
import type { StorageFake } from "./helpers/kit.ts";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store, storage } = installLocalStorageFake();

const {
  DEFAULT_PER_MODEL_CONFIG,
  deletePerModelConfig,
  isDefaultConfig,
  listPerModelConfigs,
  normalizePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const {
  fromApiOverride,
  panelOverrideRow,
  resolveStoredOverride,
  toApiOverride,
} = await import("../src/features/model-picker/api/model-overrides.ts");
const { backfillModelOverrides } = await import(
  "../src/features/model-picker/api/migrate-model-overrides.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const STORAGE_KEY = "unsloth_model_configs";
const BACKFILL_FLAG = "unsloth_model_overrides_backfilled_v3";
const MODEL = "unsloth/Repo-GGUF";
const VARIANT = "Q4_K_M";

// Ceiling of the last build before the tuning group; records at or below it are erasable there.
const PRE_TUNING_CEILING = 4;

// Falsy values are deliberate: 0 checkpoints and 0/-1 cache are real decisions.
const TUNING_ONLY_PATCHES: Partial<PerModelConfig>[] = [
  { loadMode: "mmap" },
  { ctxCheckpoints: 0 },
  { ctxCheckpoints: 64 },
  { cacheRam: 0 },
  { cacheRam: -1 },
  // The dtype requires a separate-drafter mode, so the mode travels with it.
  { specDraftCacheDtype: "q8_0", speculativeType: "dspark" },
];

function config(overrides: Partial<PerModelConfig> = {}): PerModelConfig {
  return { ...DEFAULT_PER_MODEL_CONFIG, ...overrides };
}

function readMap(): Record<string, Record<string, unknown>> {
  return JSON.parse(store.get(STORAGE_KEY) ?? "{}");
}

function writeMap(map: Record<string, unknown>): void {
  store.set(STORAGE_KEY, JSON.stringify(map));
}

function onlyEntry(): Record<string, unknown> {
  const entries = Object.values(readMap());
  assert.equal(entries.length, 1, `expected one record, got ${entries.length}`);
  return entries[0];
}

function load(): PerModelConfig | null {
  const initial = resolveInitialConfig(MODEL, VARIANT);
  return initial.remembered ? initial.config : null;
}

test("a v0 record with no version key at all still loads the tuning it carried", () => {
  store.clear();
  writeMap({
    [`${MODEL}::${VARIANT}`]: {
      loadMode: "mmap",
      ctxCheckpoints: 0,
      cacheRam: -1,
      speculativeType: "dspark",
      specDraftCacheDtype: "q8_0",
    },
  });

  const loaded = load();
  assert.ok(loaded, "a v0 record must remain readable");
  assert.equal(loaded.loadMode, "mmap");
  assert.equal(loaded.ctxCheckpoints, 0);
  assert.equal(loaded.cacheRam, -1);
  assert.equal(loaded.specDraftCacheDtype, "q8_0");
});

test("a v4 record loads unchanged and is re-stamped v4, not silently upgraded", () => {
  store.clear();
  writeMap({
    [`${MODEL}::${VARIANT}`]: {
      version: PRE_TUNING_CEILING,
      customContextLength: 4096,
      disableVision: true,
    },
  });
  const loaded = load();
  assert.ok(loaded);
  assert.equal(loaded.customContextLength, 4096);
  assert.equal(loaded.disableVision, true);
  // Missing fields read as unset, not a bogus default that would be pinned onto every load.
  assert.equal(loaded.loadMode, null);
  assert.equal(loaded.ctxCheckpoints, null);
  assert.equal(loaded.cacheRam, null);

  // Over-stamping on re-save would lock out a build that can still read the record.
  assert.ok(savePerModelConfig(MODEL, VARIANT, loaded));
  assert.equal(onlyEntry().version, PRE_TUNING_CEILING);
});

test("only a record that actually carries tuning is stamped v5", () => {
  // Annotated, not `as const`: a readonly tuple will not fit Partial<PerModelConfig>.
  const cases: [Partial<PerModelConfig>, number][] = [
    [{ kvCacheDtype: "q8_0" }, 1],
    [{ nBatch: 4096 }, 2],
    [{ llamaExtraArgs: ["--numa", "distribute"] }, 3],
    [{ disableVision: true }, 4],
    [{ loadMode: "mmap" }, 5],
    [{ ctxCheckpoints: 0 }, 5],
    [{ cacheRam: -1 }, 5],
    [{ specDraftCacheDtype: "q8_0", speculativeType: "dspark" }, 5],
  ];
  for (const [patch, expected] of cases) {
    store.clear();
    assert.ok(savePerModelConfig(MODEL, VARIANT, config(patch)));
    assert.equal(
      onlyEntry().version,
      expected,
      `version for ${JSON.stringify(patch)}`,
    );
  }
});

test("a record with no tuning stays inside a pre-v5 build's reach", () => {
  // Stamping every record v5 would quarantine the store from a downgraded build.
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ nParallel: 8 })));
  assert.ok((onlyEntry().version as number) <= PRE_TUNING_CEILING);
});

test("a v5 record is out of a pre-v5 build's reach in the first place", () => {
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ cacheRam: -1 })));

  const map = readMap();
  let rewrote = false;
  for (const key of Object.keys(map)) {
    const version =
      typeof map[key].version === "number" ? (map[key].version as number) : 0;
    if (version > PRE_TUNING_CEILING) {
      continue;
    }
    const {
      loadMode: _loadMode,
      specDraftCacheDtype: _specDraftCacheDtype,
      ctxCheckpoints: _ctxCheckpoints,
      cacheRam: _cacheRam,
      ...known
    } = map[key];
    map[key] = known;
    rewrote = true;
  }
  writeMap(map);

  assert.equal(rewrote, false, "the pre-v5 build was able to rewrite the record");
  assert.equal(load()?.cacheRam, -1);
});

test("every entry point declines a record stamped beyond this build", () => {
  // Any missing guard lets an older client silently destroy a newer record.
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ loadMode: "mmap" })));
  const map = readMap();
  map[Object.keys(map)[0]].version = 99;
  writeMap(map);
  const untouched = store.get(STORAGE_KEY);

  assert.equal(load(), null, "load hides a future record");
  assert.equal(
    savePerModelConfig(MODEL, VARIANT, config({ nParallel: 1 })),
    false,
    "save must refuse rather than overwrite a future record",
  );
  assert.equal(
    deletePerModelConfig(MODEL, VARIANT),
    false,
    "delete must refuse a future record",
  );
  assert.deepEqual(
    listPerModelConfigs(),
    [],
    "a future record must not be reported to the backfill either",
  );
  assert.equal(store.get(STORAGE_KEY), untouched, "the stored bytes must be untouched");

  const restored = readMap();
  restored[Object.keys(restored)[0]].version = 5;
  writeMap(restored);
  assert.equal(load()?.loadMode, "mmap");
});

test("a tuning-only config is stored rather than deleted on the way in", () => {
  for (const patch of TUNING_ONLY_PATCHES) {
    store.clear();
    const normalized = normalizePerModelConfig(config(patch));
    assert.equal(isDefaultConfig(normalized), false, JSON.stringify(patch));
    assert.ok(savePerModelConfig(MODEL, VARIANT, normalized));
    assert.equal(
      resolveInitialConfig(MODEL, VARIANT).remembered,
      true,
      `not remembered for ${JSON.stringify(patch)}`,
    );
  }
});

test("the one-time backfill now uploads a tuning-only config", async () => {
  // Counting the four makes the backfill mirror configs it used to skip; deliberate, add-only.
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ cacheRam: -1 })));

  const puts: Record<string, unknown>[] = [];
  setAuthFetchHandler((_input, init) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)));
      return new Response(JSON.stringify({ overrides: {} }), { status: 200 });
    }
    return new Response(
      JSON.stringify({
        overrides: {
          [`${MODEL.toLowerCase()}:${VARIANT.toLowerCase()}`]: {
            // biome-ignore lint/style/useNamingConvention: API schema
            max_seq_length: 4096,
          },
        },
      }),
      { status: 200 },
    );
  });
  try {
    await backfillModelOverrides();
  } finally {
    setAuthFetchHandler(null);
  }

  assert.equal(puts.length, 1, "the tuning-only config must be offered to the server");
  assert.equal(puts[0].cache_ram, -1);
  // Fill, never replace: the server copy is the newer authority.
  assert.equal(puts[0].fill_absent_fields, true);
  assert.equal(puts[0].remove, false);
  assert.equal(store.get(BACKFILL_FLAG), "1", "a completed pass must not run again");
});

test("the backfill offers an Ollama tag's settings even after the v1 pass ran", async () => {
  store.clear();
  store.set("unsloth_model_overrides_backfilled_v1", "1");
  const ref = "ollama-manifest:%2Fh%2F.ollama%2Fmanifests%2Fllama3%2Flatest";
  assert.ok(savePerModelConfig(ref, null, config({ cacheRam: -1 })));

  const puts: unknown[] = [];
  setAuthFetchHandler((_input, init) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)).model_id);
    }
    return new Response(JSON.stringify({ overrides: {} }), { status: 200 });
  });
  try {
    await backfillModelOverrides();
  } finally {
    setAuthFetchHandler(null);
  }
  assert.deepEqual(puts, [ref.toLowerCase()]);
});

test("the backfill offers non-GGUF weights, keyed by repo id, after the v2 pass ran", async () => {
  // Uploaded under the snapshot path, the bare row would outrank the repo's and survive Forget.
  store.clear();
  store.set("unsloth_model_overrides_backfilled_v2", "1");
  const folder = "/Users/u/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
  const snapshot = "/hf/models--mlx-community--Model-4bit/snapshots/abc";
  const template = { chatTemplateOverride: "{{ messages }}" };
  assert.ok(savePerModelConfig(folder, null, config(template)));
  assert.ok(savePerModelConfig(snapshot, null, config({ mlxKvQuant: "8" })));
  const declined = "/hf/models--org--Newer/snapshots/def";
  assert.ok(savePerModelConfig("org/Newer", null, config({ mlxKvQuant: "4" })));
  const map = readMap();
  const newer = Object.keys(map).find((key) => key.includes("org/newer"));
  assert.ok(newer);
  map[newer].version = 99;
  writeMap(map);
  assert.ok(savePerModelConfig(declined, null, config({ mlxKvQuant: "4" })));
  const link = "/home/u/.ollama/.studio_links/ab12/model-latest.gguf";
  assert.ok(savePerModelConfig(link, null, config(template)));
  assert.ok(savePerModelConfig("Model-Q4_K_M.gguf", null, config(template)));

  const puts: unknown[] = [];
  setAuthFetchHandler((_input, init) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)).model_id);
    }
    return new Response(JSON.stringify({ overrides: {} }), { status: 200 });
  });
  try {
    await backfillModelOverrides();
  } finally {
    setAuthFetchHandler(null);
  }
  assert.deepEqual(
    new Set(puts),
    new Set([folder, "mlx-community/model-4bit"]),
  );
  assert.equal(resolveInitialConfig(declined, null).config.mlxKvQuant, "4");
  assert.equal(store.get(BACKFILL_FLAG), "1");
});

test("an all-default config is still filtered out of the backfill", async () => {
  // Otherwise every model the user ever opened is mirrored on first launch.
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config()));
  assert.deepEqual(readMap(), {}, "a default config must not be written");

  setAuthFetchHandler(() => {
    throw new Error("the backfill must not reach the network with nothing to send");
  });
  try {
    await backfillModelOverrides();
  } finally {
    setAuthFetchHandler(null);
  }
  assert.equal(store.get(BACKFILL_FLAG), "1");
});

test("the eviction loop terminates when the budget needs a future record", () => {
  // Future-schema entries are never evicted, so the loop must give up instead of hanging.
  for (const overBudget of [
    () => {
      const map: Record<string, unknown> = {};
      for (let index = 0; index < 505; index += 1) {
        map[`unsloth/Model-${index}-GGUF::${VARIANT}`] = {
          version: 99,
          cacheRam: index,
        };
      }
      return map;
    },
    () => {
      const map: Record<string, unknown> = {};
      for (let index = 0; index < 20; index += 1) {
        map[`unsloth/Model-${index}-GGUF::${VARIANT}`] = {
          version: 99,
          chatTemplateOverride: "x".repeat(60_000),
        };
      }
      return map;
    },
  ]) {
    store.clear();
    const map = overBudget();
    writeMap(map);
    const untouched = store.get(STORAGE_KEY);

    // Reaching here is the termination assertion; node:test does not report a hang.
    assert.equal(
      savePerModelConfig("unsloth/New-GGUF", VARIANT, config({ cacheRam: 1 })),
      false,
      "a save that cannot fit must fail rather than evict a future record",
    );
    assert.equal(
      store.get(STORAGE_KEY),
      untouched,
      "nothing may be written when the budget could not be met",
    );
  }
});

test("eviction takes the readable records and leaves the future ones", () => {
  store.clear();
  const map: Record<string, unknown> = {};
  map["unsloth/Old-A-GGUF::Q4_K_M"] = { version: 1, nParallel: 2 };
  map["unsloth/Future-GGUF::Q4_K_M"] = { version: 99, cacheRam: 7 };
  map["unsloth/Old-B-GGUF::Q4_K_M"] = { version: 1, nParallel: 3 };
  for (let index = 0; index < 499; index += 1) {
    map[`unsloth/Filler-${index}-GGUF::${VARIANT}`] = { version: 1, nParallel: 1 };
  }
  writeMap(map);

  const evicted: { modelId: string; ggufVariant: string | null }[] = [];
  assert.ok(
    savePerModelConfig("unsloth/New-GGUF", VARIANT, config({ cacheRam: 1 }), evicted),
  );

  const after = readMap();
  assert.equal(Object.keys(after).length, 500);
  assert.deepEqual(after["unsloth/Future-GGUF::Q4_K_M"], {
    version: 99,
    cacheRam: 7,
  });
  // Eviction is silent, so the list is needed to forget the evicted model's server override.
  assert.deepEqual(evicted, [
    { modelId: "unsloth/Old-A-GGUF", ggufVariant: "Q4_K_M" },
    { modelId: "unsloth/Old-B-GGUF", ggufVariant: "Q4_K_M" },
    { modelId: "unsloth/Filler-0-GGUF", ggufVariant: "Q4_K_M" },
  ]);
});

test("a non-GGUF panel reads a row without its llama-server arguments", () => {
  const row = {
    // biome-ignore lint/style/useNamingConvention: API schema
    chat_template_override: "{{ messages }}",
    // biome-ignore lint/style/useNamingConvention: API schema
    llama_extra_args: ["--no-mmap"],
  };
  assert.deepEqual(panelOverrideRow(row, true), row);
  assert.deepEqual(panelOverrideRow(row, false), {
    // biome-ignore lint/style/useNamingConvention: API schema
    chat_template_override: "{{ messages }}",
  });
  // biome-ignore lint/style/useNamingConvention: API schema
  assert.equal(panelOverrideRow({ llama_extra_args: [] }, false), null);
});

test("a row that carries no tuning leaves this browser's tuning standing", () => {
  // The mirror is lossy both ways, so a missing server field is a gap, not a choice.
  const local = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    load_mode: "mmap",
    // biome-ignore lint/style/useNamingConvention: API schema
    ctx_checkpoints: 0,
    // biome-ignore lint/style/useNamingConvention: API schema
    cache_ram: -1,
  });
  const hydrated = fromApiOverride(
    // biome-ignore lint/style/useNamingConvention: API schema
    { custom_context_length: 32768 },
    local,
  );

  assert.equal(hydrated.customContextLength, 32768);
  assert.equal(hydrated.loadMode, "mmap");
  assert.equal(hydrated.ctxCheckpoints, 0);
  assert.equal(hydrated.cacheRam, -1);
});

test("a row's tuning outranks this browser's for the fields it does carry", () => {
  const local = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    load_mode: "mmap",
    // biome-ignore lint/style/useNamingConvention: API schema
    cache_ram: 4096,
  });
  const hydrated = fromApiOverride(
    {
      // biome-ignore lint/style/useNamingConvention: API schema
      load_mode: "mlock",
      // biome-ignore lint/style/useNamingConvention: API schema
      cache_ram: 0,
    },
    local,
  );

  assert.equal(hydrated.loadMode, "mlock");
  // 0 is a value (cache off), so it must beat a local 4096 rather than read as absent.
  assert.equal(hydrated.cacheRam, 0);
});

test("a server value this build refuses falls to the app default, not the local one", () => {
  // normalizeV1 clamps after the merge, so a refused server value cannot fall back to local.
  const local = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    speculative_type: "dspark",
    // biome-ignore lint/style/useNamingConvention: API schema
    spec_draft_cache_type: "q8_0",
  });
  assert.equal(local.specDraftCacheDtype, "q8_0");

  // biome-ignore lint/style/useNamingConvention: API schema
  const hydrated = fromApiOverride({ speculative_type: "ngram" }, local);
  assert.equal(hydrated.speculativeType, "ngram");
  assert.equal(
    hydrated.specDraftCacheDtype,
    null,
    "the dtype belongs to a draft context this mode never creates",
  );

  const pinned = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    load_mode: "mmap",
  });
  // biome-ignore lint/style/useNamingConvention: API schema
  const refused = fromApiOverride({ load_mode: "swap" }, pinned);
  assert.equal(refused.loadMode, null);
});

test("an out-of-range server value clamps rather than falling through", () => {
  // Numeric knobs are clamped, not refused, so the row still wins.
  const local = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    ctx_checkpoints: 64,
  });
  // biome-ignore lint/style/useNamingConvention: API schema
  const hydrated = fromApiOverride({ ctx_checkpoints: 1_000_000 }, local);
  assert.equal(hydrated.ctxCheckpoints, 256);
});

test("an empty server argument list clears a local list rather than being ignored", () => {
  // [] is a tombstone blocking the server fallback; normalize collapses it to null, so reinstall.
  const local = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    llama_extra_args: ["--numa", "distribute"],
  });
  assert.deepEqual(local.llamaExtraArgs, ["--numa", "distribute"]);

  // biome-ignore lint/style/useNamingConvention: API schema
  const hydrated = fromApiOverride({ llama_extra_args: [] }, local);
  assert.deepEqual(hydrated.llamaExtraArgs, []);
  // Distinct from "never read", which must stay omitted so the route keeps server flags.
  assert.notEqual(hydrated.llamaExtraArgs, undefined);
});

test("an empty server GPU list does not clear a local pin", () => {
  // Deliberately unlike the tombstone: a row without ids says nothing about placement.
  const local = fromApiOverride({});
  local.selectedGpuIds = [1];
  local.selectedGpuIndexKind = "vulkan";

  // biome-ignore lint/style/useNamingConvention: API schema
  const hydrated = fromApiOverride({ gpu_ids: [] }, local);
  assert.deepEqual(hydrated.selectedGpuIds, [1]);
  assert.equal(hydrated.selectedGpuIndexKind, "vulkan");
  // The pin travels with its namespace so a Vulkan ordinal is never read as a physical index.
  const sent = toApiOverride(local);
  assert.deepEqual(sent.gpu_ids, [1]);
  assert.equal(sent.gpu_index_kind, "vulkan");
  assert.equal(
    toApiOverride({ ...local, selectedGpuIndexKind: "physical" }).gpu_index_kind,
    undefined,
  );
});

// Mirrors tests/test_model_override_schema_compatibility.py::OVERRIDE_KEY_FOLDS.
const OVERRIDE_KEY_FOLDS: [string, string, boolean][] = [
  ["C:\\models\\Foo.gguf", "c:/models/foo.gguf", true],
  ["C:\\models\\Foo.gguf", "C:\\models\\Foo.gguf\\", true],
  ["//share/models/Foo.gguf", "\\\\SHARE\\models\\foo.gguf", true],
  ["/mnt/c/models/Foo.gguf", "/mnt/C/models/foo.gguf", true],
  ["/models/Foo.gguf", "/models/foo.gguf", false],
  ["unsloth/Repo-GGUF", "UNSLOTH/repo-gguf", true],
  ["unsloth/Repo-GGUF:Q4_K_M", "unsloth/repo-gguf:q4_k_m", true],
  ["/models/foo.gguf", "models/foo.gguf", false],
  ["models/foo.gguf", "/models/foo.gguf", false],
];

for (const [storedKey, lookupKey, sameModel] of OVERRIDE_KEY_FOLDS) {
  const reach = sameModel ? "is reached from" : "is not reached from";
  test(`${JSON.stringify(storedKey)} ${reach} ${JSON.stringify(lookupKey)}`, () => {
    // biome-ignore lint/style/useNamingConvention: API schema
    const row = { cache_ram: -1 };
    assert.equal(
      resolveStoredOverride({ [storedKey]: row }, [lookupKey]),
      sameModel ? row : null,
    );
  });
}

// No renderer for the panel, so its effect is read off source.
const PANEL = readFileSync(
  path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

test("opening the panel ticks Remember for any model with a resolvable row", () => {
  // Adoption is unconditional on user intent; a guard needing intent must change here.
  // Spelled term by term so a new condition fails the whole-block match and names itself.
  const adoptGuard = [
    "if \\(",
    "resolvedRow &&",
    "serverConfig &&",
    // An edit the other editor made before this read is already in configAtStart.
    "!isModelConfigDraftEdited\\(draftKey\\) &&",
    "configRef\\.current === configAtStart &&",
    "rememberRef\\.current === rememberAtStart",
    "\\) \\{",
  ].join("\\s*\\n\\s*");
  assert.match(PANEL, new RegExp(adoptGuard));
  assert.match(
    PANEL,
    /replaceModelConfigDraft\(draftKey, serverConfig, \{\s*remember: true,\s*savedRemember: true,\s*\}\);/,
  );
  // Local-only write; a default merge is a clear that must travel, so it is unconditional.
  assert.match(
    PANEL,
    /savePerModelConfig\(\s*configId,\s*target\.ggufVariant,\s*storedSpeculativeAuto\(rememberedConfig, !target\.isGguf\),/,
  );
  // Clear evicted models first, or their server rows keep applying unforgettably.
  assert.match(PANEL, /for \(const dropped of hydrationEvicted\)[\s\S]*?return;/);
});

// Last, because it replaces the storage fake.

test("a localStorage that throws degrades on every path instead of propagating", () => {
  // Hydration ignores savePerModelConfig's result in a promise, so a throw is unhandled.
  const throwing: StorageFake = {
    getItem: () => {
      throw new Error("SecurityError");
    },
    setItem: () => {
      const error = new Error("QuotaExceededError");
      error.name = "QuotaExceededError";
      throw error;
    },
    removeItem: () => undefined,
  };
  const asWindow = globalThis as unknown as { window: { localStorage: StorageFake } };
  Object.assign(globalThis, { localStorage: throwing });
  asWindow.window.localStorage = throwing;
  try {
    assert.equal(savePerModelConfig(MODEL, VARIANT, config({ cacheRam: -1 })), false);
    assert.deepEqual(resolveInitialConfig(MODEL, VARIANT), {
      config: { ...DEFAULT_PER_MODEL_CONFIG },
      remembered: false,
    });
    assert.deepEqual(listPerModelConfigs(), []);
    assert.equal(deletePerModelConfig(MODEL, VARIANT), true);
  } finally {
    Object.assign(globalThis, { localStorage: storage });
    asWindow.window.localStorage = storage;
  }
});


test("the MLX drafter mirrors to the server and back", () => {
  const config = normalizePerModelConfig({ speculativeType: "eagle3", specDraftModel: "o/d" });
  assert.equal(toApiOverride(config).spec_draft_model, "o/d");
  // biome-ignore lint/style/useNamingConvention: API schema
  const row = { speculative_type: "auto", spec_draft_model: "o/d" };
  assert.equal(fromApiOverride(row).specDraftModel, "o/d");
  // Without the flag the backend keeps a drafter the save cleared.
  const overrides = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "../src/features/model-picker/api/model-overrides.ts",
  );
  assert.match(readFileSync(overrides, "utf8"), /mirrors_spec_draft_model: true/);
});
