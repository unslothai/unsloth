// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// S2 for the llama-server tuning group: no user setting may be lost on any upgrade
// or downgrade path, and no client may destroy a record it cannot read. Hidden is
// acceptable, lost is not.
//
// Two directions changed at once. The four fields (load mode, draft KV dtype,
// checkpoints, cache RAM) stopped being judged default, so they reach storage at
// all; and the server row became authoritative on panel open. Together they move
// data across builds, across origins, and into a backfill that used to skip it.
//
// There is no migration step, so each direction holds on its own: the version stamp
// is a downgrade LOCK, correct only while toStoredConfig stamps the OLDEST version
// that understands the record.

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
const { fromApiOverride, resolveStoredOverride, toApiOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { backfillModelOverrides } = await import(
  "../src/features/model-picker/api/migrate-model-overrides.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const STORAGE_KEY = "unsloth_model_configs";
const BACKFILL_FLAG = "unsloth_model_overrides_backfilled_v1";
const MODEL = "unsloth/Repo-GGUF";
const VARIANT = "Q4_K_M";

// Ceiling shipped by the last build BEFORE the group. A record stamped at or below
// it is readable, and therefore erasable, by that build.
const PRE_TUNING_CEILING = 4;

// Each of the four on its own, at a value a user can actually choose. The falsy ones
// are in here deliberately: 0 checkpoints and a 0 or -1 cache are decisions, and a
// version stamp or a default check written against truth passes without them.
const TUNING_ONLY_PATCHES: Partial<PerModelConfig>[] = [
  { loadMode: "mmap" },
  { ctxCheckpoints: 0 },
  { ctxCheckpoints: 64 },
  { cacheRam: 0 },
  { cacheRam: -1 },
  // The dtype is tied to a mode that loads a separate drafter, so it cannot be the
  // sole difference from default; the mode travels with it.
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

/** resolveInitialConfig is the public read path; loadPerModelConfig is module-private. */
function load(): PerModelConfig | null {
  const initial = resolveInitialConfig(MODEL, VARIANT);
  return initial.remembered ? initial.config : null;
}

// ---------------------------------------------------------------------------
// A. A NEW client reading OLD records. Nothing may be dropped.
// ---------------------------------------------------------------------------

test("a v0 record with no version key at all still loads the tuning it carried", () => {
  store.clear();
  // Pre-versioning shape. No migration step exists, so the guards read
  // storedConfigVersion() === 0 and normalizeV1 rebuilds the record as it stands.
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
  // The fields that did not exist yet read as unset, not as a bogus default: a
  // fabricated 32 checkpoints would be pinned onto every load of this model.
  assert.equal(loaded.loadMode, null);
  assert.equal(loaded.ctxCheckpoints, null);
  assert.equal(loaded.cacheRam, null);

  // Re-saving without touching a tuning field must NOT poison the record for the
  // build that wrote it: the stamp is a lock, and over-stamping locks that build out
  // of a record it can still read in full.
  assert.ok(savePerModelConfig(MODEL, VARIANT, loaded));
  assert.equal(onlyEntry().version, PRE_TUNING_CEILING);
});

// ---------------------------------------------------------------------------
// B. The property the whole scheme rests on.
// ---------------------------------------------------------------------------

test("only a record that actually carries tuning is stamped v5", () => {
  // Annotated rather than `as const`: the latter makes llamaExtraArgs a readonly
  // tuple, which Partial<PerModelConfig> will not take.
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
  // The other half of the rule. Stamping every record v5 would quarantine the whole
  // store from the build the user just downgraded to, which loses far more than it
  // protects.
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ nParallel: 8 })));
  assert.ok((onlyEntry().version as number) <= PRE_TUNING_CEILING);
});

// ---------------------------------------------------------------------------
// C. A pre-v5 client meeting a v5 record. Hidden is fine; destroyed is not.
// ---------------------------------------------------------------------------

test("a v5 record is out of a pre-v5 build's reach in the first place", () => {
  // Replay what that build actually does: read every record it is allowed to
  // interpret, drop the keys it does not know, write the result back. It must find
  // nothing to rewrite.
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
  // The five guards, exercised together. Any one of them missing is an older client
  // silently destroying a newer record, and the record is the only copy on origins
  // the server row does not reach.
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

  // And once the client understands the schema again, the settings come back.
  const restored = readMap();
  restored[Object.keys(restored)[0]].version = 5;
  writeMap(restored);
  assert.equal(load()?.loadMode, "mmap");
});

// ---------------------------------------------------------------------------
// D. What the change of default judgement moves.
// ---------------------------------------------------------------------------

test("a tuning-only config is stored rather than deleted on the way in", () => {
  // savePerModelConfig DELETES an entry it judges default. Before the four were
  // counted, a save whose only change was one of them reported success, wrote
  // nothing, and came back unremembered on the next open.
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
  // A behaviour change on the first launch after upgrade, and the reason it is
  // pinned rather than merely noted: the backfill gates on isDefaultConfig, so
  // counting the four made it start mirroring configs it used to filter out. It is
  // the right answer -- an API auto-switch of this model would otherwise run without
  // the tuning the picker shows -- but it happens once, unprompted, and only ever
  // adds fields, so it has to be deliberate.
  store.clear();
  assert.ok(savePerModelConfig(MODEL, VARIANT, config({ cacheRam: -1 })));

  const puts: Record<string, unknown>[] = [];
  setAuthFetchHandler((_input, init) => {
    if (init?.method === "PUT") {
      puts.push(JSON.parse(String(init.body)));
      return new Response(JSON.stringify({ overrides: {} }), { status: 200 });
    }
    // The pre-read: an install upgrading into this has a row with no tuning in it.
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
  // Fill, never replace: the server copy is the newer authority, so the pass may add
  // the field this browser holds and must not touch the max_seq_length already there.
  assert.equal(puts[0].fill_absent_fields, true);
  assert.equal(puts[0].remove, false);
  assert.equal(store.get(BACKFILL_FLAG), "1", "a completed pass must not run again");
});

test("an all-default config is still filtered out of the backfill", async () => {
  // The gate the case above walks through has to stay shut for everything else, or
  // every model the user ever opened is mirrored on first launch.
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

// ---------------------------------------------------------------------------
// E. Storage that refuses, and storage that is full.
// ---------------------------------------------------------------------------

test("the eviction loop terminates when the budget needs a future record", () => {
  // deleteOldestEvictableEntry skips a future-schema entry, so a map made entirely of
  // them cannot be brought inside either cap. The loop has to give up rather than
  // spin on a candidate it will never take: an infinite loop here hangs the tab on a
  // save, on a browser profile that has merely been downgraded.
  for (const overBudget of [
    // The entry-count cap: 505 records against MAX_ENTRIES = 500.
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
    // The byte cap: 20 records of 60 KiB each against MAX_PER_MODEL_CONFIG_STORAGE_BYTES.
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

    // Reached at all, which is the termination assertion: node:test kills the process
    // on a hang rather than reporting one, so the failure mode is the run not ending.
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
  // The same cap with a way out. Only the records this build could rewrite anyway are
  // candidates, and they go oldest first.
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
  // Reported back, because eviction is silent and still returns success: without the
  // list the server override of a dropped model keeps applying with nothing in the UI
  // able to forget it.
  assert.deepEqual(evicted, [
    { modelId: "unsloth/Old-A-GGUF", ggufVariant: "Q4_K_M" },
    { modelId: "unsloth/Old-B-GGUF", ggufVariant: "Q4_K_M" },
    { modelId: "unsloth/Filler-0-GGUF", ggufVariant: "Q4_K_M" },
  ]);
});

// ---------------------------------------------------------------------------
// F. The server row as the authority, and what that costs the local copy.
// ---------------------------------------------------------------------------

test("a row that carries no tuning leaves this browser's tuning standing", () => {
  // The mirror is lossy in both directions: a PUT that never landed, a save from a
  // build that did not forward the four, a row written before the route learned them.
  // All three leave the same gap, and reading a gap as a choice deletes settings.
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
  // 0 is a value (the host prompt cache off), so it has to beat a local 4096 rather
  // than read as absent and lose to it.
  assert.equal(hydrated.cacheRam, 0);
});

test("a server value this build refuses falls to the app default, not the local one", () => {
  // The merge takes the server value first and normalizeV1 clamps afterwards, so a
  // refusal is indistinguishable from a chosen default by the time the local value
  // could have been used. The reachable case is not a corrupt row: it is a row
  // carrying a speculative mode with no separate drafter, which invalidates a draft
  // KV dtype this browser legitimately holds.
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

  // Same shape for a load mode the two builds disagree about: the row wins, and
  // losing means the app default rather than what this browser had.
  const pinned = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    load_mode: "mmap",
  });
  // biome-ignore lint/style/useNamingConvention: API schema
  const refused = fromApiOverride({ load_mode: "swap" }, pinned);
  assert.equal(refused.loadMode, null);
});

test("an out-of-range server value clamps rather than falling through", () => {
  // The other resolution, and the reason the case above is worth stating separately:
  // a numeric knob is clamped into range instead of being refused, so the row still
  // wins and the user gets the nearest legal value.
  const local = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    ctx_checkpoints: 64,
  });
  // biome-ignore lint/style/useNamingConvention: API schema
  const hydrated = fromApiOverride({ ctx_checkpoints: 1_000_000 }, local);
  assert.equal(hydrated.ctxCheckpoints, 256);
});

test("an empty server argument list clears a local list rather than being ignored", () => {
  // [] is the tombstone that stops the server's fallback to a broader row, and
  // normalizePerModelConfig collapses an empty list to null, so hydration has to
  // reinstall it. Without that a cleared box comes back holding the legacy bare
  // repository row's flags on the next open.
  const local = fromApiOverride({
    // biome-ignore lint/style/useNamingConvention: API schema
    llama_extra_args: ["--numa", "distribute"],
  });
  assert.deepEqual(local.llamaExtraArgs, ["--numa", "distribute"]);

  // biome-ignore lint/style/useNamingConvention: API schema
  const hydrated = fromApiOverride({ llama_extra_args: [] }, local);
  assert.deepEqual(hydrated.llamaExtraArgs, []);
  // Distinct from "this copy never read the value", which must stay omitted so the
  // route preserves whatever flags the server holds.
  assert.notEqual(hydrated.llamaExtraArgs, undefined);
});

test("an empty server GPU list does not clear a local pin", () => {
  // Deliberate, and worth pinning next to the tombstone above so the two are not
  // "fixed" into agreement: only a PHYSICAL pin travels, so a row without ids says
  // nothing about placement and a Vulkan ordinal keeps its own namespace.
  const local = fromApiOverride({});
  local.selectedGpuIds = [1];
  local.selectedGpuIndexKind = "vulkan";

  // biome-ignore lint/style/useNamingConvention: API schema
  const hydrated = fromApiOverride({ gpu_ids: [] }, local);
  assert.deepEqual(hydrated.selectedGpuIds, [1]);
  assert.equal(hydrated.selectedGpuIndexKind, "vulkan");
  // And toApiOverride is why: a Vulkan pin is never sent, so a row that lacks ids is
  // exactly what a browser holding one produces.
  assert.equal(toApiOverride(local).gpu_ids, undefined);
});

// The identity table the panel's hydration now depends on, mirrored from
// tests/test_model_override_schema_compatibility.py::OVERRIDE_KEY_FOLDS. The server
// resolves the row and the browser has to agree on which model it belongs to, or the
// panel hydrates from another model's settings.
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

// The panel is a component this suite has no renderer for, so the one behaviour that
// only exists inside its effect is read off the source, as the rest of its hydration
// rules are in tests/llama-extra-args-panel-hydration.test.ts.
const PANEL = readFileSync(
  path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

test("opening the panel ticks Remember for any model with a resolvable row", () => {
  // Nothing in the guard asks whether the user ever chose to remember this model:
  // a row exists, the panel has not been edited since the request went out, and
  // Remember goes on and is persisted. That is defensible -- a row IS a remembered
  // setting, whichever origin wrote it -- but it is unconditional, so a later change
  // that wants user intent in the decision has to come through here.
  // Spelled out term by term rather than as one long pattern, so a guard that grows
  // a fourth condition fails on the whole-block match below and names which one.
  const adoptGuard = [
    "if \\(",
    "resolvedRow &&",
    "serverConfig &&",
    "configRef\\.current === configAtStart &&",
    "rememberRef\\.current === rememberAtStart",
    "\\) \\{",
  ].join("\\s*\\n\\s*");
  assert.match(PANEL, new RegExp(adoptGuard));
  assert.match(PANEL, /setRemember\(true\);\s*\n\s*setSavedRemember\(true\);/);
  // The write is local only. An erased server field is therefore NOT restored by
  // opening the panel, even though this browser still holds it: that takes a save.
  // Unconditional, because savePerModelConfig expresses "no settings" by deleting
  // the entry, so a merge that comes out default is a clear that has to travel.
  assert.match(
    PANEL,
    /savePerModelConfig\(\s*configId,\s*target\.ggufVariant,\s*rememberedConfig,/,
  );
  // Whatever that write evicted is cleared before the block returns, or a dropped
  // model keeps applying its server row with nothing able to forget it.
  assert.match(PANEL, /for \(const dropped of hydrationEvicted\)[\s\S]*?return;/);
});

// ---------------------------------------------------------------------------
// G. Storage that refuses every call. Last, because it replaces the fake.
// ---------------------------------------------------------------------------

test("a localStorage that throws degrades on every path instead of propagating", () => {
  // Private mode, a disabled-storage policy, and a full quota all arrive as a throw
  // from getItem or setItem. The hydration effect calls savePerModelConfig from a
  // promise callback and ignores what it returns, so a throw here is an unhandled
  // rejection in a panel the user has merely opened.
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
    // Nothing was stored, so there is nothing to forget: a refusal here would leave
    // the settings page unable to clear a model it can already not read.
    assert.equal(deletePerModelConfig(MODEL, VARIANT), true);
  } finally {
    Object.assign(globalThis, { localStorage: storage });
    asWindow.window.localStorage = storage;
  }
});
