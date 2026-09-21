// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Rows written by an Unsloth without per-chat sampling, rows written by one that has more of
// it than this build, and the values at the edges of what the sanitizer accepts. Every chat
// in an existing installation is the first case: the snapshot is re-read through
// sanitizeThreadScopedSettings on every open, and anything it drops is a setting the user
// watched themselves choose.
//
// The falsy set is the one to watch: temperature 0, minP 0, topP 0, topK -1 and an empty
// prompt are deliberate choices and all falsy or negative, so a `||` where a `??` belongs,
// or an `if (value)` guard, silently reverts them. ?? only defers for null and undefined,
// which is why the store's fallbacks have to use it:
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Nullish_coalescing

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";
import {
  drainMockedTimers,
  enableCountedTimers,
} from "./helpers/mock-timer-drain.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);
const { SAMPLING_KEYS, hasThreadScopedSettings, sanitizeThreadScopedSettings } =
  {
    ...(await import("../src/features/chat/utils/thread-scoped-settings.ts")),
    SAMPLING_KEYS: [
      "temperature",
      "topP",
      "topK",
      "minP",
      "repetitionPenalty",
      "presencePenalty",
      "systemPrompt",
      "systemVariables",
    ] as const,
  };

const STORE_URL = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;

const INSTALLATION = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  repetitionPenalty: 1,
  presencePenalty: 0,
  systemPrompt: "INSTALLATION PROMPT",
  systemVariables: "scope=installation",
};

/** A row exactly as an Unsloth from before per-chat sampling would have left it. */
const LEGACY_ROW = {
  reasoningEnabled: false,
  reasoningEffort: "low",
  toolsEnabled: true,
  codeToolsEnabled: false,
  imageToolsEnabled: false,
  webFetchToolsEnabled: true,
  deepResearchEnabled: false,
  artifactsEnabled: true,
  mcpEnabledForChat: false,
  permissionMode: "off",
  ragEnabled: true,
  ragSource: { type: "kb", kbId: "notes" },
  ragMode: "dense",
  ragTopK: 7,
  ragAutoInject: "on",
  ragAutoInjectMinScore: 0.4,
};

let scenario = 0;

/** A store, its two sinks, and the pairing sequence the provider drives. */
async function world(rows: Record<string, Record<string, unknown>> = {}) {
  scenario += 1;
  settingsHttp.settings = { inferenceParams: { ...INSTALLATION } };
  settingsHttp.puts.length = 0;
  settingsHttp.gate = null;
  settingsHttp.release = null;
  threadRows.reset();
  for (const [threadId, row] of Object.entries(rows)) {
    threadRows.rows.set(threadId, row);
  }
  const mod = await import(`${STORE_URL}?scenario=compat${scenario}`);
  const store = () => mod.useChatRuntimeStore.getState();
  await store().hydratePersistedSettings();
  return {
    mod,
    store,
    open(threadId: string, override?: unknown) {
      store().setActiveThreadId(threadId);
      mod.beginThreadScopedPairing(threadId);
      const raw =
        override === undefined ? threadRows.rows.get(threadId) : override;
      const settings =
        raw === null || raw === undefined
          ? null
          : sanitizeThreadScopedSettings(raw);
      store().applyThreadScopedSettings(
        threadId,
        settings !== null && hasThreadScopedSettings(settings)
          ? settings
          : null,
      );
    },
    sampling() {
      const out: Record<string, unknown> = {};
      for (const key of SAMPLING_KEYS) out[key] = store().params[key];
      return out;
    },
  };
}

// Wait out the debounced write each case asserts on. The wait is on the store's own
// outstanding timers and on the module loader, not on a round count: three rounds passed on
// a dev box's node 24 and failed on the node 22 CI pins, and every count picked since has
// been a guess that fails silently in one direction. See tests/helpers/mock-timer-drain.ts.
async function settle(
  mod: { awaitStartedThreadScopedSettingsWrites: () => Promise<void> },
  tick: (ms: number) => void,
  until?: () => boolean,
): Promise<void> {
  await drainMockedTimers(tick, {
    until,
    label: "settle",
    barrier: () => mod.awaitStartedThreadScopedSettingsWrites(),
  });
}

function assertUsable(sampling: Record<string, unknown>, where: string): void {
  for (const key of SAMPLING_KEYS) {
    const value = sampling[key];
    if (key === "systemPrompt" || key === "systemVariables") {
      assert.equal(typeof value, "string", `${where}: ${key} is not a string`);
      continue;
    }
    assert.equal(typeof value, "number", `${where}: ${key} is not a number`);
    assert.ok(
      Number.isFinite(value as number),
      `${where}: ${key} is ${String(value)}`,
    );
  }
}

// ---------------------------------------------------------------------------
// C1 -- a row from before any of this existed
// ---------------------------------------------------------------------------

test("C1: a legacy row opens on the installation sampling, and nothing is zeroed", async (t) => {
  enableCountedTimers(t);
  const w = await world({ L: { ...LEGACY_ROW } });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  w.open("L");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  assertUsable(w.sampling(), "legacy open");
  // the installation's values, since the row says nothing about sampling
  assert.deepEqual(w.sampling(), INSTALLATION);
  // and everything the row DID say is applied
  assert.equal(w.store().ragTopK, 7);
  assert.equal(w.store().permissionMode, "off");
  assert.equal(w.store().reasoningEnabled, false);
  assert.equal(w.store().toolsEnabled, true);
  // the row is left as it was: a chat that already has a snapshot is not re-pinned,
  // so an old Unsloth reading it back still finds only the keys it knows.
  assert.deepEqual(threadRows.rows.get("L"), LEGACY_ROW);
});

// The behaviour a live review item asks about, reported as observed, not as desired: a
// legacy chat stored no sampling, so it follows the installation defaults and a model load
// moves those. Nothing chosen is lost, but the second visit shows different numbers.
test("C1b: a legacy chat follows the installation defaults, which a model load moves", async (t) => {
  enableCountedTimers(t);
  const w = await world({ L: { ...LEGACY_ROW } });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("L");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  const onFirstVisit = w.sampling();
  assert.equal(onFirstVisit.temperature, INSTALLATION.temperature);

  // a different model loads, publishing its own recommendation
  w.store().setParams(
    {
      ...w.store().params,
      temperature: 0.31,
      topP: 0.41,
      checkpoint: "unsloth/Qwen3.5-9B-GGUF",
    },
    { fromModelDefaults: true },
  );
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  // the chat pinned nothing, so it takes the model's values, as it did before this
  assert.equal(w.store().params.temperature, 0.31);

  w.open("Z");
  w.open("L");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  const onSecondVisit = w.sampling();
  assertUsable(onSecondVisit, "legacy reopen");
  // OBSERVED: the reopened legacy chat shows the model's sampling. It stored none, so the
  // fallback is the installation default, and the load moved that.
  assert.equal(onSecondVisit.temperature, 0.31);
  assert.notEqual(onSecondVisit.temperature, onFirstVisit.temperature);
  // and it is still not carrying any sampling of its own
  assert.deepEqual(threadRows.rows.get("L"), LEGACY_ROW);
});

test("C1c: a legacy chat that the user then edits pins the WHOLE set, not just the edit", async (t) => {
  enableCountedTimers(t);
  const w = await world({ L: { ...LEGACY_ROW } });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("L");
  w.store().setParams({ ...w.store().params, temperature: 1.37 });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // The write for a chat that already had a snapshot is a full replacement built from the
  // store, so all seven other keys are pinned too and the chat stops drifting.
  const row = threadRows.rows.get("L") as Record<string, unknown>;
  assert.equal(row.temperature, 1.37);
  for (const key of SAMPLING_KEYS) {
    assert.notEqual(row[key], undefined, `${key} was not pinned`);
  }
  // the legacy keys survive the replacement
  assert.equal(row.ragTopK, 7);
  assert.equal(row.permissionMode, "off");
  assert.deepEqual(row.ragSource, { type: "kb", kbId: "notes" });

  // and a model load no longer moves it
  w.store().setParams(
    {
      ...w.store().params,
      temperature: 0.31,
      checkpoint: "unsloth/Qwen3.5-9B-GGUF",
    },
    { fromModelDefaults: true },
  );
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal(w.store().params.temperature, 1.37);
});

// ---------------------------------------------------------------------------
// C2 -- nothing at all
// ---------------------------------------------------------------------------

test("C2: an empty, null or absent snapshot opens on the installation settings", async (t) => {
  enableCountedTimers(t);
  for (const snapshot of [null, {}, undefined]) {
    const w = await world();
    await settle(w.mod, (ms) => t.mock.timers.tick(ms));
    w.open("E", snapshot === undefined ? null : snapshot);
    await settle(w.mod, (ms) => t.mock.timers.tick(ms));
    const where = `snapshot ${JSON.stringify(snapshot ?? null)}`;
    assertUsable(w.sampling(), where);
    assert.deepEqual(w.sampling(), INSTALLATION, where);
    // a chat with nothing stored pins what it opened on, so later default changes
    // cannot rewrite what it runs with
    const row = threadRows.rows.get("E") as Record<string, unknown>;
    for (const key of SAMPLING_KEYS) {
      assert.equal(
        row[key],
        INSTALLATION[key as keyof typeof INSTALLATION],
        where,
      );
    }
  }
});

test("C2b: hasThreadScopedSettings tells an empty snapshot from a falsy one", () => {
  assert.equal(hasThreadScopedSettings(null), false);
  assert.equal(hasThreadScopedSettings(undefined), false);
  assert.equal(hasThreadScopedSettings({}), false);
  // every one of these IS a snapshot, however falsy the value
  assert.equal(hasThreadScopedSettings({ temperature: 0 }), true);
  assert.equal(hasThreadScopedSettings({ topP: 0 }), true);
  assert.equal(hasThreadScopedSettings({ minP: 0 }), true);
  assert.equal(hasThreadScopedSettings({ topK: -1 }), true);
  assert.equal(hasThreadScopedSettings({ systemPrompt: "" }), true);
  assert.equal(hasThreadScopedSettings({ systemVariables: "" }), true);
});

// ---------------------------------------------------------------------------
// C3 -- a row from an Unsloth newer or stranger than this one
// ---------------------------------------------------------------------------

test("C3: unknown future keys are dropped, and the known ones still arrive", () => {
  const settings = sanitizeThreadScopedSettings({
    temperature: 0.42,
    systemPrompt: "kept",
    // whatever a later build decides to store per chat
    samplerOrder: ["min_p", "temperature"],
    dryMultiplier: 0.8,
    xtcThreshold: 0.1,
    speculativeDraft: { model: "unsloth/tiny", tokens: 4 },
    reasoningBudget: 8192,
    __proto__: { polluted: true },
  });
  assert.deepEqual(settings, { temperature: 0.42, systemPrompt: "kept" });
  assert.equal(
    (Object.prototype as unknown as Record<string, unknown>).polluted,
    undefined,
    "the sanitizer let a prototype through",
  );
});

test("C3b: wrong-typed and out-of-range values are dropped, never coerced", () => {
  const settings = sanitizeThreadScopedSettings({
    temperature: "hot",
    topK: 999,
    topP: -5,
    minP: null,
    repetitionPenalty: 0.5,
    presencePenalty: -2,
    systemPrompt: 12345,
    systemVariables: {},
  });
  assert.deepEqual(settings, {});
});

test("C3c: the sanitizer never throws, whatever it is handed", () => {
  const hostile: unknown[] = [
    null,
    undefined,
    0,
    "",
    "temperature",
    [1, 2, 3],
    Number.NaN,
    () => undefined,
    new Date(),
    Object.create(null),
    { temperature: Number.NaN },
    { temperature: Number.POSITIVE_INFINITY },
    { temperature: Number.NEGATIVE_INFINITY },
    { topK: 1.5 },
    { topK: Number.MAX_SAFE_INTEGER },
    { systemPrompt: Symbol("s") },
    { systemPrompt: { toString: () => "not a string" } },
    { ragSource: { type: "kb" } },
    { ragSource: "kb" },
    {
      get temperature() {
        throw new Error("should not be reached twice");
      },
    },
  ];
  for (const value of hostile) {
    let settings: Record<string, unknown> = {};
    try {
      settings = sanitizeThreadScopedSettings(value) as Record<string, unknown>;
    } catch (error) {
      // A throwing getter is the only case allowed to propagate, and only because a
      // row is JSON: it cannot carry one. Everything else must be handled.
      assert.ok(
        value !== null &&
          typeof value === "object" &&
          Object.getOwnPropertyDescriptor(value, "temperature")?.get !==
            undefined,
        `sanitizeThreadScopedSettings threw on ${String(value)}: ${String(error)}`,
      );
      continue;
    }
    for (const [key, sanitized] of Object.entries(settings)) {
      assert.notEqual(sanitized, undefined, `${key} came through as undefined`);
      assert.notEqual(sanitized, null, `${key} came through as null`);
      if (typeof sanitized === "number") {
        assert.ok(
          Number.isFinite(sanitized),
          `${key} came through as ${sanitized}`,
        );
      }
    }
  }
});

test("C3d: a NaN or Infinity in a row cannot reach the store", async (t) => {
  enableCountedTimers(t);
  const w = await world({
    N: {
      ...LEGACY_ROW,
      temperature: Number.NaN,
      topP: Number.POSITIVE_INFINITY,
      minP: Number.NEGATIVE_INFINITY,
      systemPrompt: null,
    },
  });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("N");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assertUsable(w.sampling(), "NaN row");
  assert.deepEqual(w.sampling(), INSTALLATION);
});

// A model recommendation the sanitizer refuses. Nothing ships one today (Llasa's top_p:
// 1.2 was brought back to 1.0), but the load path applies a recommendation to the live
// params unclamped, so a custom model_defaults yaml still reaches this.
test("C3e: an out-of-range recommendation never reaches a chat that pinned that key", async (t) => {
  enableCountedTimers(t);
  const w = await world();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("A");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  w.store().setParams(
    { ...w.store().params, topP: 1.2, checkpoint: "unsloth/Llasa-3B" },
    { fromModelDefaults: true },
  );
  w.store().setParams({ ...w.store().params, temperature: 1.37 });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // A pinned every key when it opened, so it keeps its own in-range value. Pinning on open
  // is what makes this safe: an unpinned chat has nothing to put back.
  assert.equal(w.store().params.topP, INSTALLATION.topP);
  const row = threadRows.rows.get("A") as Record<string, unknown>;
  assert.equal(row.topP, INSTALLATION.topP);
  assert.equal(row.temperature, 1.37);

  w.open("Z");
  w.open("A");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assertUsable(w.sampling(), "after an out-of-range recommendation");
  assert.equal(w.store().params.temperature, 1.37);
  assert.equal(w.store().params.topP, INSTALLATION.topP);
});

test("C3f: an out-of-range recommendation taken with no chat open cannot be pinned", async (t) => {
  enableCountedTimers(t);
  const w = await world();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // no chat open, so nothing puts an in-range value back
  w.store().setParams(
    { ...w.store().params, topP: 1.2, checkpoint: "unsloth/Llasa-3B" },
    { fromModelDefaults: true },
  );
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  // OBSERVED: the live params run it, and it reaches the installation settings, which
  // have no such bound. Only the per-chat row does.
  assert.equal(w.store().params.topP, 1.2);
  assert.match(JSON.stringify(settingsHttp.puts), /"topP":1\.2/);

  // a chat opened on it pins what it can, omitting topP rather than sending a body the
  // PATCH would refuse whole, which would cost the chat all seven other keys
  w.open("A");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  const row = threadRows.rows.get("A") as Record<string, unknown>;
  assert.equal(row.topP, undefined);
  assert.equal(row.temperature, INSTALLATION.temperature);
  assert.equal(row.systemPrompt, INSTALLATION.systemPrompt);

  // so the chat keeps running on 1.2, and on reopen has no stored value for it
  assertUsable(w.sampling(), "out-of-range recommendation, no chat open");
  assert.equal(w.store().params.topP, 1.2);
});

// ---------------------------------------------------------------------------
// C4 -- the falsy and negative edges
// ---------------------------------------------------------------------------

const FALSY_EDGE = {
  temperature: 0,
  topP: 0,
  topK: -1,
  minP: 0,
  repetitionPenalty: 1,
  presencePenalty: 0,
  systemPrompt: "",
  systemVariables: "",
};

test("C4: the sanitizer keeps every falsy and negative value", () => {
  assert.deepEqual(sanitizeThreadScopedSettings(FALSY_EDGE), FALSY_EDGE);
  // topK 0 is inside the range too, and distinct from -1
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: 0 }), { topK: 0 });
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: -1 }), { topK: -1 });
  // and each on its own, so a partial row cannot lose one
  for (const [key, value] of Object.entries(FALSY_EDGE)) {
    assert.deepEqual(
      sanitizeThreadScopedSettings({ [key]: value }),
      { [key]: value },
      `${key}=${JSON.stringify(value)} was dropped`,
    );
  }
});

test("C4b: a falsy edit round-trips capture -> persist -> restore", async (t) => {
  enableCountedTimers(t);
  for (const topK of [-1, 0]) {
    const edge = { ...FALSY_EDGE, topK };
    const w = await world();
    await settle(w.mod, (ms) => t.mock.timers.tick(ms));
    w.open("A");
    w.store().setParams({ ...w.store().params, ...edge });
    await settle(w.mod, (ms) => t.mock.timers.tick(ms));

    // captured onto the chat, whole
    const row = threadRows.rows.get("A") as Record<string, unknown>;
    for (const [key, value] of Object.entries(edge)) {
      assert.equal(row[key], value, `topK ${topK}: ${key} was not stored`);
    }
    // and not onto the installation
    assert.deepEqual(settingsHttp.puts, [], `topK ${topK}: an edit leaked`);

    // another chat opens on the installation values, untouched by any of it
    w.open("B");
    await settle(w.mod, (ms) => t.mock.timers.tick(ms));
    assert.deepEqual(w.sampling(), INSTALLATION, `topK ${topK}: B inherited A`);

    // and A gets all of it back
    w.open("A");
    await settle(w.mod, (ms) => t.mock.timers.tick(ms));
    assert.deepEqual(
      w.sampling(),
      edge,
      `topK ${topK}: A lost a value on reopen`,
    );
  }
});

test("C4c: a falsy pinned value survives a model load and a model switch", async (t) => {
  enableCountedTimers(t);
  const w = await world();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("A");
  w.store().setParams({ ...w.store().params, ...FALSY_EDGE });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // a load with a full, non-falsy recommendation
  w.store().setParams(
    {
      ...w.store().params,
      temperature: 0.31,
      topP: 0.41,
      topK: 33,
      minP: 0.02,
      presencePenalty: 0.25,
      systemPrompt: "the model's",
      checkpoint: "unsloth/Qwen3.5-9B-GGUF",
    },
    { fromModelDefaults: true },
  );
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.deepEqual(w.sampling(), FALSY_EDGE, "the load overwrote a falsy pin");

  // and an external switch, which has no load after it to put anything back
  w.store().setCheckpoint("external::anthropic::claude-opus-5", null);
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.deepEqual(
    w.sampling(),
    FALSY_EDGE,
    "the switch overwrote a falsy pin",
  );

  // none of it reached the model's memory either
  const byModel = JSON.stringify(w.store().paramsByModel);
  assert.doesNotMatch(byModel, /"topK":-1/);
});

test("C4d: an empty system prompt is a choice, not an absent one", async (t) => {
  enableCountedTimers(t);
  const w = await world();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  // A is given a prompt, B is deliberately cleared
  w.open("A");
  w.store().setParams({ ...w.store().params, systemPrompt: "A's prompt" });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("B");
  w.store().setParams({ ...w.store().params, systemPrompt: "" });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  assert.equal(
    (threadRows.rows.get("B") as Record<string, unknown>).systemPrompt,
    "",
  );
  w.open("A");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal(w.store().params.systemPrompt, "A's prompt");
  w.open("B");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal(
    w.store().params.systemPrompt,
    "",
    "the cleared prompt came back as the installation's",
  );
});

// ---------------------------------------------------------------------------
// C5 -- a prompt far larger than anything a slider produces
// ---------------------------------------------------------------------------

test("C5: a one-megabyte system prompt is neither truncated nor fatal", async (t) => {
  enableCountedTimers(t);
  const huge = "x".repeat(1024 * 1024);
  assert.equal(huge.length, 1_048_576);

  // through the sanitizer untouched
  assert.equal(
    sanitizeThreadScopedSettings({ systemPrompt: huge }).systemPrompt,
    huge,
  );

  const w = await world();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("A");
  w.store().setParams({ ...w.store().params, systemPrompt: huge });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  const row = threadRows.rows.get("A") as Record<string, unknown>;
  assert.equal((row.systemPrompt as string).length, huge.length);
  assert.equal(row.systemPrompt, huge);

  // and back again after a visit elsewhere
  w.open("B");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal(w.store().params.systemPrompt, INSTALLATION.systemPrompt);
  w.open("A");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal((w.store().params.systemPrompt as string).length, huge.length);
  // and it stayed the chat's: a megabyte in the installation payload would be sent
  // on every settings write for the rest of the session
  assert.doesNotMatch(
    JSON.stringify(settingsHttp.puts).slice(0, 200_000),
    /xxxxxxxxxx/,
  );
});
