// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The eight sampling keys became per-chat, and unlike every other thread-scoped setting
// they live under `params` rather than as store fields of their own. That difference is
// enough to lose an edit or leak it, and neither shows until a chat is reopened, so this
// drives the real store through every ordering of the ops that touch them, checking after
// EVERY step that:
//
//   I1  a value the user set in a chat is still what that chat shows when it is reopened
//   I2  it reaches neither another chat nor the installation-wide settings
//   I3  an edit made with NO chat open does reach the installation-wide settings
//   I4  a model's own recommendation reaches the installation and the model's memory,
//       and is never pinned onto the open chat
//   I5  a chat's pinned values never reach paramsByModel
//   I6  the Qwen Think toggle does land on a chat that pins sampling
//   I7  no operation leaves a sampling param unusable or outside the range the row stores
//
// The orderings are generated, not hand-written: the failures this is for are all
// "these three things in the other order".

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";
import {
  drainMockedTimers,
  enableCountedTimers,
} from "./helpers/mock-timer-drain.ts";

const { store: localStorageFake, fireWindowEvent } = installLocalStorageFake();
// Skip the legacy import path: it would look for settings this test never wrote.
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);
const {
  EXTERNAL,
  INSTALLATION,
  LLAMA,
  MODEL_DEFAULTS,
  QWEN,
  SAMPLING_KEYS,
  runScenario,
} = await import("./helpers/thread-sampling-world.ts");
type Op = Parameters<typeof runScenario>[0][number];

const STORE_URL = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;

/** Every ordering of `items`. */
function permutations<T>(items: readonly T[]): T[][] {
  if (items.length <= 1) return [[...items]];
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += 1) {
    const rest = [...items.slice(0, i), ...items.slice(i + 1)];
    for (const tail of permutations(rest)) out.push([items[i], ...tail]);
  }
  return out;
}

/** Every ordered selection of `size` distinct items from `items`. */
function arrangements<T>(items: readonly T[], size: number): T[][] {
  if (size === 0) return [[]];
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += 1) {
    const rest = [...items.slice(0, i), ...items.slice(i + 1)];
    for (const tail of arrangements(rest, size - 1))
      out.push([items[i], ...tail]);
  }
  return out;
}

/** Run a batch of orderings and fail with the first broken invariant of each. */
async function sweep(
  label: string,
  orderings: readonly Op[][],
  tick: (ms: number) => void,
  strict = true,
): Promise<void> {
  const failures: string[] = [];
  for (const ops of orderings) {
    const violations = await runScenario(ops, tick, strict);
    for (const violation of violations) {
      failures.push(
        `[${violation.invariant}] after ${violation.step} in ${ops.join(" > ")}: ${violation.detail}`,
      );
    }
  }
  assert.deepEqual(
    failures.slice(0, 12),
    [],
    `${label}: ${failures.length} violation(s) across ${orderings.length} orderings`,
  );
}

// ---------------------------------------------------------------------------
// A. the ordering matrix
// ---------------------------------------------------------------------------

test("A1: every ordering of edit / load / Think / model switch / reopen", async (t) => {
  enableCountedTimers(t);
  const orderings = permutations<Op>([
    "editTemp",
    "loadQwen",
    "qwenToggleOn",
    "switchLlama",
    "reopenA",
  ]).map((tail) => ["hydrate", "openA", ...tail] as Op[]);
  assert.equal(orderings.length, 120);
  await sweep("A1", orderings, (ms) => t.mock.timers.tick(ms));
});

test("A2: every ordering of prompt edit / post-load defaults / external / unload / second chat", async (t) => {
  enableCountedTimers(t);
  const orderings = permutations<Op>([
    "editPrompt",
    "qwenPostLoad",
    "switchExternal",
    "unload",
    "openB",
  ]).map(
    (tail) => ["hydrate", "loadQwen", "openA", ...tail, "reopenA"] as Op[],
  );
  assert.equal(orderings.length, 120);
  await sweep("A2", orderings, (ms) => t.mock.timers.tick(ms));
});

test("A3: both chats, both edits, both Think positions, in every order", async (t) => {
  enableCountedTimers(t);
  const orderings = permutations<Op>([
    "editTemp",
    "editPrompt",
    "openB",
    "qwenToggleOff",
    "switchLlama",
  ]).map(
    (tail) =>
      ["hydrate", "loadQwen", "openA", ...tail, "reopenA", "openB"] as Op[],
  );
  assert.equal(orderings.length, 120);
  await sweep("A3", orderings, (ms) => t.mock.timers.tick(ms));
});

test("A4: every four-step interleaving over the wider alphabet", async (t) => {
  enableCountedTimers(t);
  const alphabet: Op[] = [
    "editTemp",
    "editPrompt",
    "loadQwen",
    "qwenPostLoad",
    "openB",
    "switchExternal",
    "unload",
  ];
  const orderings = arrangements(alphabet, 4).map(
    (middle) => ["hydrate", "openA", ...middle, "reopenA"] as Op[],
  );
  assert.equal(orderings.length, 840);
  await sweep("A4", orderings, (ms) => t.mock.timers.tick(ms));
});

test("A5: hydration interleaved -- nothing leaks, whatever the order", async (t) => {
  enableCountedTimers(t);
  // Without the shadow model: a chat opened before the server answered follows this
  // browser's cache, so what it is "owed" is not yet decided. The leak and usability
  // invariants still hold, and they are the ones that matter here.
  const orderings = permutations<Op>([
    "hydrate",
    "openA",
    "editTemp",
    "loadQwen",
    "openB",
  ]).map((ops) => [...ops, "reopenA"] as Op[]);
  assert.equal(orderings.length, 120);
  await sweep("A5", orderings, (ms) => t.mock.timers.tick(ms), false);
});

test("A6: hand-picked long sequences", async (t) => {
  enableCountedTimers(t);
  const long: Op[][] = [
    // the reported gap: two chats, a model in between, back to the first
    [
      "hydrate",
      "openA",
      "editTemp",
      "editPrompt",
      "loadQwen",
      "openB",
      "editTemp",
      "switchLlama",
      "reopenA",
      "openB",
      "reopenA",
    ],
    // a mode toggled either side of a model switch
    [
      "hydrate",
      "loadQwen",
      "openA",
      "qwenToggleOn",
      "switchLlama",
      "qwenToggleOff",
      "reopenA",
      "switchExternal",
      "reopenA",
    ],
    // an unload in the middle of a pinned chat
    [
      "hydrate",
      "loadQwen",
      "openA",
      "editTemp",
      "unload",
      "reopenA",
      "loadQwen",
      "reopenA",
    ],
    // every model transition there is, with a pinned chat open throughout
    [
      "hydrate",
      "openA",
      "editTemp",
      "editPrompt",
      "loadQwen",
      "switchLlama",
      "switchExternal",
      "unload",
      "loadQwen",
      "reopenA",
    ],
    // the installation edited first, then a chat that must not inherit the next one
    [
      "hydrate",
      "editTemp",
      "editPrompt",
      "openA",
      "editTemp",
      "openB",
      "reopenA",
      "openB",
    ],
    // post-load defaults arriving repeatedly, as a status poll does
    [
      "hydrate",
      "loadQwen",
      "openA",
      "editTemp",
      "qwenPostLoad",
      "qwenPostLoad",
      "qwenPostLoad",
      "reopenA",
    ],
    // a chat opened, left, and returned to twice over
    [
      "hydrate",
      "openA",
      "editPrompt",
      "openB",
      "reopenA",
      "openB",
      "reopenA",
      "openB",
      "reopenA",
    ],
    // Think toggled in one chat must not follow the user into the other
    [
      "hydrate",
      "loadQwen",
      "openA",
      "qwenToggleOn",
      "openB",
      "qwenToggleOff",
      "reopenA",
      "openB",
    ],
  ];
  await sweep("A6", long, (ms) => t.mock.timers.tick(ms));
});

// ---------------------------------------------------------------------------
// The three facts the matrix asserts negatively, asserted positively once each.
// A sweep that leaks nothing because nothing is ever stored would pass otherwise.
// ---------------------------------------------------------------------------

test("the pinned values really are stored on the chat's own row", async (t) => {
  enableCountedTimers(t);
  await runScenario(
    ["hydrate", "openA", "editTemp", "editPrompt", "openB", "reopenA"],
    (ms) => t.mock.timers.tick(ms),
  );
  const row = threadRows.rows.get("A") as Record<string, unknown>;
  assert.equal(row.temperature, 1.37);
  assert.equal(row.systemPrompt, "CHAT A ONLY 5f3a");
  // and the whole set, so a later default change cannot rewrite what it runs with
  for (const key of SAMPLING_KEYS) {
    assert.notEqual(row[key], undefined, `${key} is not on the row`);
  }
});

test("I3: an edit with no chat open reaches the installation-wide settings", async (t) => {
  enableCountedTimers(t);
  await runScenario(["hydrate", "editTemp", "editPrompt"], (ms) =>
    t.mock.timers.tick(ms),
  );
  const sent = JSON.stringify(settingsHttp.puts);
  assert.match(sent, /0\.83/);
  assert.match(sent, /NO CHAT OPEN 7e44/);
});

test("I4: a model's recommendation reaches the installation and the model's memory", async (t) => {
  enableCountedTimers(t);
  await runScenario(
    ["hydrate", "openA", "editTemp", "editPrompt", "loadQwen", "switchLlama"],
    (ms) => t.mock.timers.tick(ms),
  );
  const sent = JSON.stringify(settingsHttp.puts);
  // the installation copy, even though the chat that was open kept its own values
  assert.match(sent, /"temperature":0\.31/);
  assert.match(sent, /"topP":0\.41/);
  // and the model's own memory, taken when the model was left
  const remembered = JSON.parse(
    JSON.stringify(
      (settingsHttp.puts.find((put) => "inferenceParamsByModel" in put)
        ?.inferenceParamsByModel as Record<string, unknown>) ?? {},
    ),
  ) as Record<string, Record<string, unknown>>;
  assert.equal(remembered[QWEN]?.temperature, MODEL_DEFAULTS.temperature);
  // ...carrying the installation's prompt, never the open chat's
  assert.equal(remembered[QWEN]?.systemPrompt, INSTALLATION.systemPrompt);
});

// ---------------------------------------------------------------------------
// B. races
// ---------------------------------------------------------------------------

let raceScenario = 0;

/** A store, a Qwen module bound to it, and the two sinks, all freshly wired. */
async function raceWorld() {
  raceScenario += 1;
  settingsHttp.settings = { inferenceParams: { ...INSTALLATION } };
  settingsHttp.puts.length = 0;
  settingsHttp.gate = null;
  settingsHttp.release = null;
  threadRows.reset();
  const mod = await import(`${STORE_URL}?scenario=race${raceScenario}`);
  const { sanitizeThreadScopedSettings } = await import(
    "../src/features/chat/utils/thread-scoped-settings.ts"
  );
  const store = () => mod.useChatRuntimeStore.getState();
  return {
    mod,
    store,
    /** setActiveThreadId, open the pairing window, but do NOT answer the read yet. */
    beginOpen(threadId: string) {
      store().setActiveThreadId(threadId);
      mod.beginThreadScopedPairing(threadId);
    },
    /** The chat's read comes back. */
    finishOpen(threadId: string, settings?: Record<string, unknown> | null) {
      const row =
        settings === undefined ? threadRows.rows.get(threadId) : settings;
      store().applyThreadScopedSettings(
        threadId,
        row ? sanitizeThreadScopedSettings(row) : null,
      );
    },
    open(threadId: string) {
      this.beginOpen(threadId);
      this.finishOpen(threadId);
    },
  };
}

// Wait out the debounced write each race test asserts on. The wait is on the store's own
// outstanding timers and on the module loader, not on a round count, so a slower runtime
// takes more rounds instead of silently returning short and turning a stale read into a
// wrong-value failure; see tests/helpers/mock-timer-drain.ts for why a count was never the
// right bound.
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

test("B1: a slider moved while /api/chat/settings is still in flight survives it", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  settingsHttp.hold();
  const hydrating = w.store().hydratePersistedSettings();
  w.store().setParams({ ...w.store().params, temperature: 1.42 });
  settingsHttp.release?.();
  await hydrating;
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal(
    w.store().params.temperature,
    1.42,
    "the edit was hydrated over",
  );
  // and a key the user did not touch still takes the server's value
  assert.equal(w.store().params.systemPrompt, INSTALLATION.systemPrompt);
});

test("B2: an edit made while the chat's own read is out is stored on that chat", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  settingsHttp.puts.length = 0;

  w.beginOpen("A");
  // the user moves a slider and types a prompt before A's snapshot lands
  w.store().setParams({
    ...w.store().params,
    temperature: 1.37,
    systemPrompt: "HELD EDIT 5f3a",
  });
  w.finishOpen("A", { temperature: 0.22, topP: 0.33, systemPrompt: "STORED" });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // the held edit wins over what the read brought back
  assert.equal(w.store().params.temperature, 1.37);
  assert.equal(w.store().params.systemPrompt, "HELD EDIT 5f3a");
  // a key the user did not touch takes the stored value
  assert.equal(w.store().params.topP, 0.33);
  // and it reached A's row, not the installation
  const row = threadRows.rows.get("A") as Record<string, unknown>;
  assert.equal(row.temperature, 1.37);
  assert.equal(row.systemPrompt, "HELD EDIT 5f3a");
  assert.doesNotMatch(JSON.stringify(settingsHttp.puts), /HELD EDIT 5f3a/);
  assert.doesNotMatch(JSON.stringify(settingsHttp.puts), /1\.37/);
});

test("B3: leaving mid-read sends the held edit to its own chat, not the next one", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  settingsHttp.puts.length = 0;

  w.beginOpen("A");
  w.store().setParams({
    ...w.store().params,
    temperature: 1.37,
    systemPrompt: "HELD EDIT 5f3a",
  });
  // the user gives up on A and opens B before A's read lands
  await w.mod.commitHeldThreadScopedEditsToTheirThread();
  w.open("B");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // A's row carries the edit as a merge, leaving the rest of its snapshot alone
  const merged = threadRows
    .writesFor("A")
    .filter((write) => write.settingsPatch !== undefined);
  assert.equal(merged.length, 1);
  assert.deepEqual(merged[0].settingsPatch, {
    temperature: 1.37,
    systemPrompt: "HELD EDIT 5f3a",
  });
  // and it is A's alone as far as the installation is concerned
  assert.doesNotMatch(JSON.stringify(settingsHttp.puts), /HELD EDIT/);
  assert.doesNotMatch(JSON.stringify(settingsHttp.puts), /1\.37/);

  // What B shows is NOT asserted here. Leaving a chat mid-read leaves the store holding its
  // edits, and B's applyThreadScopedSettings captures the store as the in-memory defaults,
  // so B opens on A's values and pins them. Long-standing behaviour of the held-edit path,
  // not new: `ragTopK` leaks through the same line. Asserting it would freeze it in place.
});

test("B4: a model load landing during the pairing window does not take the chat's edit", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  settingsHttp.puts.length = 0;

  w.beginOpen("A");
  w.store().setParams({ ...w.store().params, temperature: 1.37 });
  // the load finishes while A's read is still out
  w.store().setParams(
    { ...w.store().params, ...MODEL_DEFAULTS, checkpoint: QWEN },
    { fromModelDefaults: true },
  );
  w.finishOpen("A", null);
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  assert.equal(w.store().params.temperature, 1.37, "the load took the edit");
  // and the chat was pinned with the user's value, not the one the load published
  const row = threadRows.rows.get("A") as Record<string, unknown>;
  assert.equal(
    row.temperature,
    1.37,
    "the model's value was pinned onto the chat",
  );
  // a key the user did not touch still follows the model
  assert.equal(w.store().params.topP, MODEL_DEFAULTS.topP);
  // the model's own recommendation still reached the installation
  assert.match(JSON.stringify(settingsHttp.puts), /"topP":0\.41/);
  // but the chat's temperature did not
  assert.doesNotMatch(JSON.stringify(settingsHttp.puts), /1\.37/);
});

test("B4b: the held sampling edit that survives a load is the LAST one made", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  w.beginOpen("A");
  // a slider dragged twice, then the load, then the read
  w.store().setParams({ ...w.store().params, temperature: 1.2 });
  w.store().setParams({ ...w.store().params, temperature: 1.37 });
  w.store().setParams(
    { ...w.store().params, ...MODEL_DEFAULTS, checkpoint: QWEN },
    { fromModelDefaults: true },
  );
  w.finishOpen("A", null);
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal(w.store().params.temperature, 1.37);
});

test("B4c: a falsy held edit is not treated as no edit at all", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  w.beginOpen("A");
  // 0, "" and -1 are all deliberate choices, and all falsy or negative
  w.store().setParams({
    ...w.store().params,
    temperature: 0,
    minP: 0,
    topK: -1,
    systemPrompt: "",
  });
  w.store().setParams(
    { ...w.store().params, ...MODEL_DEFAULTS, checkpoint: QWEN },
    { fromModelDefaults: true },
  );
  w.finishOpen("A", null);
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  const params = w.store().params;
  assert.equal(params.temperature, 0);
  assert.equal(params.minP, 0);
  assert.equal(params.topK, -1);
  assert.equal(params.systemPrompt, "");
  const row = threadRows.rows.get("A") as Record<string, unknown>;
  assert.equal(row.temperature, 0);
  assert.equal(row.minP, 0);
  assert.equal(row.topK, -1);
  assert.equal(row.systemPrompt, "");
});

test("B5: two rapid model switches with a pinned chat open", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("A");
  w.store().setParams({
    ...w.store().params,
    temperature: 1.37,
    systemPrompt: "CHAT A ONLY 5f3a",
  });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  settingsHttp.puts.length = 0;

  // no settle between them: the second switch lands while the first is still writing
  w.store().setCheckpoint(LLAMA, null);
  w.store().setCheckpoint(EXTERNAL, null);
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  assert.equal(w.store().params.temperature, 1.37);
  assert.equal(w.store().params.systemPrompt, "CHAT A ONLY 5f3a");
  const byModel = JSON.stringify(w.store().paramsByModel);
  assert.doesNotMatch(
    byModel,
    /CHAT A ONLY/,
    "the chat's prompt became a model's",
  );
  assert.doesNotMatch(
    byModel,
    /1\.37/,
    "the chat's temperature became a model's",
  );
  assert.doesNotMatch(JSON.stringify(settingsHttp.puts), /CHAT A ONLY/);
});

test("B6: a thread switch while a load is in flight keeps each chat's own values", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  w.open("A");
  w.store().setParams({ ...w.store().params, temperature: 1.37 });
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // the user opens B, and the load that was started in A finishes into B
  w.beginOpen("B");
  w.store().setParams(
    { ...w.store().params, ...MODEL_DEFAULTS, checkpoint: QWEN },
    { fromModelDefaults: true },
  );
  w.finishOpen("B", null);
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.notEqual(
    w.store().params.temperature,
    1.37,
    "A's temperature followed into B",
  );

  w.open("A");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  assert.equal(w.store().params.temperature, 1.37, "A lost its temperature");
});

test("B7: a tab closing with a held edit beacons it to the chat it was made in", async (t) => {
  enableCountedTimers(t);
  const w = await raceWorld();
  await w.store().hydratePersistedSettings();
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));
  settingsHttp.puts.length = 0;

  w.beginOpen("A");
  w.store().setParams({
    ...w.store().params,
    temperature: 1.37,
    systemPrompt: "HELD EDIT 5f3a",
  });
  // pagehide, which is what the store listens on; the read never landed
  const delivered = fireWindowEvent("pagehide", {});
  assert.ok(delivered > 0, "the store is not listening for the terminal event");
  await settle(w.mod, (ms) => t.mock.timers.tick(ms));

  // The beacon is a PATCH, so it is not in `puts`; what it queued for replay is the
  // durable record of what the closing tab tried to save, and for which chat.
  const beaconed = JSON.parse(
    localStorageFake.get("unsloth_chat_thread_settings_replay") ?? "{}",
  ) as Record<string, { settingsPatch?: Record<string, unknown> }>;
  assert.deepEqual(beaconed.A?.settingsPatch, {
    temperature: 1.37,
    systemPrompt: "HELD EDIT 5f3a",
  });
  // the tab-close flush of the installation settings must not carry it
  assert.doesNotMatch(JSON.stringify(settingsHttp.puts), /HELD EDIT 5f3a/);
});
