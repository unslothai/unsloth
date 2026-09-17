// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One scenario of the per-chat sampling state machine: a fresh copy of the store, the
// twelve things a user can do to it, a shadow model of what each chat is owed, and the
// invariants that have to hold after every single one of them.
//
// The store keeps the whole feature in module state, so a scenario gets its own module
// instance through the "?scenario=" query thread-sampling-resolver.mjs understands.
// Register that resolver and install the localStorage fake before importing this.

import { drainMockedTimers } from "./mock-timer-drain.ts";
import { threadRows } from "./store-stubs/chat-history-storage.ts";
import { settingsHttp } from "./store-stubs/settings-http.ts";

const STORE_URL = new URL(
  "../../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;
const QWEN_URL = new URL(
  "../../src/features/chat/utils/qwen-params.ts",
  import.meta.url,
).href;
const SETTINGS_URL = new URL(
  "../../src/features/chat/utils/thread-scoped-settings.ts",
  import.meta.url,
).href;

export const SAMPLING_KEYS = [
  "temperature",
  "topP",
  "topK",
  "minP",
  "repetitionPenalty",
  "presencePenalty",
  "systemPrompt",
  "systemVariables",
] as const;

export type SamplingKey = (typeof SAMPLING_KEYS)[number];

/** The ranges PATCH /api/chat/threads/{id} enforces, which the sanitizer mirrors. A live
 * value outside one of these cannot be stored, so the chat loses it on the next reopen. */
const BOUNDS: Record<string, { min: number; max: number }> = {
  temperature: { min: 0, max: 2 },
  topP: { min: 0, max: 1 },
  topK: { min: -1, max: 100 },
  minP: { min: 0, max: 1 },
  repetitionPenalty: { min: 1, max: 2 },
  presencePenalty: { min: 0, max: 2 },
};

export const QWEN = "unsloth/Qwen3.5-9B-GGUF";
export const LLAMA = "unsloth/Llama-4-8B";
export const EXTERNAL = "external::anthropic::claude-opus-5";

/** What the installation is holding before anything happens. */
export const INSTALLATION = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  repetitionPenalty: 1,
  presencePenalty: 0,
  systemPrompt: "INSTALLATION PROMPT",
  systemVariables: "scope=installation",
};

/** The recommendation a Qwen load publishes. None of it may be pinned onto a chat. */
export const MODEL_DEFAULTS = {
  temperature: 0.31,
  topP: 0.41,
  topK: 33,
  minP: 0.02,
  repetitionPenalty: 1.05,
  presencePenalty: 0.25,
};

/** Per-chat sentinels. A value tagged for one chat appearing anywhere else is the leak. */
const EDITS: Record<string, { temperature: number; systemPrompt: string }> = {
  A: { temperature: 1.37, systemPrompt: "CHAT A ONLY 5f3a" },
  B: { temperature: 1.11, systemPrompt: "CHAT B ONLY 9c21" },
  // With no chat open an edit is the installation's, and must reach the server.
  "": { temperature: 0.83, systemPrompt: "NO CHAT OPEN 7e44" },
};

/** The Qwen3 recommended table, as qwen-params.ts applies it. */
function qwenTable(thinkingOn: boolean, checkpoint: string) {
  const base = thinkingOn
    ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0 }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0 };
  const lower = checkpoint.toLowerCase();
  return lower.includes("qwen3.5") || lower.includes("qwen3.6")
    ? { ...base, presencePenalty: 1.5 }
    : base;
}

export const OPS = [
  "hydrate",
  "openA",
  "openB",
  "reopenA",
  "editTemp",
  "editPrompt",
  "loadQwen",
  "qwenPostLoad",
  "qwenToggleOn",
  "qwenToggleOff",
  "switchLlama",
  "switchExternal",
  "unload",
] as const;

export type Op = (typeof OPS)[number];

export interface Violation {
  invariant: string;
  step: string;
  detail: string;
}

let scenarioCounter = 0;

export interface World {
  run(ops: readonly Op[]): Promise<Violation[]>;
}

interface StoreModule {
  useChatRuntimeStore: {
    getState: () => Record<string, (...args: never[]) => unknown> & {
      params: Record<string, unknown>;
      paramsByModel: Record<string, Record<string, unknown>>;
      activePresetSource: string;
      settingsHydrated: boolean;
    };
  };
  beginThreadScopedPairing: (threadId: string) => void;
  awaitStartedThreadScopedSettingsWrites: () => Promise<void>;
}

/** Let the debounced writers and their promise chains finish.
 *
 * On the store's own pending timers and its own write chains, not on a round count: a
 * scenario whose write has not landed yet looks exactly like one that wrote the wrong
 * value, so an under-drain here
 * surfaces as an ORDERING violation ("chat A temperature: owed 0.6, shows 1.37") and sends
 * the reader into the store instead of into the wait. drainMockedTimers throws when it
 * gives up, which is the whole point of it; see tests/helpers/mock-timer-drain.ts for the
 * measurements that killed the fixed count and for the observable NOT to drain on.
 *
 * The caller must have enabled the clock with enableCountedTimers(t). */
async function drain(mod: StoreModule, tick: (ms: number) => void): Promise<void> {
  await drainMockedTimers(tick, {
    label: "runScenario drain",
    barrier: () => mod.awaitStartedThreadScopedSettingsWrites(),
  });
}

/**
 * Run one ordering against a fresh store and return every invariant it broke.
 *
 * `tick` advances the mocked clock so the 400ms settings and snapshot debounces settle
 * between ops instead of bleeding into the next scenario.
 *
 * `strict` turns on the shadow-model check (I1/I4/I6): every value a chat is owed is still
 * what it shows. Orderings that hydrate in the middle run without it, because a chat opened
 * before the server answered is following a cache, not a choice.
 */
export async function runScenario(
  ops: readonly Op[],
  tick: (ms: number) => void,
  strict = true,
): Promise<Violation[]> {
  scenarioCounter += 1;
  settingsHttp.settings = { inferenceParams: { ...INSTALLATION } };
  settingsHttp.puts.length = 0;
  settingsHttp.gate = null;
  settingsHttp.release = null;
  threadRows.reset();

  const mod: StoreModule = await import(
    `${STORE_URL}?scenario=${scenarioCounter}`
  );
  const qwen: { applyQwenThinkingParams: (on: boolean) => void } = await import(
    `${QWEN_URL}?scenario=${scenarioCounter}`
  );
  const {
    sanitizeThreadScopedSettings,
  }: {
    sanitizeThreadScopedSettings: (value: unknown) => Record<string, unknown>;
  } = await import(SETTINGS_URL);

  const violations: Violation[] = [];
  /** What each chat is owed: every sampling key it has been given a value for. */
  const owed = new Map<string, Record<string, unknown>>();
  /** The chat whose snapshot the store is showing, or "" for none. */
  let applied = "";
  /** Edits made with no chat open: the installation owes each of these a write. */
  const globalEdits: Record<string, unknown> = {};

  const state = () => mod.useChatRuntimeStore.getState();

  const record = (invariant: string, step: string, detail: string) => {
    violations.push({ invariant, step, detail });
  };

  const sampling = (params: Record<string, unknown>) => {
    const out: Record<string, unknown> = {};
    for (const key of SAMPLING_KEYS) out[key] = params[key];
    return out;
  };

  /** Does `blob` mention a value that belongs to `threadId` alone? */
  const mentions = (blob: unknown, threadId: string): string | null => {
    const text = JSON.stringify(blob ?? null);
    const edit = EDITS[threadId];
    if (text.includes(JSON.stringify(edit.systemPrompt))) {
      return edit.systemPrompt;
    }
    // Numbers are matched structurally, not by substring: 1.37 inside 11.37 is not a hit.
    const hit = (value: unknown): boolean => {
      if (value === edit.temperature) return true;
      if (Array.isArray(value)) return value.some(hit);
      if (value !== null && typeof value === "object") {
        return Object.values(value).some(hit);
      }
      return false;
    };
    return hit(blob) ? String(edit.temperature) : null;
  };

  const check = (step: string) => {
    const live = state();
    const params = live.params;

    // I7 -- every sampling param the backend would be sent is a usable value.
    for (const key of SAMPLING_KEYS) {
      const value = params[key];
      if (key === "systemPrompt" || key === "systemVariables") {
        if (typeof value !== "string") {
          record("I7", step, `${key} is ${JSON.stringify(value)}`);
        }
        continue;
      }
      if (typeof value !== "number" || !Number.isFinite(value)) {
        record("I7", step, `${key} is ${JSON.stringify(value)}`);
        continue;
      }
      const bound = BOUNDS[key];
      if (value < bound.min || value > bound.max) {
        // Out of the PATCH range: the sanitizer drops it, so the chat cannot store it.
        record(
          "I7-range",
          step,
          `${key}=${value} outside [${bound.min},${bound.max}]`,
        );
      }
    }

    // I2 -- a value one chat owns reaches neither the installation nor any other chat.
    for (const threadId of ["A", "B"]) {
      const leaked = mentions(settingsHttp.puts, threadId);
      if (leaked !== null) {
        record(
          "I2-installation",
          step,
          `chat ${threadId}'s ${leaked} in a PUT`,
        );
      }
      const other = threadId === "A" ? "B" : "A";
      const inOther = mentions(threadRows.rows.get(other) ?? null, threadId);
      if (inOther !== null) {
        record(
          "I2-other-chat",
          step,
          `chat ${threadId}'s ${inOther} in ${other}'s row`,
        );
      }
      if (applied === other) {
        const onScreen = mentions(sampling(params), threadId);
        if (onScreen !== null) {
          record(
            "I2-on-screen",
            step,
            `chat ${threadId}'s ${onScreen} shown in ${other}`,
          );
        }
      }
      // I5 -- and never against a model, which every other chat replays from.
      const inMemory = mentions(live.paramsByModel, threadId);
      if (inMemory !== null) {
        record("I5", step, `chat ${threadId}'s ${inMemory} in paramsByModel`);
      }
    }

    // I3 -- an edit made with no chat open is the installation's, and has to reach it.
    if (live.settingsHydrated) {
      for (const [key, value] of Object.entries(globalEdits)) {
        const sent = settingsHttp.puts.some((put) => {
          const params = put.inferenceParams as
            | Record<string, unknown>
            | undefined;
          return params !== undefined && Object.is(params[key], value);
        });
        if (!sent) {
          record(
            "I3",
            step,
            `${key}=${JSON.stringify(value)} set with no chat open never reached /api/chat/settings`,
          );
        }
      }
    }

    // I1/I4/I6 -- every value the open chat is owed is still the value it shows.
    if (strict && applied !== "" && owed.has(applied)) {
      const want = owed.get(applied) as Record<string, unknown>;
      for (const [key, value] of Object.entries(want)) {
        if (!Object.is(params[key], value)) {
          record(
            "I1",
            step,
            `chat ${applied} ${key}: owed ${JSON.stringify(value)}, shows ${JSON.stringify(params[key])}`,
          );
        }
      }
    }
  };

  /** The chat the user is acting in, or "" when none is applied. */
  const actor = () => applied;

  const open = (threadId: string) => {
    state().setActiveThreadId(threadId as never);
    mod.beginThreadScopedPairing(threadId);
    const row = threadRows.rows.get(threadId);
    state().applyThreadScopedSettings(
      threadId as never,
      (row ? sanitizeThreadScopedSettings(row) : null) as never,
    );
    applied = threadId;
    // A chat with no snapshot pins whatever it opened on, so from here that is what it
    // is owed. A chat that has one keeps the model built up on its earlier visits.
    if (!owed.has(threadId)) {
      owed.set(threadId, sampling(state().params));
    }
  };

  const editParam = (patch: Record<string, unknown>) => {
    const live = state();
    live.setParams({ ...live.params, ...patch } as never);
    const who = actor();
    if (who === "") {
      // Only once hydration answered: setParams gates its HTTP write on settingsHydrated,
      // so an earlier edit is fenced instead and reaches the server on the NEXT edit.
      if (live.settingsHydrated) Object.assign(globalEdits, patch);
      return;
    }
    Object.assign(owed.get(who) ?? {}, patch);
  };

  const perform = async (op: Op): Promise<void> => {
    const live = state();
    switch (op) {
      case "hydrate":
        await live.hydratePersistedSettings();
        break;
      case "openA":
      case "reopenA":
        open("A");
        break;
      case "openB":
        open("B");
        break;
      case "editTemp":
        editParam({ temperature: EDITS[actor()].temperature });
        break;
      case "editPrompt":
        editParam({ systemPrompt: EDITS[actor()].systemPrompt });
        break;
      case "loadQwen":
        // An interactive load: the destination checkpoint and its recommendation.
        live.setParams(
          { ...live.params, ...MODEL_DEFAULTS, checkpoint: QWEN } as never,
          { fromModelDefaults: true } as never,
        );
        break;
      case "qwenPostLoad":
        // The toggle's table, but published by the load, so it is marked and never pinned.
        live.setParams(
          {
            ...live.params,
            ...qwenTable(true, String(live.params.checkpoint ?? "")),
          } as never,
          { fromModelDefaults: true } as never,
        );
        break;
      case "qwenToggleOn":
      case "qwenToggleOff": {
        const on = op === "qwenToggleOn";
        const checkpoint = String(live.params.checkpoint ?? "");
        qwen.applyQwenThinkingParams(on);
        // A user action, so a chat takes it -- but only where the toggle applies.
        if (
          checkpoint.toLowerCase().includes("qwen3") &&
          live.activePresetSource === "builtin-default"
        ) {
          const who = actor();
          if (who !== "") {
            Object.assign(owed.get(who) ?? {}, qwenTable(on, checkpoint));
          }
        }
        break;
      }
      case "switchLlama":
        live.setCheckpoint(LLAMA as never, null as never);
        break;
      case "switchExternal":
        live.setCheckpoint(EXTERNAL as never, null as never);
        break;
      case "unload":
        live.clearCheckpoint();
        break;
    }
  };

  for (const op of ops) {
    await perform(op);
    await drain(mod, tick);
    check(op);
    if (violations.length > 0) break;
  }
  return violations;
}
