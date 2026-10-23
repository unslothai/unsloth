// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Each scenario gets its own store module via the "?scenario=" query.
// Register thread-sampling-resolver.mjs and the localStorage fake before importing.

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

/** Ranges PATCH /api/chat/threads/{id} enforces, mirrored by the sanitizer. */
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

export const MODEL_DEFAULTS = {
  temperature: 0.31,
  topP: 0.41,
  topK: 33,
  minP: 0.02,
  repetitionPenalty: 1.05,
  presencePenalty: 0.25,
};

const EDITS: Record<string, { temperature: number; systemPrompt: string }> = {
  A: { temperature: 1.37, systemPrompt: "CHAT A ONLY 5f3a" },
  B: { temperature: 1.11, systemPrompt: "CHAT B ONLY 9c21" },
  "": { temperature: 0.83, systemPrompt: "NO CHAT OPEN 7e44" },
};

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

/** Drain on pending timers and write chains. Needs enableCountedTimers(t). */
async function drain(mod: StoreModule, tick: (ms: number) => void): Promise<void> {
  await drainMockedTimers(tick, {
    label: "runScenario drain",
    barrier: () => mod.awaitStartedThreadScopedSettingsWrites(),
  });
}

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
  const owed = new Map<string, Record<string, unknown>>();
  let applied = "";
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

  const mentions = (blob: unknown, threadId: string): string | null => {
    const text = JSON.stringify(blob ?? null);
    const edit = EDITS[threadId];
    if (text.includes(JSON.stringify(edit.systemPrompt))) {
      return edit.systemPrompt;
    }
    // Numbers are matched structurally: 1.37 inside 11.37 is not a hit.
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
        record(
          "I7-range",
          step,
          `${key}=${value} outside [${bound.min},${bound.max}]`,
        );
      }
    }

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
      const inMemory = mentions(live.paramsByModel, threadId);
      if (inMemory !== null) {
        record("I5", step, `chat ${threadId}'s ${inMemory} in paramsByModel`);
      }
    }

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
    if (!owed.has(threadId)) {
      owed.set(threadId, sampling(state().params));
    }
  };

  const editParam = (patch: Record<string, unknown>) => {
    const live = state();
    live.setParams({ ...live.params, ...patch } as never);
    const who = actor();
    if (who === "") {
      // setParams gates its HTTP write on settingsHydrated.
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
        live.setParams(
          { ...live.params, ...MODEL_DEFAULTS, checkpoint: QWEN } as never,
          { fromModelDefaults: true } as never,
        );
        break;
      case "qwenPostLoad":
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
