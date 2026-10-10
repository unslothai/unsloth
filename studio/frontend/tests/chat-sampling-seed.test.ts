// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A null seed means "let the server draw one", so every layer must test `!== undefined`.

import assert from "node:assert/strict";
import test from "node:test";

import { readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

import {
  PERSISTED_INFERENCE_PARAM_KEYS,
  REMEMBERED_INFERENCE_PARAM_KEYS,
  getReplayedParams,
  pickRememberedParams,
} from "../src/features/chat/lib/per-model-params.ts";
import {
  type ChatModelRow,
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
  MAX_SAMPLING_SEED,
  modelReadsSamplingSeed,
} from "../src/features/chat/types/runtime.ts";
import {
  THREAD_SCOPED_PARAM_KEYS,
  isThreadScopedSettingKey,
  sanitizeThreadScopedSettings,
} from "../src/features/chat/utils/thread-scoped-settings.ts";

// Dynamic: preset-policy imports ../types/runtime extensionless, and a static import would
// resolve before registerBundlerResolver's hook is live.
const { applyPresetParams, getPresetOwnedParams, isSamePresetConfig } =
  await import("../src/features/chat/presets/preset-policy.ts");

const GGUF = "unsloth/Qwen3.5-9B-GGUF";

function params(overrides: Partial<InferenceParams> = {}): InferenceParams {
  return { ...DEFAULT_INFERENCE_PARAMS, checkpoint: GGUF, ...overrides };
}

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start + from.length);
  assert.ok(start !== -1, `not found: ${from}`);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

test("an untouched install sends no seed", () => {
  assert.equal(DEFAULT_INFERENCE_PARAMS.seed, null);
});

test("the seed persists and is remembered per model", () => {
  assert.ok(PERSISTED_INFERENCE_PARAM_KEYS.includes("seed"));
  assert.ok(REMEMBERED_INFERENCE_PARAM_KEYS.includes("seed"));
});

test("a seed of 0 is remembered rather than read as unset", () => {
  const picked = pickRememberedParams(params({ seed: 0 }));
  assert.equal(picked.seed, 0);
});

test("a cleared seed is remembered as the clear", () => {
  const picked = pickRememberedParams(params({ seed: null }));
  assert.ok("seed" in picked);
  assert.equal(picked.seed, null);
});

test("switching models replays each model's own seed", () => {
  const replayed = getReplayedParams(
    true,
    { [GGUF]: { seed: 3407 } },
    params({ seed: 11 }),
    GGUF,
    true,
  );
  assert.equal(replayed.seed, 3407);
});

test("a model whose row cleared the seed replays the clear", () => {
  const replayed = getReplayedParams(
    true,
    { [GGUF]: { seed: null } },
    params({ seed: 3407 }),
    GGUF,
    true,
  );
  assert.equal(replayed.seed, null);
});

test("a row written before the seed existed keeps what is on screen", () => {
  const replayed = getReplayedParams(
    true,
    { [GGUF]: { temperature: 0.7 } },
    params({ seed: 3407 }),
    GGUF,
    true,
  );
  assert.equal(replayed.seed, 3407);
});

test("the settings sanitizer lets a cleared seed through", () => {
  const storage = readText("../src/features/chat/utils/chat-settings-storage.ts");
  const body = slice(
    storage,
    "function sanitizeInferenceParams(",
    "\nfunction sanitizeInferenceParamsByModel(",
  );
  assert.match(body, /value\.seed === null/);
  assert.match(body, /Number\.isInteger\(value\.seed\)/);
});

test("the request omits the seed when it is unset", () => {
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /params\.seed == null \|\|/);
  assert.match(adapter, /\{ seed: params\.seed \}/);
});

test("the pin covers llama.cpp's whole uint32 range bar the sentinel", () => {
  // 0xFFFFFFFF is LLAMA_DEFAULT_SEED; llama-server parses a uint32, so do not cap at int32.
  assert.equal(MAX_SAMPLING_SEED, 0xffffffff - 1);
});

function row(
  overrides: Partial<Parameters<typeof modelReadsSamplingSeed>[0] & object>,
) {
  return {
    isGguf: false,
    isMlx: false,
    isAudio: false,
    hasAudioInput: false,
    ...overrides,
  };
}

test("a seed reaches only the backends that read one", () => {
  assert.ok(modelReadsSamplingSeed(row({ isGguf: true })));
  assert.ok(modelReadsSamplingSeed(row({ isMlx: true })));
  assert.ok(!modelReadsSamplingSeed(row({})));
  assert.ok(!modelReadsSamplingSeed(null));
  assert.ok(!modelReadsSamplingSeed(undefined));
  assert.ok(
    !modelReadsSamplingSeed(
      row({ isGguf: true, isAudio: true, hasAudioInput: false }),
    ),
  );
  assert.ok(
    modelReadsSamplingSeed(
      row({ isGguf: true, isAudio: true, hasAudioInput: true }),
    ),
  );
});

test("the panel offers the seed only where the backend reads it", () => {
  const sheet = readText("../src/features/chat/chat-settings-sheet.tsx");
  assert.match(sheet, /const showSeed = modelReadsSamplingSeed\(/);
  // type="number" reports unparseable input as "", which would silently clear the pin.
  const field = slice(sheet, "{showSeed ? (", 'aria-label="Seed"');
  const props = field
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => !line.startsWith("//"));
  assert.ok(props.includes('type="text"'));
  assert.ok(!props.includes('type="number"'));
  // isGguf also caps Max Tokens, so it must stay separate from the seed gate.
  assert.doesNotMatch(sheet, /const isGguf = modelReadsSamplingSeed\(/);
});

test("the panel and the request body gate on the same argument", () => {
  const gateArguments = (source: string): string[] =>
    [...source.matchAll(/modelReadsSamplingSeed\(([^)]*)\)/g)].map((match) =>
      match[1].replace(/\s+/g, " ").trim(),
    );
  assert.deepEqual(
    gateArguments(readText("../src/features/chat/chat-settings-sheet.tsx")),
    ["activeModel"],
  );
  assert.deepEqual(
    gateArguments(readText("../src/features/chat/api/chat-adapter.ts")),
    ["activeModel"],
  );
});

test("an over-long entry clamps rather than becoming another number", () => {
  const sheet = readText("../src/features/chat/chat-settings-sheet.tsx");
  const handler = slice(sheet, "const committedSeed = useMemo", "const paramsWithCommittedSeed");
  assert.doesNotMatch(handler, /\.slice\(0, ?10\)/);
  assert.match(handler, /digits\.length > 10\s*\?\s*MAX_SAMPLING_SEED/);
  assert.match(handler, /\^0\+\(\?=\\d\)/);
});

test("typing is not rewritten before the entry is finished", () => {
  const sheet = readText("../src/features/chat/chat-settings-sheet.tsx");
  const field = slice(sheet, "{showSeed ? (", "placeholder=\"Random\"");
  assert.match(field, /value=\{\s*seedDraft \?\?/);
  assert.match(field, /onChange=\{\(e\) =>\s*setSeedDraft\(/);
  assert.match(field, /if \(seedDraft === null\) return;/);
  assert.match(field, /if \(e\.key === "Enter"\) e\.currentTarget\.blur\(\);/);
});

test("an abandoned entry does not outlive the box it was typed into", () => {
  // Removing a focused element fires no blur, so a draft must also be committed on unmount.
  const sheet = readText("../src/features/chat/chat-settings-sheet.tsx");
  const reset = slice(sheet, "useEffect(() => {\n    setSeedDraft(null);", ");");
  assert.match(reset, /\[currentCheckpoint, showSeed\]/);
});

test("a preset carries the seed it was saved with", () => {
  const saved = getPresetOwnedParams(params({ seed: 3407 }));
  assert.equal(saved.seed, 3407);
  const applied = applyPresetParams(
    params({ seed: 11 }),
    params({ seed: 3407 }),
  );
  assert.equal(applied.seed, 3407);
  assert.equal(
    applyPresetParams(params({ seed: 3407 }), params({ seed: null })).seed,
    null,
  );
});

test("moving the seed marks the preset modified", () => {
  assert.ok(!isSamePresetConfig(params({ seed: 3407 }), params({ seed: 11 })));
  assert.ok(isSamePresetConfig(params({ seed: 3407 }), params({ seed: 3407 })));
});

test("the stored seed is range-checked, not just the keystroke", () => {
  const storage = readText("../src/features/chat/utils/chat-settings-storage.ts");
  const body = slice(
    storage,
    "function sanitizeInferenceParams(",
    "\nfunction sanitizeInferenceParamsByModel(",
  );
  assert.match(body, /value\.seed >= 0/);
  assert.match(body, /value\.seed <= MAX_SAMPLING_SEED/);
});

test("the request drops a seed the loaded model cannot use", () => {
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /!modelReadsSamplingSeed\(/);
  assert.match(adapter, /modelReadsSamplingSeed\(activeModel\)/);
});

test("the seed belongs to the chat, not the installation", () => {
  assert.ok(THREAD_SCOPED_PARAM_KEYS.includes("seed"));
  assert.equal(isThreadScopedSettingKey("seed"), true);
});

test("a chat stores a cleared seed rather than dropping the key", () => {
  assert.deepEqual(sanitizeThreadScopedSettings({ seed: null }), { seed: null });
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: 40 }), { topK: 40 });
  assert.deepEqual(sanitizeThreadScopedSettings({ seed: 3407 }), { seed: 3407 });
  for (const bad of [-1, 1.5, MAX_SAMPLING_SEED + 1, true, "3407"]) {
    assert.deepEqual(sanitizeThreadScopedSettings({ seed: bad }), {}, String(bad));
  }
});

test("a cleared seed is not read as a missing key", () => {
  const store = readText("../src/features/chat/stores/chat-runtime-store.ts");
  const helper = slice(store, "function firstSetThreadScopedValue", "\n}");
  // `??` would fall through a cleared null seed to the default; only undefined means unset.
  assert.match(helper, /values\.find\(\(value\) => value !== undefined\)/);
});

// Compile-time check that every summary minting site sets the capability flags.
type RequiredKeys<T> = {
  [K in keyof T]-?: object extends Pick<T, K> ? never : K;
}[keyof T];
type Assert<T extends true> = T;
type _EveryFlagTheSeedGateReadsIsRequired = Assert<
  "isGguf" | "isMlx" | "isAudio" | "hasAudioInput" extends
    RequiredKeys<ChatModelRow>
    ? true
    : false
>;

const everyFlagTheSeedGateReads: _EveryFlagTheSeedGateReadsIsRequired = true;

test("a models[] row states every flag the seed gate reads", () => {
  assert.equal(everyFlagTheSeedGateReads, true);
  const row: ChatModelRow = {
    id: "m",
    name: "m",
    isVision: false,
    isLora: false,
    isGguf: true,
    isMlx: false,
    isAudio: false,
    hasAudioInput: false,
  };
  assert.ok(modelReadsSamplingSeed(row));
});

test("saving a preset takes the seed being typed, not the one before it", () => {
  const panel = readText("../src/features/chat/chat-settings-sheet.tsx");
  // React has not re-rendered by onClick after blur, so Save must read the committed seed.
  assert.match(panel, /params: toPresetParams\(paramsWithCommittedSeed\)/);
  const unsaved = slice(panel, "const hasUnsavedPresetChanges", "const presetSaveState");
  assert.match(unsaved, /isSamePresetConfig\(\s*activePresetDefinition\.params,\s*paramsWithCommittedSeed,/);
  assert.match(unsaved, /committedSeed !== \(params\.seed \?\? null\)/);
  assert.match(panel, /setSeedDraft\(null\);\s*setSeed\(committedSeed\);/);
});
