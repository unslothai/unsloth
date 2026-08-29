// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sampling seed is the one inference param whose unset state is a value, not an
// absence: null means "let the server draw one". Every layer it crosses tests presence
// with `!== undefined`, so these pin the places a truthiness check or an omission would
// quietly drop a seed of 0 or a deliberate clear.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

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

// Dynamic, unlike the two above: preset-policy value-imports ../types/runtime without an
// extension, and a static import resolves before registerBundlerResolver's hook is live.
const { applyPresetParams, getPresetOwnedParams, isSamePresetConfig } =
  await import("../src/features/chat/presets/preset-policy.ts");

const GGUF = "unsloth/Qwen3.5-9B-GGUF";

function params(overrides: Partial<InferenceParams> = {}): InferenceParams {
  return { ...DEFAULT_INFERENCE_PARAMS, checkpoint: GGUF, ...overrides };
}

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start + from.length);
  assert.ok(start !== -1, `not found: ${from}`);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

test("an untouched install sends no seed", () => {
  // Blank is the shipped state, so nothing changes for a user who never opens the field.
  assert.equal(DEFAULT_INFERENCE_PARAMS.seed, null);
});

test("the seed persists and is remembered per model", () => {
  assert.ok(PERSISTED_INFERENCE_PARAM_KEYS.includes("seed"));
  // Derived from the persisted list by dropping maxSeqLength; a seed belongs to the
  // sampling config, not the load, so it must survive that filter.
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
  // An explicit null, so the switch does not inherit whichever seed was last on screen.
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
  // A gap is not a clear: the memory stores rows as written and invents nothing, so a
  // pre-feature row leaves the seed alone rather than replaying a value it never held.
  const replayed = getReplayedParams(
    true,
    { [GGUF]: { temperature: 0.7 } },
    params({ seed: 3407 }),
    GGUF,
    true,
  );
  assert.equal(replayed.seed, 3407);
});

// The storage module, the adapter and the panel all pull a .tsx barrel into their
// graph, so pin their source the way the sibling settings tests do.

test("the settings sanitizer lets a cleared seed through", () => {
  const storage = read("../src/features/chat/utils/chat-settings-storage.ts");
  const body = slice(
    storage,
    "function sanitizeInferenceParams(",
    "\nfunction sanitizeInferenceParamsByModel(",
  );
  // It gates both the read and the outgoing PUT, so a branch that only accepts numbers
  // would drop the clear and leave the server's old pin in place.
  assert.match(body, /value\.seed === null/);
  assert.match(body, /Number\.isInteger\(value\.seed\)/);
});

test("the request omits the seed when it is unset", () => {
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /params\.seed == null \|\|/);
  assert.match(adapter, /\{ seed: params\.seed \}/);
});

test("the pin covers llama.cpp's whole uint32 range bar the sentinel", () => {
  // 0xFFFFFFFF is LLAMA_DEFAULT_SEED, llama.cpp's "draw one" value, so it is the one
  // number a pin cannot name. llama-server parses the field as a uint32, so stopping at
  // int32 max instead would refuse half the seeds it accepts for no reason.
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
  // llama-server reads the seed, and so does MLX: generate_chat_response takes it and
  // builds _make_seeded_mlx_sampler, and worker.py's _backend_declares lets it through.
  assert.ok(modelReadsSamplingSeed(row({ isGguf: true })));
  assert.ok(modelReadsSamplingSeed(row({ isMlx: true })));
  // The transformers backend declares no seed kwarg, so the same gate drops it there.
  assert.ok(!modelReadsSamplingSeed(row({})));
  // No summary at all is a model the panel knows nothing about, so it is offered nothing.
  assert.ok(!modelReadsSamplingSeed(null));
  assert.ok(!modelReadsSamplingSeed(undefined));
  // An audio-output model answers through generateAudio, whose request carries no seed, so
  // the control would promise a reproducibility it cannot deliver on that path.
  assert.ok(
    !modelReadsSamplingSeed(
      row({ isGguf: true, isAudio: true, hasAudioInput: false }),
    ),
  );
  // A model that takes audio IN still decodes through llama-server, so it keeps the field.
  assert.ok(
    modelReadsSamplingSeed(
      row({ isGguf: true, isAudio: true, hasAudioInput: true }),
    ),
  );
});

test("the panel offers the seed only where the backend reads it", () => {
  const sheet = read("../src/features/chat/chat-settings-sheet.tsx");
  assert.match(sheet, /const showSeed = modelReadsSamplingSeed\(/);
  // type="number" reports an entry the engine cannot parse as "", which would clear
  // the pin with no error. chat-providers-dialog documents the same trap.
  const field = slice(sheet, "{showSeed ? (", 'aria-label="Seed"');
  const props = field
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => !line.startsWith("//"));
  assert.ok(props.includes('type="text"'));
  assert.ok(!props.includes('type="number"'));
  // isGguf is deliberately not this helper: it also caps Max Tokens, and the two
  // questions must stay separable even though both read the same summary today.
  assert.doesNotMatch(sheet, /const isGguf = modelReadsSamplingSeed\(/);
});

test("the panel and the request body gate on the same argument", () => {
  // A regex on the call name alone cannot see a call site passing an extra argument,
  // which is how the panel and the body drifted apart the first time. Compare the
  // argument text so an added or dropped signal at one site fails here.
  const gateArguments = (source: string): string[] =>
    [...source.matchAll(/modelReadsSamplingSeed\(([^)]*)\)/g)].map((match) =>
      match[1].replace(/\s+/g, " ").trim(),
    );
  assert.deepEqual(
    gateArguments(read("../src/features/chat/chat-settings-sheet.tsx")),
    ["activeModel"],
  );
  assert.deepEqual(
    gateArguments(read("../src/features/chat/api/chat-adapter.ts")),
    ["activeModel"],
  );
});

test("an over-long entry clamps rather than becoming another number", () => {
  const sheet = read("../src/features/chat/chat-settings-sheet.tsx");
  // The clamp lives in committedSeed now, so a click on Save reads the same value blur
  // would have written rather than the one from before the entry.
  const handler = slice(sheet, "const committedSeed = useMemo", "const paramsWithCommittedSeed");
  // Truncating to 10 digits before clamping turned 12345678901234 into 1234567890.
  assert.doesNotMatch(handler, /\.slice\(0, ?10\)/);
  assert.match(handler, /digits\.length > 10\s*\?\s*MAX_SAMPLING_SEED/);
  // And the length is measured after the padding, so a pasted 0000000003407 stays 3407
  // instead of reading as 13 digits and clamping to the maximum.
  assert.match(handler, /\^0\+\(\?=\\d\)/);
});

test("typing is not rewritten before the entry is finished", () => {
  const sheet = read("../src/features/chat/chat-settings-sheet.tsx");
  const field = slice(sheet, "{showSeed ? (", "placeholder=\"Random\"");
  // The box shows the raw draft while it is being typed into, so a clamp cannot
  // rewrite it mid-entry. NumericValueInput keeps a draft for the same reason.
  assert.match(field, /value=\{\s*seedDraft \?\?/);
  assert.match(field, /onChange=\{\(e\) =>\s*setSeedDraft\(/);
  // Blur is the only commit, so it must not fire for a field nobody typed into,
  // and Enter has to reach it.
  assert.match(field, /if \(seedDraft === null\) return;/);
  assert.match(field, /if \(e\.key === "Enter"\) e\.currentTarget\.blur\(\);/);
});

test("an abandoned entry does not outlive the box it was typed into", () => {
  // Blur is the only commit and removing a focused element fires none, so both keys
  // matter: a model switch and the field going away each strand a draft in committedSeed.
  // Shape only: there is no DOM here, so the no-blur premise was measured in a browser.
  const sheet = read("../src/features/chat/chat-settings-sheet.tsx");
  const reset = slice(sheet, "useEffect(() => {\n    setSeedDraft(null);", ");");
  assert.match(reset, /\[currentCheckpoint, showSeed\]/);
});

test("a preset carries the seed it was saved with", () => {
  // The field sits with Temperature through Max Tokens, all preset-owned, so saving a
  // preset while a seed is pinned has to capture it rather than quietly drop it.
  const saved = getPresetOwnedParams(params({ seed: 3407 }));
  assert.equal(saved.seed, 3407);
  const applied = applyPresetParams(
    params({ seed: 11 }),
    params({ seed: 3407 }),
  );
  assert.equal(applied.seed, 3407);
  // And a preset saved with no pin clears one left on screen, so "Default" means default.
  assert.equal(
    applyPresetParams(params({ seed: 3407 }), params({ seed: null })).seed,
    null,
  );
});

test("moving the seed marks the preset modified", () => {
  // Without this the panel would show a saved preset while sampling no longer matches it.
  assert.ok(!isSamePresetConfig(params({ seed: 3407 }), params({ seed: 11 })));
  assert.ok(isSamePresetConfig(params({ seed: 3407 }), params({ seed: 3407 })));
});

test("the stored seed is range-checked, not just the keystroke", () => {
  const storage = read("../src/features/chat/utils/chat-settings-storage.ts");
  const body = slice(
    storage,
    "function sanitizeInferenceParams(",
    "\nfunction sanitizeInferenceParamsByModel(",
  );
  // The panel clamps what the user types; a row from another client or a hand-edited
  // studio.db meets only this, and it feeds the request body straight through.
  assert.match(body, /value\.seed >= 0/);
  assert.match(body, /value\.seed <= MAX_SAMPLING_SEED/);
});

test("the request drops a seed the loaded model cannot use", () => {
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  // A pin set on a GGUF outlives a switch to transformers, where the panel hides the
  // field: without this the user would keep sending a seed they can no longer see.
  assert.match(adapter, /!modelReadsSamplingSeed\(/);
  // The same summary the body already reads isGguf from for context_overflow, so
  // "is this a GGUF" cannot be answered two ways inside one request.
  assert.match(adapter, /activeModel\?\.isGguf === true/);
});

test("the seed belongs to the chat, not the installation", () => {
  // Every other per-turn sampling control travels with the conversation. Left out, a pin
  // taken in one chat would fix the draw for every other chat opened on the same model,
  // and reopening the first could not bring its own value back.
  assert.ok(THREAD_SCOPED_PARAM_KEYS.includes("seed"));
  assert.equal(isThreadScopedSettingKey("seed"), true);
});

test("a chat stores a cleared seed rather than dropping the key", () => {
  assert.deepEqual(sanitizeThreadScopedSettings({ seed: null }), { seed: null });
  // The other half of that: a snapshot written before the field existed carries no seed,
  // and the thread response now says so rather than spelling it null, so the chat falls
  // through to the pin it inherits instead of reading as one that cleared it.
  assert.deepEqual(sanitizeThreadScopedSettings({ topK: 40 }), { topK: 40 });
  assert.deepEqual(sanitizeThreadScopedSettings({ seed: 3407 }), { seed: 3407 });
  // The bound the panel and the installation copy already share, applied to the snapshot
  // a chat writes: a row from another client meets only this.
  for (const bad of [-1, 1.5, MAX_SAMPLING_SEED + 1, true, "3407"]) {
    assert.deepEqual(sanitizeThreadScopedSettings({ seed: bad }), {}, String(bad));
  }
});

test("a cleared seed is not read as a missing key", () => {
  const store = read("../src/features/chat/stores/chat-runtime-store.ts");
  const helper = slice(store, "function firstSetThreadScopedValue", "\n}");
  // `??` would fall through a cleared seed's null to the installation default, putting
  // the pin back on the one chat that dropped it. Only undefined means "not set".
  assert.match(helper, /values\.find\(\(value\) => value !== undefined\)/);
});

// A model outside /api/models/list gets its models[] summary minted from a load or
// status response, and the seed gate reads isMlx off that summary. Four separate places
// mint or refresh one, and three of them were each missing the flag, so this is a rule
// over the capability cluster rather than a spot check per place.
// Compile-time, not runtime: a row omitting one of the four fails to build at the site
// that mints it, including sites no test knows about. ChatModelSummary keeps them optional.
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

test("a models[] row states every flag the seed gate reads", () => {
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
  const panel = read("../src/features/chat/chat-settings-sheet.tsx");
  // Clicking Save blurs the box during mousedown, but React has not re-rendered by the
  // time onClick runs, so a save reading `params` writes the pre-entry seed and, over an
  // otherwise-unmodified preset, the button is still disabled and the click does nothing.
  assert.match(panel, /params: toPresetParams\(paramsWithCommittedSeed\)/);
  const unsaved = slice(panel, "const hasUnsavedPresetChanges", "const presetSaveState");
  assert.match(unsaved, /isSamePresetConfig\(\s*activePresetDefinition\.params,\s*paramsWithCommittedSeed,/);
  assert.match(unsaved, /committedSeed !== \(params\.seed \?\? null\)/);
  // and blur commits the same value, so the two cannot answer differently
  assert.match(panel, /setSeedDraft\(null\);\s*setSeed\(committedSeed\);/);
});
