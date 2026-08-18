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
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
  MAX_SAMPLING_SEED,
  modelReadsSamplingSeed,
} from "../src/features/chat/types/runtime.ts";

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
  // number a pin cannot name. Stopping at int32 max instead would put every seed above
  // 2^31-1 out of reach, and a run whose server-drawn seed landed there could not be replayed.
  assert.equal(MAX_SAMPLING_SEED, 0xffffffff - 1);
});

test("a seed reaches only the backend that reads one", () => {
  // A loaded variant, a GGUF context, or a direct .gguf file each mean llama-server.
  assert.ok(modelReadsSamplingSeed("Q4_K_M", null, "unsloth/Qwen3.5-9B-GGUF"));
  assert.ok(modelReadsSamplingSeed(null, 8192, "unsloth/Qwen3.5-9B"));
  assert.ok(modelReadsSamplingSeed(null, null, "/models/local-file.GGUF"));
  // Transformers and MLX take the field and ignore it, so neither is offered one.
  assert.ok(!modelReadsSamplingSeed(null, null, "unsloth/Llama-4-8B"));
  assert.ok(!modelReadsSamplingSeed(null, null, undefined));
});

test("the panel offers the seed only where llama-server reads it", () => {
  const sheet = read("../src/features/chat/chat-settings-sheet.tsx");
  assert.match(
    sheet,
    /const showSeed =\s*!isExternalModel &&\s*modelReadsSamplingSeed\(/,
  );
  // type="number" reports an entry the engine cannot parse as "", which would clear
  // the pin with no error. chat-providers-dialog documents the same trap.
  const field = slice(sheet, "{showSeed ? (", 'aria-label="Seed"');
  const props = field
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => !line.startsWith("//"));
  assert.ok(props.includes('type="text"'));
  assert.ok(!props.includes('type="number"'));
  // The panel and the request body answer "reaches llama-server" with the same helper, so
  // neither can drift into hiding the field while the wire still carries a pin. isGguf is
  // deliberately not that helper: it also caps Max Tokens, and the two must stay separable.
  assert.match(sheet, /modelReadsSamplingSeed\(/);
  assert.doesNotMatch(sheet, /const isGguf = modelReadsSamplingSeed\(/);
});

test("an over-long paste clamps rather than becoming another number", () => {
  const sheet = read("../src/features/chat/chat-settings-sheet.tsx");
  const handler = slice(sheet, "onChange={(e) => {", 'placeholder="Random"');
  // Truncating to 10 digits before clamping turned 12345678901234 into 1234567890.
  assert.doesNotMatch(handler, /\.slice\(0, ?10\)/);
  assert.match(handler, /digits\.length > 10\s*\?\s*MAX_SAMPLING_SEED/);
  // And the length is measured after the padding, so a pasted 0000000003407 stays 3407
  // instead of reading as 13 digits and clamping to the maximum.
  assert.match(handler, /\^0\+\(\?=\\d\)/);
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
  // A pin set on a GGUF outlives a switch to safetensors, where the panel hides the
  // field: without this the user would keep sending a seed they can no longer see.
  assert.match(adapter, /!modelReadsSamplingSeed\(/);
  assert.match(adapter, /runtime\.activeGgufVariant,/);
});
