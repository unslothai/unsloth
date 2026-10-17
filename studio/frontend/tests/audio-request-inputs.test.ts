// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { audioModelContextFor, readsRequestInput } = await import(
  "../src/features/audio/tools/select.ts"
);

const page = {
  musicGeneration: false,
  cudaMusicGeneration: false,
  musicNeedsDescription: false,
};
const inputsOf = (audioRequestInputs: unknown) =>
  audioModelContextFor({ audio_request_inputs: audioRequestInputs as [] }, page)
    .requestInputs;

test("audio_request_inputs reaches the context; anything but a list is unknown", () => {
  for (const unknown of [undefined, null, "instruct"])
    assert.equal(inputsOf(unknown), null);
  assert.deepEqual(inputsOf([]), []);
  assert.deepEqual(inputsOf(["instruct", "language"]), [
    "instruct",
    "language",
  ]);
});

test("a model that may read an input keeps its field; a strict one without it loses it", () => {
  const reads = (requestInputs: string[] | null, ...names: string[]) =>
    readsRequestInput({ requestInputs }, ...names);
  const description = ["instruct", "instruction"];
  // Non-strict specs (null) keep both; Kokoro ([]) has no description, F5 no language.
  assert.equal(reads(null, ...description) && reads(null, "language"), true);
  assert.equal(reads([], ...description) || reads([], "language"), false);
  assert.equal(reads(["instruction"], ...description), true);
  assert.equal(reads(["instruct"], "language"), false);
  assert.equal(reads(["language"], "language"), true);
});

test("the voice description and Clone's language follow the gate", () => {
  const registry = readSrc("features/audio/tools/registry.tsx");
  assert.match(
    registry,
    /kind !== "voice" \|\| readsRequestInput\(ctx, "instruct", "instruction"\)/,
  );
  const clone = readSrc("features/audio/hooks/use-clone-generation.ts");
  assert.match(
    clone,
    /languageShown = readsRequestInput\(toolContext, "language"\)/,
  );
  assert.match(
    clone,
    /language: languageShown \? state\.language \|\| null : null/,
  );
  assert.match(
    readSrc("features/audio/pages/clone-page.tsx"),
    /\{clone\.languageShown \? \(\s*<LanguageSelect/,
  );
});
