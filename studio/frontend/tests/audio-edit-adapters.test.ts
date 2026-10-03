// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// fixtures/audio-edit-requests.json is shared with the backend's test_audio_edit_requests.py.

import assert from "node:assert/strict";
import test from "node:test";

import { readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  DOTS_ANGLE_BRACKETS,
  DOTS_BAD_CHARACTERS,
  EDIT_ADAPTERS,
  EDIT_TOO_LONG,
  FIRERED_INSERT_AT_END,
  FIRERED_TOO_MANY_CHANGES,
  buildEditRun,
  deliveryInstructions,
  editAdapterFor,
  editPhaseLabel,
  markupFor,
} = await import("../src/features/audio/edit-adapters.ts");
const { buildAudioRunBody } = await import(
  "../src/features/audio/audio-run-request.ts"
);

interface FixtureCase {
  name: string;
  family: string;
  input: {
    original: string;
    edited: string;
    mode: "words" | "delivery";
    delivery?: { speed: number; pitchSteps: number };
    advanced: Record<string, boolean | number | string>;
  };
  run_body: Record<string, unknown>;
}

const fixture = JSON.parse(readText("fixtures/audio-edit-requests.json")) as {
  cases: FixtureCase[];
};
const SOURCE = { input_id: "in_0123456789abcdef" };
const S2 = "Okay, I'm Cemo and what you just heard wasn't a human voice.";

const runFor = (family: string, input: FixtureCase["input"]) =>
  buildEditRun(EDIT_ADAPTERS[family], {
    source: SOURCE,
    transcript: input.original,
    edited: input.edited,
    mode: input.mode,
    delivery: input.delivery ?? null,
    advanced: input.advanced,
  });

for (const testCase of fixture.cases) {
  test(`golden: ${testCase.name}`, () => {
    const adapter = EDIT_ADAPTERS[testCase.family];
    assert.ok(adapter, testCase.family);
    assert.equal(
      adapter.validateWords(testCase.input.original, testCase.input.edited),
      null,
    );
    const run = runFor(testCase.family, testCase.input);
    assert.deepEqual(run, testCase.run_body);
    assert.deepEqual(buildAudioRunBody(run), testCase.run_body);
  });
}

test("Advanced options reach the body minus the adapter's claims", () => {
  const run = buildEditRun(EDIT_ADAPTERS.firered_audio, {
    source: SOURCE,
    transcript: S2,
    edited: S2.replace("human", "robot"),
    mode: "words",
    advanced: {
      num_inference_steps: 4,
      guidance_scale: 1.2,
      flag: true,
      template_name: "x",
      instruction: "Speak faster.",
    },
  });
  assert.deepEqual(run.options, {
    num_inference_steps: 4,
    guidance_scale: 1.2,
    flag: true,
  });
});

test("FireRedAudio cannot insert at the very end", () => {
  assert.equal(
    EDIT_ADAPTERS.firered_audio.validateWords(S2, `${S2} Really.`),
    FIRERED_INSERT_AT_END,
  );
  assert.equal(EDIT_ADAPTERS.dots_tts.validateWords(S2, `${S2} Really.`), null);
});

test("FireRedAudio takes at most five changes", () => {
  const original =
    "one two three four five six seven eight nine ten eleven twelve";
  const five = "ONE two THREE four FIVE six SEVEN eight NINE ten eleven twelve";
  assert.equal(EDIT_ADAPTERS.firered_audio.validateWords(original, five), null);
  const six = five.replace("eleven", "ELEVEN");
  assert.equal(
    EDIT_ADAPTERS.firered_audio.validateWords(original, six),
    FIRERED_TOO_MANY_CHANGES,
  );
  assert.equal(EDIT_ADAPTERS.dots_tts.validateWords(original, six), null);
});

test("DotTTS refuses quotes and angle brackets the markup cannot carry", () => {
  for (const word of ['"robot"', "<robot>", "ro>bot"]) {
    assert.equal(
      EDIT_ADAPTERS.dots_tts.validateWords(S2, S2.replace("human", word)),
      DOTS_BAD_CHARACTERS,
      word,
    );
  }
  assert.equal(
    EDIT_ADAPTERS.dots_tts.validateWords(
      `<b> ${S2}`,
      `<b> ${S2.replace("human", "robot")}`,
    ),
    DOTS_ANGLE_BRACKETS,
  );
  const quoted = S2.replace("human", '"robot"');
  assert.equal(EDIT_ADAPTERS.firered_audio.validateWords(S2, quoted), null);
});

test("past the word cap every adapter says so", () => {
  const long = Array.from({ length: 401 }, (_, i) => `w${i}`).join(" ");
  for (const adapter of Object.values(EDIT_ADAPTERS)) {
    assert.equal(adapter.validateWords(long, `${long} x`), EDIT_TOO_LONG);
  }
});

test("Vevo2 sends neither markup nor instructions, and drops an empty transcript", () => {
  const run = buildEditRun(EDIT_ADAPTERS.vevo2, {
    source: { clip_id: "c1" },
    transcript: "  ",
    edited: "A robot voice.",
    mode: "words",
  });
  assert.deepEqual(run.edit, { mode: "words" });
  assert.equal(run.inputs?.reference_text, undefined);
});

test("Delivery is FireRedAudio's only, speed then pitch, skipping what is unchanged", () => {
  assert.deepEqual(deliveryInstructions(1, 0), []);
  assert.deepEqual(deliveryInstructions(2, 3), [
    "adjust the speed to 2x",
    "shift the pitch by 3 steps",
  ]);
  assert.deepEqual(deliveryInstructions(0.7000000000000001, 0), [
    "adjust the speed to 0.7x",
  ]);
  const pitchOnly = buildEditRun(EDIT_ADAPTERS.firered_audio, {
    transcript: "",
    edited: "",
    mode: "delivery",
    delivery: { speed: 1, pitchSteps: 2 },
    sourceName: "take-3.wav",
  });
  assert.equal(pitchOnly.text, "take-3.wav");
  assert.deepEqual(pitchOnly.edit, { mode: "delivery", pitch_steps: 2 });
  assert.equal(pitchOnly.inputs?.source, undefined);
  // Delivery on a model without it falls back to Words.
  const dots = buildEditRun(EDIT_ADAPTERS.dots_tts, {
    transcript: S2,
    edited: S2,
    mode: "delivery",
    delivery: { speed: 1.5, pitchSteps: 3 },
  });
  assert.deepEqual(dots.edit, { mode: "words" });
});

test("the adapter follows the loaded model's family and edit workflow", () => {
  const ctx = (audioFamily: string | null, audioWorkflows?: string[]) => ({
    audioFamily,
    audioWorkflows,
  });
  for (const family of ["dots_tts", "vevo2", "firered_audio"]) {
    assert.equal(editAdapterFor(ctx(family, ["edit"])), EDIT_ADAPTERS[family]);
  }
  // DotTTS-MF is dots_tts but cannot edit.
  assert.equal(editAdapterFor(ctx("dots_tts", ["speak"])), null);
  assert.equal(editAdapterFor(ctx("qwen3_tts", ["clone", "edit"])), null);
  assert.equal(editAdapterFor(ctx("toString", ["edit"])), null);
  assert.equal(editAdapterFor(ctx("vevo2")), null);
});

test("markup and the chained-run progress line", () => {
  assert.equal(markupFor("a b c", "a b c"), "a b c");
  assert.equal(markupFor("a b", "x a b"), "<ins>x</ins> a b");
  const fireRun = runFor("firered_audio", fixture.cases[4].input);
  assert.equal(
    editPhaseLabel(EDIT_ADAPTERS.firered_audio, fireRun),
    "Applying 2 changes, one pass each",
  );
  const dotsRun = runFor("dots_tts", fixture.cases[0].input);
  assert.equal(editPhaseLabel(EDIT_ADAPTERS.dots_tts, dotsRun), null);
});
