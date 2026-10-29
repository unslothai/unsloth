// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// fixtures/audio-edit-requests.json is shared with the backend's test_audio_edit_requests.py.

import assert from "node:assert/strict";
import test from "node:test";

import { readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const A = await import("../src/features/audio/edit-adapters.ts");
const D = await import("../src/features/audio/edit-diff.ts");
const { EDIT_ADAPTERS } = A;
const R = await import("../src/features/audio/audio-run-request.ts");

type RunInput = Parameters<typeof A.buildEditRun>[1];
type Input = RunInput & { original: string };
const { cases } = JSON.parse(readText("fixtures/audio-edit-requests.json")) as {
  cases: { name: string; family: string; input: Input; run_body: object }[];
};
const S2 = "Okay, I'm Cemo and what you just heard wasn't a human voice.";
const sub = (word: string) => S2.replace("human", word);
const LONG = Array.from(
  { length: D.EDIT_DIFF_MAX_WORDS + 1 },
  (_, i) => `w${i}`,
).join(" ");
const TWELVE = "one two three four five six seven eight nine ten eleven twelve";
const FIVE = "ONE two THREE four FIVE six SEVEN eight NINE ten eleven twelve";
const SIX = FIVE.replace("eleven", "ELEVEN");

const words = (text: string) => (text ? text.split(" ") : []);
const change = (
  kind: string,
  old: string,
  add: string,
  before: string | null,
  index: number,
) => ({ kind, old: words(old), new: words(add), before, index });

const build = (family: string, input: Partial<RunInput>) =>
  A.buildEditRun(EDIT_ADAPTERS[family], {
    transcript: S2,
    edited: S2,
    mode: "words",
    ...input,
  });
const runFor = (family: string, { original, ...input }: Input) =>
  build(family, {
    ...input,
    source: { input_id: "in_0123456789abcdef" },
    transcript: original,
    delivery: input.delivery ?? null,
  });

for (const { name, family, input, run_body } of cases) {
  test(`golden: ${name}`, () => {
    const adapter = EDIT_ADAPTERS[family];
    assert.equal(adapter.validateWords(input.original, input.edited), null);
    const run = runFor(family, input);
    assert.deepEqual(run, run_body);
    assert.deepEqual(R.buildAudioRunBody(run), run_body);
  });
}

test("each family's word checks", () => {
  const rows: [string, string, string, string | null][] = [
    ["firered_audio", S2, `${S2} Really.`, A.FIRERED_INSERT_AT_END],
    ["dots_tts", S2, `${S2} Really.`, null],
    ["firered_audio", TWELVE, FIVE, null],
    ["firered_audio", TWELVE, SIX, A.FIRERED_TOO_MANY_CHANGES],
    ["dots_tts", TWELVE, SIX, null],
    ["dots_tts", S2, sub('"robot"'), A.DOTS_BAD_CHARACTERS],
    ["dots_tts", S2, sub("<robot>"), A.DOTS_BAD_CHARACTERS],
    ["dots_tts", S2, sub("ro>bot"), A.DOTS_BAD_CHARACTERS],
    ["dots_tts", `<b> ${S2}`, `<b> ${sub("robot")}`, A.DOTS_ANGLE_BRACKETS],
    ["firered_audio", S2, sub('"robot"'), null],
  ];
  for (const [family, original, edited, reason] of rows) {
    const got = EDIT_ADAPTERS[family].validateWords(original, edited);
    assert.equal(got, reason, `${family}: ${edited}`);
  }
  for (const adapter of Object.values(EDIT_ADAPTERS)) {
    assert.equal(adapter.validateWords(LONG, `${LONG} x`), A.EDIT_TOO_LONG);
  }
});

test("run bodies: Advanced minus claims, Vevo2, and Delivery", () => {
  const advanced = { num_inference_steps: 4, guidance_scale: 1.2, flag: true };
  const claimed = { template_name: "x", instruction: "Speak faster." };
  const fire = build("firered_audio", {
    edited: sub("robot"),
    advanced: { ...advanced, ...claimed },
  });
  assert.deepEqual(fire.options, advanced);
  const vevo = build("vevo2", { source: { clip_id: "c1" }, transcript: "  " });
  assert.deepEqual(vevo.edit, { mode: "words" });
  assert.equal(vevo.inputs?.reference_text, undefined);
  const pitchOnly = build("firered_audio", {
    transcript: "",
    edited: "",
    mode: "delivery",
    delivery: { speed: 1, pitchSteps: 2 },
    sourceName: "take-3.wav",
  });
  assert.equal(pitchOnly.text, "take-3.wav");
  assert.deepEqual(pitchOnly.edit, { mode: "delivery", pitch_steps: 2 });
  assert.equal(pitchOnly.inputs?.source, undefined);
  const dots = build("dots_tts", {
    mode: "delivery",
    delivery: { speed: 1.5, pitchSteps: 3 },
  });
  assert.deepEqual(dots.edit, { mode: "words" });
  assert.deepEqual(A.deliveryInstructions(1, 0), []);
  assert.deepEqual(A.deliveryInstructions(2, 3), [
    "adjust the speed to 2x",
    "shift the pitch by 3 steps",
  ]);
  assert.deepEqual(A.deliveryInstructions(0.7000000000000001, 0), [
    "adjust the speed to 0.7x",
  ]);
});

test("the adapter follows the loaded model's family and edit workflow", () => {
  const rows: [string, string[] | undefined, boolean][] = [
    ["dots_tts", ["edit"], true],
    ["vevo2", ["edit"], true],
    ["firered_audio", ["edit"], true],
    ["dots_tts", ["speak"], false],
    ["qwen3_tts", ["clone", "edit"], false],
    ["toString", ["edit"], false],
    ["vevo2", undefined, false],
  ];
  for (const [audioFamily, audioWorkflows, found] of rows) {
    const adapter = A.editAdapterFor({ audioFamily, audioWorkflows });
    assert.equal(adapter, found ? EDIT_ADAPTERS[audioFamily] : null);
  }
});

test("markup and the chained-run progress line", () => {
  assert.equal(A.markupFor("a b c", "a b c"), "a b c");
  assert.equal(A.markupFor("a b", "x a b"), "<ins>x</ins> a b");
  const fire = runFor("firered_audio", cases[4].input);
  assert.equal(
    A.editPhaseLabel(EDIT_ADAPTERS.firered_audio, fire),
    "Applying 2 changes, one pass each",
  );
  const dots = runFor("dots_tts", cases[0].input);
  assert.equal(A.editPhaseLabel(EDIT_ADAPTERS.dots_tts, dots), null);
});

test("words are whitespace tokens; punctuation stays on its word", () => {
  assert.deepEqual(D.tokenizeWords("  Okay,  I'm\nCemo. "), [
    "Okay,",
    "I'm",
    "Cemo.",
  ]);
  assert.deepEqual(D.tokenizeWords("   "), []);
  assert.equal(D.countChanges(S2, `  ${S2.replaceAll(" ", "   ")} `), 0);
  assert.equal(D.countChanges("", ""), 0);
});

test("changes group per run of words, anchored on the next word", () => {
  const s2 = (from: string, to: string) => S2.replace(from, to);
  const rows: [string, ReturnType<typeof change>[], string?][] = [
    [s2("voice.", "voice!"), [change("replace", "voice.", "voice!", null, 11)]],
    [s2("human", "robot"), [change("replace", "human", "robot", "voice.", 10)]],
    [
      s2("just heard", "heard really"),
      [
        change("delete", "just", "", "heard", 6),
        change("insert", "", "really", "wasn't", 8),
      ],
    ],
    ["a d", [change("delete", "b c", "", "d", 1)], "a b c d"],
    ["a b x y", [change("insert", "", "x y", null, 2)], "a b"],
  ];
  for (const [b, changes, a = S2] of rows) {
    assert.deepEqual(D.changesBetween(a, b), changes, b);
  }
  const ops = D.diffWords("a human voice.", "a robot sound.");
  assert.ok(ops);
  assert.deepEqual(D.groupChanges(ops), [
    change("replace", "human voice.", "robot sound.", null, 1),
  ]);
});

test("past the word cap the diff is not computed", () => {
  assert.equal(D.diffWords("short", LONG), null);
  assert.equal(D.countChanges(LONG, LONG), null);
  assert.equal(D.diffSegments(LONG, "x"), null);
  const atCap = LONG.split(" ").slice(1).join(" ");
  assert.equal(D.countChanges(atCap, atCap), 0);
});

test("segments carry the inserted and deleted labels for screen readers", () => {
  assert.deepEqual(D.diffSegments("a human voice", "a robot voice really"), [
    { kind: "equal", text: "a", srLabel: null },
    { kind: "delete", text: "human", srLabel: "deleted " },
    { kind: "insert", text: "robot", srLabel: "inserted " },
    { kind: "equal", text: "voice", srLabel: null },
    { kind: "insert", text: "really", srLabel: "inserted " },
  ]);
});

test("DotTTS says so before its markup would outgrow the backend cap", () => {
  const words = Array.from({ length: 400 }, (_, i) => `word${i}`);
  const original = words.join(" ");
  const edited = words.map((w, i) => (i % 2 ? `${w}x` : w)).join(" ");
  assert.equal(
    EDIT_ADAPTERS.dots_tts.validateWords(original, edited),
    A.DOTS_MARKUP_TOO_LONG,
  );
  assert.equal(
    EDIT_ADAPTERS.dots_tts.validateWords(original, original.replace("word1 ", "word1x ")),
    null,
  );
});
