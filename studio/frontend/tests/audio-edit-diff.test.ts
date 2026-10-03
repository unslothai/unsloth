// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const {
  EDIT_DIFF_MAX_WORDS,
  changesBetween,
  countChanges,
  diffSegments,
  diffWords,
  groupChanges,
  tokenizeWords,
} = await import("../src/features/audio/edit-diff.ts");

const S2 = "Okay, I'm Cemo and what you just heard wasn't a human voice.";

test("words are whitespace tokens; punctuation stays on its word", () => {
  assert.deepEqual(tokenizeWords("  Okay,  I'm\nCemo. "), [
    "Okay,",
    "I'm",
    "Cemo.",
  ]);
  assert.deepEqual(tokenizeWords("   "), []);
  assert.equal(countChanges(S2, `  ${S2.replaceAll(" ", "   ")} `), 0);
  assert.equal(countChanges("", ""), 0);
  const changes = changesBetween(S2, S2.replace("voice.", "voice!"));
  assert.deepEqual(changes, [
    {
      kind: "replace",
      old: ["voice."],
      new: ["voice!"],
      before: null,
      index: 11,
    },
  ]);
});

test("the S2 sentence human→robot is one replace before 'voice.'", () => {
  assert.deepEqual(changesBetween(S2, S2.replace("human", "robot")), [
    {
      kind: "replace",
      old: ["human"],
      new: ["robot"],
      before: "voice.",
      index: 10,
    },
  ]);
});

test("deletes and inserts are grouped into their own changes", () => {
  const edited =
    "Okay, I'm Cemo and what you heard really wasn't a human voice.";
  assert.deepEqual(changesBetween(S2, edited), [
    { kind: "delete", old: ["just"], new: [], before: "heard", index: 6 },
    { kind: "insert", old: [], new: ["really"], before: "wasn't", index: 8 },
  ]);
  // Several adjacent words in one run are one change.
  assert.deepEqual(changesBetween("a b c d", "a d"), [
    { kind: "delete", old: ["b", "c"], new: [], before: "d", index: 1 },
  ]);
  assert.deepEqual(changesBetween("a b", "a b x y"), [
    { kind: "insert", old: [], new: ["x", "y"], before: null, index: 2 },
  ]);
});

test("an adjacent delete and insert merge into one replace", () => {
  const ops = diffWords("a human voice.", "a robot sound.");
  assert.ok(ops);
  assert.deepEqual(groupChanges(ops), [
    {
      kind: "replace",
      old: ["human", "voice."],
      new: ["robot", "sound."],
      before: null,
      index: 1,
    },
  ]);
});

test("past the word cap the diff is not computed", () => {
  const long = Array.from(
    { length: EDIT_DIFF_MAX_WORDS + 1 },
    (_, i) => `w${i}`,
  ).join(" ");
  assert.equal(diffWords("short", long), null);
  assert.equal(countChanges(long, long), null);
  assert.equal(diffSegments(long, "x"), null);
  const atCap = long.split(" ").slice(1).join(" ");
  assert.equal(countChanges(atCap, atCap), 0);
});

test("segments carry the inserted and deleted labels for screen readers", () => {
  assert.deepEqual(diffSegments("a human voice", "a robot voice really"), [
    { kind: "equal", text: "a", srLabel: null },
    { kind: "delete", text: "human", srLabel: "deleted " },
    { kind: "insert", text: "robot", srLabel: "inserted " },
    { kind: "equal", text: "voice", srLabel: null },
    { kind: "insert", text: "really", srLabel: "inserted " },
  ]);
});
