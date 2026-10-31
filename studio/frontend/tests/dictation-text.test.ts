// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  appendTranscriptText,
  applySpokenPunctuation,
  cleanDictationText,
  collapseStutters,
  joinTranscriptParts,
  removeFillerWords,
  stripShortPausePeriod,
} from "../src/features/chat/adapters/dictation-text.ts";

test("short pause periods are dropped without touching real sentences", () => {
  assert.equal(stripShortPausePeriod("Thanks."), "Thanks");
  assert.equal(stripShortPausePeriod("Okay."), "Okay");
  assert.equal(
    stripShortPausePeriod("This is a longer sentence."),
    "This is a longer sentence.",
  );
  assert.equal(stripShortPausePeriod("Are you coming?"), "Are you coming?");
});

test("filler words go without taking the sentence's capital with them", () => {
  assert.equal(
    removeFillerWords("Um, so I think uh this is fine", "en"),
    "So I think this is fine",
  );
  // A lowercase filler had no capital to hand on; the model's casing is kept.
  assert.equal(removeFillerWords("hmm let me see", "de"), "let me see");
});

test("a filler that is a real word in another language is kept", () => {
  // "um" is Portuguese for "a/an"; without English evidence it is text.
  assert.equal(removeFillerWords("um livro", "pt"), "um livro");
  assert.equal(removeFillerWords("um livro", ""), "um livro");
  assert.equal(removeFillerWords("um book", "en"), "book");
});

test("filler removal can be switched off", () => {
  assert.equal(removeFillerWords("uh hello", "en", false), "uh hello");
});

test("repeated stutters collapse but deliberate pairs stay", () => {
  assert.equal(collapseStutters("I I I think that that works"), "I think that that works");
  assert.equal(collapseStutters("very very good"), "very very good");
  assert.equal(collapseStutters("I I I think\nso so so"), "I think\nso");
});

test("spoken punctuation becomes marks and capitalizes the next sentence", () => {
  assert.equal(
    applySpokenPunctuation("hello comma world period"),
    "Hello, world.",
  );
  assert.equal(
    applySpokenPunctuation("is it ready question mark yes"),
    "Is it ready? Yes",
  );
  assert.equal(
    applySpokenPunctuation("first line new line second line"),
    "First line\nSecond line",
  );
  assert.equal(
    applySpokenPunctuation("open quote hello close quote"),
    '"Hello"',
  );
});

test("a period in the middle of a sentence is left alone", () => {
  // "period" is an ordinary English word; only the final one is punctuation.
  assert.equal(
    applySpokenPunctuation("this period is important"),
    "This period is important",
  );
});

test("the full cleanup pipeline composes filler and punctuation rules", () => {
  assert.equal(
    cleanDictationText("Um, hello comma world period", {
      language: "en",
      removeFillers: true,
      spokenPunctuation: true,
    }),
    "Hello, world.",
  );
  assert.equal(
    cleanDictationText("hello comma world", {
      removeFillers: true,
    }),
    "hello comma world",
  );
  // The stutter pass must not flatten the break a command just inserted.
  assert.equal(
    cleanDictationText("first line new line second line", {
      spokenPunctuation: true,
    }),
    "First line\nSecond line",
  );
  assert.equal(
    cleanDictationText("first line new paragraph second line", {
      spokenPunctuation: true,
    }),
    "First line\n\nSecond line",
  );
});

test("a segment that is only a break survives cleanup", () => {
  assert.equal(
    cleanDictationText("new paragraph", { spokenPunctuation: true }),
    "\n\n",
  );
  assert.equal(
    cleanDictationText("hello new paragraph", { spokenPunctuation: true }),
    "Hello\n\n",
  );
});

test("segments join without a space before leading punctuation", () => {
  assert.equal(joinTranscriptParts(["Hello", ", world", "again"]), "Hello, world again");
  assert.equal(joinTranscriptParts(["", "  first  ", "second"]), "first second");
  assert.equal(joinTranscriptParts([]), "");
  assert.equal(joinTranscriptParts(["Hello", "\n\n", "world"]), "Hello\n\nworld");
  assert.equal(joinTranscriptParts(["\n\n", "Hello"]), "Hello");
  assert.equal(joinTranscriptParts(["Hello.", "world"]), "Hello world");
  assert.equal(joinTranscriptParts(["Hello.", "World"]), "Hello. World");
});

test("a draft keeps the whitespace the user typed", () => {
  assert.equal(appendTranscriptText("", "hello"), "hello");
  assert.equal(appendTranscriptText("hello", "world"), "hello world");
  assert.equal(appendTranscriptText("hello ", "world"), "hello world");
  assert.equal(appendTranscriptText("hello\n", "world"), "hello\nworld");
  assert.equal(appendTranscriptText("hello", ", world"), "hello, world");
  assert.equal(appendTranscriptText("hello", "  "), "hello");
  assert.equal(appendTranscriptText("hello", "\n\nworld"), "hello\n\nworld");
  assert.equal(appendTranscriptText("hello", "world\n\n"), "hello world\n\n");
  assert.equal(appendTranscriptText("hello\n", "\n\nworld"), "hello\nworld");
  assert.equal(appendTranscriptText("", "\n\n"), "");
});
