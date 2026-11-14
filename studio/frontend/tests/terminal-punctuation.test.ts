// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { ensureTerminalPunctuation } from "../src/features/voice-typing/terminal-punctuation.ts";

test("turns a question full stop into a question mark", () => {
  assert.equal(
    ensureTerminalPunctuation("How to switch Git branch."),
    "How to switch Git branch?",
  );
  assert.equal(
    ensureTerminalPunctuation("What time does the train leave."),
    "What time does the train leave?",
  );
  assert.equal(
    ensureTerminalPunctuation("Is the build ready."),
    "Is the build ready?",
  );
  assert.equal(
    ensureTerminalPunctuation("Could you open the settings window."),
    "Could you open the settings window?",
  );
  assert.equal(
    ensureTerminalPunctuation("Can it be fixed."),
    "Can it be fixed?",
  );
  assert.equal(
    ensureTerminalPunctuation("Does this compile."),
    "Does this compile?",
  );
});

test("adds a question mark when the model drops the final mark", () => {
  assert.equal(
    ensureTerminalPunctuation("How to switch Git branch"),
    "How to switch Git branch?",
  );
  assert.equal(
    ensureTerminalPunctuation("Why not ask the maintainer"),
    "Why not ask the maintainer?",
  );
  assert.equal(
    ensureTerminalPunctuation("What about the release notes"),
    "What about the release notes?",
  );
});

test("keeps statements, imperatives, and exclamations as full stops", () => {
  assert.equal(
    ensureTerminalPunctuation("The branch is called main."),
    "The branch is called main.",
  );
  assert.equal(
    ensureTerminalPunctuation("Do it before noon."),
    "Do it before noon.",
  );
  assert.equal(
    ensureTerminalPunctuation("Have a pleasant afternoon."),
    "Have a pleasant afternoon.",
  );
  assert.equal(
    ensureTerminalPunctuation("What a relief."),
    "What a relief.",
  );
  assert.equal(
    ensureTerminalPunctuation("How lovely the garden looks."),
    "How lovely the garden looks.",
  );
});

test("fixes only the last sentence", () => {
  assert.equal(
    ensureTerminalPunctuation("The build is ready. How to switch Git branch."),
    "The build is ready. How to switch Git branch?",
  );
});

test("preserves a closing quote or bracket", () => {
  assert.equal(
    ensureTerminalPunctuation('She asked, "How to switch Git branch."'),
    'She asked, "How to switch Git branch?"',
  );
});
