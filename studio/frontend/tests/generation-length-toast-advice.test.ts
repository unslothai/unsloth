// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Asserted against the source like the other chat-adapter tests: importing the
// module would drag in the stores and the toast layer for one catch block.
const source = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const TOAST_BRANCH =
  /err instanceof GenerationLengthError\) \{[\s\S]*?\n {10}\} else if/;
const FROM_THE_ERROR = /description:\s*\n?\s*msg \|\|/;
const THE_ERROR_CLASS = /class GenerationLengthError/;
const CAP_REMEDY = /Increase Max Tokens or disable thinking/;
// Retargeted. The old wording claimed Max Tokens was "already unlimited", which is
// only true for one of the two cases that reach this branch: a finite cap the prompt
// left no room for lands here too, and telling that user their cap is unlimited
// contradicts the number in their own Settings.
const WINDOW_REMEDY = /cannot create room the window does not have/;
const NO_UNLIMITED_CLAIM = /already unlimited/;
const WINDOW_SETTING = /Length in Model settings/;

test("the toast repeats the advice the error chose, not the Max Tokens advice", () => {
  // GenerationLengthError already decides between the Max Tokens and the Context
  // Length remedy, from the cap and the prompt size. The toast hardcoded the Max
  // Tokens wording, which overrode that decision in the one place the user reads:
  // on a prompt that left no room, it sent them to raise a setting already at its
  // maximum, while the message body two lines away said the opposite.
  const branch = TOAST_BRANCH.exec(source);
  assert.ok(branch, "the GenerationLengthError toast branch moved");
  assert.match(branch[0], FROM_THE_ERROR);
});

test("the two remedies really are different text, so passing it through matters", () => {
  // Read rather than imported, as tests/padded-response.test.ts reads it: importing
  // chat-api pulls in the asset graph for two string literals.
  const chatApi = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    ),
    "utf8",
  );

  assert.match(chatApi, THE_ERROR_CLASS);
  assert.match(chatApi, CAP_REMEDY);
  assert.match(chatApi, WINDOW_REMEDY);
  assert.match(chatApi, WINDOW_SETTING);
  assert.doesNotMatch(chatApi, NO_UNLIMITED_CLAIM);
});
