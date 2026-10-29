// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// Read as source: importing would drag in the stores and the toast layer.
const source = readSrc("features/chat/api/chat-adapter.ts");

// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const TOAST_BRANCH =
  /err instanceof GenerationLengthError\) \{[\s\S]*?\n {10}\} else if/;
const FROM_THE_ERROR = /description:\s*\n?\s*msg \|\|/;
const THE_ERROR_CLASS = /class GenerationLengthError/;
const CAP_REMEDY = /Increase Max Tokens or disable thinking/;
const WINDOW_REMEDY = /cannot create room the window does not have/;
const NO_UNLIMITED_CLAIM = /already unlimited/;
const WINDOW_SETTING = /Length in Model settings/;
const HIDDEN_WINDOW_REMEDY =
  /Start a new chat, or shorten this one, then retry/;
const BOTH_REMEDIES = /Max Tokens or its context window before answering/;

test("the toast repeats the advice the error chose, not the Max Tokens advice", () => {
  const branch = TOAST_BRANCH.exec(source);
  assert.ok(branch, "the GenerationLengthError toast branch moved");
  assert.match(branch[0], FROM_THE_ERROR);
});

test("the two remedies really are different text, so passing it through matters", () => {
  const chatApi = readSrc("features/chat/api/chat-api.ts");

  assert.match(chatApi, THE_ERROR_CLASS);
  assert.match(chatApi, CAP_REMEDY);
  assert.match(chatApi, WINDOW_REMEDY);
  assert.match(chatApi, WINDOW_SETTING);
  assert.doesNotMatch(chatApi, NO_UNLIMITED_CLAIM);
  assert.match(chatApi, HIDDEN_WINDOW_REMEDY);
  assert.match(chatApi, BOTH_REMEDIES);
});
