// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { containsAbusiveLanguage } from "../src/features/voice-typing/abusive-language.ts";

test("detects strong abuse without flagging ordinary words", () => {
  assert.equal(containsAbusiveLanguage("this is fucking awful"), true);
  assert.equal(containsAbusiveLanguage("shiiiit happened"), true);
  assert.equal(containsAbusiveLanguage("this is a normal class"), false);
  assert.equal(containsAbusiveLanguage("I have 13 cents"), false);
  assert.equal(containsAbusiveLanguage("piece of shit"), true);
});
