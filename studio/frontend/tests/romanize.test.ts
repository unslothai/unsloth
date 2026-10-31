// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { romanizeToLatin } from "../src/features/chat/adapters/romanize.ts";

test("romanizes the reported Urdu example", () => {
  assert.equal(romanizeToLatin("آپ کیسے ہیں؟"), "aap kaise hain?");
});

test("romanizes Hindi and Russian without touching Latin text", () => {
  assert.equal(romanizeToLatin("आप कैसे हैं?"), "aap kaise hain?");
  assert.equal(romanizeToLatin("привет мир"), "privet mir");
  assert.equal(romanizeToLatin("already latin"), "already latin");
});

test("leaves scripts it cannot romanize unchanged", () => {
  assert.equal(romanizeToLatin("こんにちは"), "こんにちは");
});
