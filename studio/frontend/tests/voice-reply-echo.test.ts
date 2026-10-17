// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A transcript that reads back the reply's own words is the speaker leaking, not a turn.

import assert from "node:assert/strict";
import test from "node:test";

import { isReplyEcho } from "../src/features/chat/voice/turn-complete.ts";

const reply = "I can't \"hear\" you in the literal sense, but I can see your message. What's on your mind?";

test("a run of the reply's words is an echo", () => {
  assert.equal(isReplyEcho("What's on your mind", reply), true);
  assert.equal(isReplyEcho("see your message. What's on", reply), true);
});

test("the user's own words are not", () => {
  assert.equal(isReplyEcho("what's the weather like", reply), false);
  assert.equal(isReplyEcho("", reply), false);
  assert.equal(isReplyEcho("hello", ""), false);
});
