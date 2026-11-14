// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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

test("a one-word command the reply also says is still the user's", () => {
  const long = "Sure. You can stop the server any time. Yes, it keeps your chats.";
  assert.equal(isReplyEcho("Stop.", long), false);
  assert.equal(isReplyEcho("yes", long), false);
  assert.equal(isReplyEcho("stop the server", long), true);
  assert.equal(isReplyEcho("Sure.", long), true);
});

test("a word is matched whole, not inside a longer one", () => {
  const reply = "They can help again with skipping to the next step.";
  assert.equal(isReplyEcho("Hey.", reply), false);
  assert.equal(isReplyEcho("Skip.", reply), false);
  assert.equal(isReplyEcho("can help", reply), true);
  assert.equal(isReplyEcho("an help", reply), false);
});

test("an answer picked from the reply's closing question is the user's, not echo", () => {
  const reply = "I can write it either way. Which language should I use, Python or JavaScript?";
  assert.equal(isReplyEcho("Python", reply), false);
  assert.equal(isReplyEcho("javascript please", reply), false);
  assert.equal(isReplyEcho("I can write it either way", reply), true);
  assert.equal(isReplyEcho("either way", "I can write it either way."), true);
});
