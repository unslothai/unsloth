// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isTypedVoiceSubmit } from "../src/features/chat/voice/typed-send.ts";
import {
  registerVoiceTypedSubmit,
  requestVoiceTypedSubmit,
} from "../src/features/chat/voice/voice-loop-bridge.ts";

function composer(state: { paused: boolean; held?: string; parked?: string }) {
  const sent: string[] = [];
  let stopped = 0;
  registerVoiceTypedSubmit((box) => {
    const shown = [state.parked ?? "", state.held ?? ""].filter(Boolean).join(" ");
    if (!isTypedVoiceSubmit({ box, shown, paused: state.paused })) return false;
    sent.push(box.trim());
    return true;
  });
  const enter = (box: string) => {
    if (requestVoiceTypedSubmit(box)) return;
    stopped += 1;
  };
  return { sent, enter, stops: () => stopped };
}

test("paused + typed Enter sends the box", () => {
  const c = composer({ paused: true });
  c.enter("what time is it in Tokyo");
  assert.deepEqual(c.sent, ["what time is it in Tokyo"]);
  assert.equal(c.stops(), 0);
  registerVoiceTypedSubmit(null);
});

test("typed text while listening sends instead of only ending the spoken turn", () => {
  const c = composer({ paused: false });
  c.enter("hello there");
  assert.deepEqual(c.sent, ["hello there"]);
  registerVoiceTypedSubmit(null);
});

test("typing onto a held fragment sends the merged box", () => {
  const c = composer({ paused: false, held: "so what about" });
  c.enter("so what about the weather tomorrow");
  assert.deepEqual(c.sent, ["so what about the weather tomorrow"]);
  registerVoiceTypedSubmit(null);
});

test("Enter on only what voice put in the box still ends the spoken turn", () => {
  const c = composer({ paused: false, held: "so what about", parked: "hi" });
  c.enter("hi so what about");
  c.enter("");
  assert.deepEqual(c.sent, []);
  assert.equal(c.stops(), 2);
  registerVoiceTypedSubmit(null);
});

test("an unmounted loop leaves the old end-the-turn behaviour", () => {
  registerVoiceTypedSubmit(null);
  assert.equal(requestVoiceTypedSubmit("typed"), false);
});
