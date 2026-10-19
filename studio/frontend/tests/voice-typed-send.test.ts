// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isTypedVoiceSubmit } from "../src/features/chat/voice/typed-send.ts";
import {
  registerVoiceTypedSubmit,
  requestVoiceTypedSubmit,
} from "../src/features/chat/voice/voice-loop-bridge.ts";

const SRC = new URL("../src/", import.meta.url);
const read = (relative: string) => readFileSync(new URL(relative, SRC), "utf8");

// The composer's voice branch of handleSubmit, with the engine's handler behind it.
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

test("the composer asks the loop to send typed text before ending dictation", () => {
  const src = read("components/assistant-ui/thread.tsx");
  const branch = src.slice(src.indexOf("if (voiceEngaged && isDictating) {"));
  const typed = branch.indexOf("requestVoiceTypedSubmit(composerText)");
  const stop = branch.indexOf("aui.composer().stopDictation()");
  assert.ok(typed > 0 && typed < stop);
});

test("the engine registers the typed send, cancels the session and skips the interject model", () => {
  const src = read("features/chat/voice/voice-engine.tsx");
  assert.match(src, /registerVoiceTypedSubmit\(\(box\) =>/);
  assert.match(src, /cancelVoiceSession\(\);\s*submitTranscript\(box, true\)/);
  assert.match(src, /if \(!typed && thread\.getState\(\)\.isRunning && latestAssistantIsReasoning\(\)\)/);
});
