// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isRevealPrefix, isTypedVoiceSubmit } from "../src/features/chat/voice/typed-send.ts";

const SRC = new URL("../src/features/chat/", import.meta.url);
const read = (relative: string) => readFileSync(new URL(relative, SRC), "utf8");

test("Enter mid-reveal is not a typed send of the cut transcript", () => {
  const revealing = "What is the weather like today?";
  assert.equal(isRevealPrefix("What is the weath", revealing), true);
  assert.equal(isRevealPrefix("What is ", revealing), true);
  assert.equal(
    isTypedVoiceSubmit({ box: "What is the weath", shown: "", paused: false, revealing }),
    false,
  );
});

test("typing something else during a reveal is still a typed send", () => {
  const revealing = "What is the weather like today?";
  assert.equal(isRevealPrefix("Actually never mind", revealing), false);
  assert.equal(
    isTypedVoiceSubmit({ box: "Actually never mind", shown: "", paused: false, revealing }),
    true,
  );
  // No reveal running: unchanged behaviour.
  assert.equal(isRevealPrefix("What is the weath", null), false);
  assert.equal(isTypedVoiceSubmit({ box: "What is the weath", shown: "", paused: false }), true);
});

test("the engine rushes the reveal instead of sending the box, and the loop honours it", () => {
  const engine = read("voice/voice-engine.tsx");
  const handler = engine.slice(engine.indexOf("registerVoiceTypedSubmit((box) =>"));
  const rush = handler.indexOf("rushVoiceReveal();");
  assert.ok(rush > 0, "Enter mid-reveal must rush the reveal");
  assert.ok(rush < handler.indexOf("submitTranscript(box, true)"), "rush comes before a typed send");
  const adapter = read("adapters/studio-voice-loop-adapter.ts");
  // The typing loop stops at the rush and falls through to the full-transcript send.
  assert.match(adapter, /for \(let i = stepChars; i < total && !mine\.rush; i \+= stepChars\)/);
  assert.match(adapter, /if \(reveal === mine\) reveal = null;/);
});
