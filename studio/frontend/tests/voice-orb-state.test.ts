// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveOrbState,
  type VoiceOrbInputs,
} from "../src/features/chat/voice/orb-state.ts";

/** Every input false, voice running: the resting state of a live loop. */
function idle(overrides: Partial<VoiceOrbInputs> = {}): VoiceOrbInputs {
  return {
    voiceMode: "active",
    voiceSlotLoading: false,
    voiceHearing: false,
    isPlaying: false,
    voiceTranscribing: false,
    isThreadRunning: false,
    isSpeaking: false,
    ...overrides,
  };
}

test("the orb is blank unless the loop is actually running", () => {
  for (const voiceMode of ["off", "configuring"] as const) {
    // Even with everything else lit: configuring opens the picker, it does not
    // start the mic, so an orb state here would claim a loop that is not running.
    const decision = deriveOrbState(
      idle({ voiceMode, voiceSlotLoading: true, isPlaying: true }),
    );
    assert.deepEqual(decision, { kind: "set", state: null });
  }
});

test("a warming model reads as loading, not as a ready orb", () => {
  assert.deepEqual(deriveOrbState(idle({ voiceSlotLoading: true })), {
    kind: "set",
    state: "loading",
  });
});

test("hearing you outranks the model speaking, so barge-in is visible", () => {
  // This is the ordering that makes talking over the model legible: the mic is
  // live during playback on the streaming engine, and the moment it picks you up
  // the orb must stop claiming the model has the floor.
  assert.deepEqual(
    deriveOrbState(idle({ voiceHearing: true, isPlaying: true, isSpeaking: true })),
    { kind: "set", state: "hearing" },
  );
});

test("playback outranks the work still running behind it", () => {
  // The model keeps writing and the next sentence keeps synthesizing while the
  // first one plays. Playback is what the user perceives, so it wins.
  assert.deepEqual(
    deriveOrbState(
      idle({ isPlaying: true, isThreadRunning: true, isSpeaking: true }),
    ),
    { kind: "set", state: "speaking" },
  );
});

test("transcribing is reported ahead of generating, the order they happen in", () => {
  assert.deepEqual(
    deriveOrbState(idle({ voiceTranscribing: true, isThreadRunning: true })),
    { kind: "set", state: "transcribing" },
  );
  assert.deepEqual(deriveOrbState(idle({ isThreadRunning: true })), {
    kind: "set",
    state: "generating",
  });
});

test("a synthesis gap holds the current state instead of switching", () => {
  // A TTS session with nothing playing is usually two already-synthesized
  // sentences swapping the audio element. Deciding "synthesizing" here would
  // flicker the orb at every sentence boundary; the caller waits it out instead.
  assert.deepEqual(deriveOrbState(idle({ isSpeaking: true })), { kind: "hold" });
});

test("an armed mic with nothing else happening is listening", () => {
  assert.deepEqual(deriveOrbState(idle()), { kind: "set", state: "listening" });
});
