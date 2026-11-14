// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("voice mode takes the mic before the model-bound clip recorder", () => {
  const adapter = readSrc("features/chat/adapters/studio-dictation-adapter.tsx");
  const body = adapter.slice(adapter.indexOf("private createSession()"));
  const voice = body.indexOf('this.chatId !== null && getVoiceMode() === "active"');
  const clip = body.indexOf("micRecordsForModel(this.chatId)");
  assert.ok(voice > 0 && clip > 0 && voice < clip);
});

const loop = readSrc("features/chat/adapters/studio-voice-loop-adapter.ts");

test("a fragment held from a barge-in keeps listening as that barge-in, so its flush timeout runs", () => {
  assert.match(loop, /let bargedIn = heldTurn !== null && heldFromBarge;/);
  assert.match(loop, /at: heldTurn\?\.at \?\? performance\.now\(\),\s*\};\s*heldFromBarge = bargedIn;/);
});

test("the barge-in preroll is not trimmed while the detector is asked", () => {
  assert.match(loop, /if \(!bargeConfirming\) while \(preroll\.length \* frameMs > BARGE_PREROLL_MS\) preroll\.shift\(\);/);
});

test("a barge-in pause latches even before the first clip is queued", () => {
  const player = readSrc("features/chat/hooks/use-tts-player.ts");
  const pause = player.slice(player.indexOf("const pausePlayback = useCallback("));
  assert.doesNotMatch(pause.slice(0, 400), /clipsRef\.current\.size === 0/);
});

test("the re-arm clears only the transcript the loop wrote, never a typed draft", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /if \(voiceWroteBox\(fresh\.getState\(\)\.text\)\) fresh\.setText\(""\);/);
  assert.doesNotMatch(engine, /if \(fresh\.getState\(\)\.text\) \{\s*fresh\.setText\(""\);/);
  // Every transcript reaches the composer through emitSpeech, which records it.
  assert.equal((loop.match(/for \(const cb of speechCallbacks\)/g) ?? []).length, 1);
  assert.match(loop, /const emitSpeech = \(transcript: string, isFinal: boolean\) => \{\s*voiceBoxText = transcript;/);
});
