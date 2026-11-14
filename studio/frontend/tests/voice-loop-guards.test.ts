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

test("a voice turn with attachments goes through the composer, and the run waits for it to land", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const commit = engine.slice(engine.indexOf("const submitTranscript = useCallback("));
  assert.match(
    commit,
    /if \(composer\.getState\(\)\.attachments\.length > 0\) \{[\s\S]*?composer\.setText\(text\);\s*composer\.send\(\{ startRun: false \}\);/,
  );
  assert.match(commit, /if \(msgs\.length <= before && attempt < VOICE_COALESCE_MAX_RETRIES\) \{/);
});

test("a barge-in that ends before the detector answers keeps its voiced length", () => {
  assert.match(loop, /if \(bargeConfirming\) bargePeakMs = Math\.max\(bargePeakMs, bargeGate\.loudMs\);/);
  assert.match(loop, /voicedMs = Math\.max\(bargeGate\.loudMs, bargePeakMs\);/);
});

test("the barge-in probe is judged on the shorter bar its clip can meet", () => {
  assert.match(loop, /const BARGE_VAD_MIN_SPEECH_MS = 200;/);
  assert.match(loop, /BARGE_VAD_MIN_SPEECH_MS,\s*\)\.then\(\(voiced\) => \{/);
});

test("the delayed composer cleanup clears only the refilled transcript", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const clear = engine.slice(engine.indexOf("const deferredClear = (n: number) => {"));
  assert.match(clear.slice(0, 400), /if \(!voiceWroteBox\(fresh\.getState\(\)\.text\)\) return;/);
});

test("frames stay in order when the reply ends while the barge-in probe is out", () => {
  assert.match(loop, /if \(!bargedIn && \(isVoicePlaybackActive\(\) \|\| bargeConfirming\)\) \{/);
});
