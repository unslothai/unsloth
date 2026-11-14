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

test("a short barge-in (\"stop\") that passed its probe is transcribed, not dropped as too short", () => {
  assert.match(loop, /const minMs = bargedIn \? BARGE_CONFIRMED_MS : MIN_SPEECH_MS;/);
  assert.match(loop, /clipHasSpeech\(wav, bargedIn \? BARGE_VAD_MIN_SPEECH_MS : VAD_MIN_SPEECH_MS\)/);
});

test("speech begun just before the reply ended keeps its opening frames", () => {
  assert.match(loop, /for \(const frame of preroll\.splice\(0\)\) \{[\s\S]{0,300}chunks\.push\(frame\);\s*chunkVoiced\.push\(loud\);/);
});

test("turning voice off forgets the open reply stream", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const off = engine.slice(engine.indexOf('if (next === "off") {'));
  assert.match(off.slice(0, 600), /streamBegunRef\.current = false;[\s\S]*stop\(\);/);
});

test("a clip still being transcribed when voice turns off is dropped, not held", () => {
  const off = loop.indexOf('if (getVoiceMode() === "off") {\n          finish("cancelled");');
  assert.ok(off > 0 && off < loop.indexOf("heldTurn = {\n            text: transcript,"));
});

test("the coalesced run waits for a phrase still being heard", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const fire = engine.slice(engine.indexOf("const fireCoalescedRun = "));
  const wait = fire.indexOf("(rt.voiceHearing || rt.voiceTranscribing) && attempt < VOICE_COALESCE_MAX_RETRIES");
  assert.ok(wait > 0 && wait < fire.indexOf("t.startRun("));
});

test("an empty continuation keeps the held phrase", () => {
  assert.match(loop, /if \(transcript\) heldTurn = null;\s*if \(transcript\) \{\s*const total = transcript\.length;/);
  assert.doesNotMatch(loop, /\n\s*heldTurn = null;\s*if \(transcript\) \{\s*const total/);
});
