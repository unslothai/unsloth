// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const loop = readSrc("features/chat/adapters/studio-voice-loop-adapter.ts");
const engine = readSrc("features/chat/voice/voice-engine.tsx");

test("voice mode takes the mic before the model-bound clip recorder", () => {
  const adapter = readSrc("features/chat/adapters/studio-dictation-adapter.tsx");
  const body = adapter.slice(adapter.indexOf("private createSession()"));
  const voice = body.indexOf('this.chatId !== null && getVoiceMode() === "active"');
  const clip = body.indexOf("micRecordsForModel(this.chatId)");
  assert.ok(voice > 0 && clip > 0 && voice < clip);
});

test("a session dropped by a clear re-arms the listener", () => {
  const drop = loop.slice(loop.indexOf('why: "cleared"'));
  assert.match(drop.slice(0, 300), /finish\("stopped"\);[\s\S]*?requestVoiceResume\(\)[\s\S]*?return;/);
});

test("a transcript revealed across a Stop is dropped, not sent", () => {
  const reveal = loop.slice(loop.indexOf("const mine = { shown: shown(transcript), rush: false };"));
  const submit = reveal.indexOf("requestVoiceSubmit(transcript)");
  const guard = reveal.indexOf("if (epoch !== heldEpoch) {");
  assert.ok(guard > 0 && guard < submit);
});

test("parked turns flush only once the voice actually loaded", () => {
  const effect = engine.slice(engine.indexOf("wasSlotLoadingRef.current = voiceSlotLoading;"));
  assert.match(
    effect.slice(0, 900),
    /if \(pendingForVoiceRef\.current\.length && useChatRuntimeStore\.getState\(\)\.voiceSlotLoaded\) \{[\s\S]*?submitRef\.current\(""\)/,
  );
});

test("Stop closes the open session before re-arming, as a thread reset does", () => {
  const handler = engine.slice(engine.indexOf("registerVoiceStopSpeaking(() => {"));
  const body = handler.slice(0, handler.indexOf("registerVoiceStopSpeaking(null)"));
  assert.match(body, /if \(composer\.getState\(\)\.dictation\) composer\.stopDictation\(\);\s*resumeListen\(\);/);
});

test("a voice turn with attachments goes through the composer, and the run waits for it to land", () => {
  const commit = engine.slice(engine.indexOf("const commitUtterance = useCallback("));
  assert.match(
    commit,
    /if \(composer\.getState\(\)\.attachments\.length > 0\) \{[\s\S]*?composer\.setText\(text\);\s*composer\.send\(\{ startRun: false \}\);/,
  );
  assert.match(commit, /if \(msgs\.length <= before && attempt < VOICE_COALESCE_MAX_RETRIES\) \{/);
});

test("only the same request again counts as impatience", () => {
  assert.match(engine, /const sameAsk = isSameAsk\(lastUserPrompt\(\), text\);/);
});

test("a fragment held from a barge-in keeps listening as that barge-in, so its flush timeout runs", () => {
  assert.match(loop, /let bargedIn = heldTurn !== null && heldFromBarge;/);
  assert.match(loop, /heldFromBarge = bargedIn;/);
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
  assert.match(engine, /if \(voiceWroteBox\(fresh\.getState\(\)\.text\) && !isComposerHeldForEdit\(\)\) fresh\.setText\(""\);/);
  assert.equal((loop.match(/for \(const cb of speechCallbacks\)/g) ?? []).length, 1);
  assert.match(loop, /voiceBoxText = shown\(transcript\);/);
});

test("a barge-in that ends before the detector answers keeps its voiced length", () => {
  assert.match(loop, /if \(bargeConfirming\) bargePeakMs = Math\.max\(bargePeakMs, bargeGate\.loudMs\);/);
  assert.match(loop, /voicedMs = Math\.max\(bargeGate\.loudMs, bargePeakMs\);/);
});
