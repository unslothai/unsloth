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

test("a turn queued behind a reply that already ended is sent, not parked", () => {
  assert.match(
    engine,
    /const replyGone =\s*!auiRef\.current\.thread\(\)\.getState\(\)\.isRunning && !isSpeakingRef\.current && !isPlayingRef\.current;\s*if \(action === "ignore" \|\| \(action === "queue" && !replyGone\)\) \{/,
  );
});

test("the barge-in probe is judged on the shorter bar its clip can meet", () => {
  assert.match(loop, /const BARGE_VAD_MIN_SPEECH_MS = 200;/);
  assert.match(loop, /BARGE_VAD_MIN_SPEECH_MS,?\s*\)\.then\(\(voiced\) => \{/);
});

test("a draft typed before speaking leads the spoken turn instead of being cleared", () => {
  assert.match(engine, /const draft = spoken && !typed \? takeTypedDraft\(composer\.getState\(\)\.text\) : "";/);
  assert.match(engine, /if \(draft\) text = `\$\{draft\} \$\{text\}`;/);
  assert.match(loop, /export function takeTypedDraft\(box: string\): string \{/);
});

test("a mic lost mid-utterance sends what was heard instead of waiting for a frame that never comes", () => {
  const reopen = loop.slice(loop.indexOf("const reopen = (why: string"));
  assert.match(reopen.slice(0, 400), /if \(utteranceInFlight\) \{[\s\S]*?void finalize\(\);\s*return;/);
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
  assert.match(loop, /detectClip\(wav, bargedIn \? BARGE_VAD_MIN_SPEECH_MS : VAD_MIN_SPEECH_MS\)/);
});

test("a voice pause survives the engine remount: the new player starts held", () => {
  assert.match(
    engine,
    /useEffect\(\(\) => \{\s*if \(useChatRuntimeStore\.getState\(\)\.voicePaused\) holdPlayback\(true\);\s*\}, \[holdPlayback\]\);/,
  );
});


test("a voice that failed to load is not warmed (warm, fail, reload looped)", () => {
  const effect = engine.slice(engine.indexOf("wasSlotLoadingRef.current = voiceSlotLoading;"));
  assert.match(effect.slice(0, 500), /if \(useChatRuntimeStore\.getState\(\)\.voiceSlotLoaded\) void warmAsides(IfTts)?\(asidePhrases\(\)\);/);
});

test("an image-share Allow from the floating orb brings the chat card up", () => {
  const host = readSrc("features/chat/voice/orb-widget-host.ts");
  assert.match(host, /if \(outcome === "needs_card"\) reveal\(t\);/);
});

test("speech begun just before the reply ended keeps its opening frames", () => {
  assert.match(loop, /for \(const frame of preroll\.splice\(0\)\) \{[\s\S]{0,300}chunks\.push\(frame\);\s*chunkVoiced\.push\(loud\);\s*chunkNotSilent\.push\(/);
  // Every pushed frame gets a notSilent entry, or utteranceSpan reads a later frame's flag.
  assert.match(loop, /chunkVoiced\.push\(true\);\s*chunkNotSilent\.push\(true\);/);
});

test("a temporary chat never asks for its voice decisions to be logged", () => {
  const client = readSrc("features/chat/voice/voice-models-client.ts");
  assert.match(client, /const temporary = useChatRuntimeStore\.getState\(\)\.incognito \|\| \(threadId !== null && isThreadIncognito\(threadId\)\);/);
  assert.match(client, /threadId !== null &&\s*voice\.voiceLogEnabled &&\s*!temporary &&/);
});

test("turning voice off forgets the open reply stream", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const off = engine.slice(engine.indexOf("const turnOff = useCallback("));
  assert.match(off.slice(0, off.indexOf("}, [stop")), /streamBegunRef\.current = false;\s*stop\(\);/);
});

test("opening a saved chat from an unsent draft drops the draft's held fragment", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /if \(prev === null && current !== null\) \{\s*if \(getVoiceMode\(\) !== "off"\) clearHeldTurn\(\);/);
});

test("a clip still being transcribed when voice turns off is dropped, not held", () => {
  const off = loop.indexOf('if (getVoiceMode() === "off") {\n          finish("cancelled");');
  assert.ok(off > 0 && off < loop.indexOf("heldTurn = {\n            text: transcript,"));
});

test("the coalesced run waits for a phrase still being heard", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const fire = engine.slice(engine.indexOf("const fireCoalescedRun = "));
  const wait = fire.indexOf("(rt.voiceHearing || rt.voiceTranscribing) && Date.now() - armedAt < VOICE_HEARING_HOLD_MAX_MS");
  assert.ok(wait > 0 && wait < fire.indexOf("t.startRun("));
});

test("an empty continuation keeps the held phrase", () => {
  assert.match(loop, /if \(transcript\) heldTurn = null;\s*if \(transcript\) \{\s*const total = transcript\.length;/);
  assert.doesNotMatch(loop, /\n\s*heldTurn = null;\s*if \(transcript\) \{\s*const total/);
});

test("a refused mic retry does not spend the silence watchdog", () => {
  assert.match(loop, /if \(!reacquireBackoff\.allow\(now\) \|\| !silence\.due\(now\)\) return;/);
});

test("a coalesced start still settling after a newer utterance does not fire", () => {
  const start = engine.slice(engine.indexOf("const start = () => {"));
  assert.match(start.slice(0, 500), /coalesceSeqRef\.current !== seq/);
  assert.match(start.slice(0, 600), /now\.isRunning \|\| now\.messages\[now\.messages\.length - 1\]\?\.id !== last\.id/);
  assert.match(engine, /const seq = \+\+coalesceSeqRef\.current;/);
});

test("a filler decision that lands after the run started leaves the quiet fallback on", () => {
  assert.match(engine, /if \(!fromModel \|\| auiRef\.current\.thread\(\)\.getState\(\)\.isRunning\) return;\s*fillerAnsweredRef\.current = true;/);
});

test("Escape on the orb hides it without reaching the composer's cancel", () => {
  const orb = readSrc("components/assistant-ui/voice-orb.tsx");
  assert.match(orb, /e\.stopPropagation\(\);\s*setVoiceOrbCollapsed\(true\);/);
  assert.match(orb, /window\.addEventListener\("keydown", onKeyDown, true\);/);
});
