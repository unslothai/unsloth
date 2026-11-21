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
  assert.match(engine, /const typed = withoutVoiceText\(fresh\.getState\(\)\.text\);/);
  assert.doesNotMatch(engine, /if \(fresh\.getState\(\)\.text\) \{\s*fresh\.setText\(""\);/);
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
  assert.match(commit, /if \(msgs\.length <= before && Date\.now\(\) - armedAt < VOICE_APPEND_WAIT_MAX_MS\) \{/);
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
  assert.match(clear.slice(0, 400), /if \(voiceWroteBox\(fresh\.getState\(\)\.text\)\) \{\s*fresh\.setText\(""\);/);
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
  const off = loop.indexOf('if (switchedOff()) {\n          dropSwitched();');
  assert.ok(off > 0 && off < loop.indexOf("heldTurn = {\n            text: transcript,"));
});

test("the coalesced run waits for a phrase still being heard", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const fire = engine.slice(engine.indexOf("const fireCoalescedRun = "));
  const wait = fire.indexOf("(rt.voiceHearing || rt.voiceTranscribing || voiceUtterancePending()) && Date.now() - armedAt < VOICE_HEARING_HOLD_MAX_MS");
  assert.ok(wait > 0 && wait < fire.indexOf("t.startRun("));
});

test("an empty continuation keeps the held phrase", () => {
  assert.match(loop, /if \(transcript\) heldTurn = null;\s*if \(transcript\) \{\s*const total = transcript\.length;/);
  assert.doesNotMatch(loop, /\n\s*heldTurn = null;\s*if \(transcript\) \{\s*const total/);
});

test("Escape on the orb hides it without reaching the composer's cancel", () => {
  const orb = readSrc("components/assistant-ui/voice-orb.tsx");
  assert.match(orb, /e\.stopPropagation\(\);\s*setVoiceOrbCollapsed\(true\);/);
  assert.match(orb, /window\.addEventListener\("keydown", onKeyDown, true\);/);
});

test("turning voice off mid-reveal stops the transcript reaching the composer", () => {
  const reveal = loop.slice(loop.indexOf("const stepChars = Math.max("));
  const emitFinal = reveal.indexOf("emitSpeech(transcript, true);");
  const offChecks = reveal.slice(0, emitFinal).match(/if \(switchedOff\(\)\) \{\s*dropSwitched\(\);/g) ?? [];
  assert.equal(offChecks.length, 2);
});

test("echo left in the preroll after the reply is not a turn on its own", () => {
  assert.match(loop, /if \(loud\) prerollVoicedMs \+= frameMs;/);
  assert.match(loop, /if \(speech\) \{\s*voicedMs \+= frameMs;\s*if \(!heardSpeech\) \{\s*voicedMs \+= prerollVoicedMs;/);
  assert.doesNotMatch(loop, /voicedMs > 0 && !heardSpeech/);
});

test("voice off waits for a detect in flight before unloading the detector", () => {
  const vad = readSrc("features/chat/voice/vad-client.ts");
  assert.match(vad, /return track\(runDetect\(wav, model, signal\)\);/);
  assert.match(vad, /return track\(startVadModel\(model, signal\)\);/);
  assert.match(vad, /await Promise\.all\(\[\.\.\.pendingLoads\]\);\s*await authFetch\("\/api\/inference\/audio\/vad\/unload"/);
});

test("a recording that crossed a voice off stays dropped after voice comes back", () => {
  const bridge = readSrc("features/chat/voice/voice-loop-bridge.ts");
  assert.match(bridge, /if \(next === "off" && voiceMode !== "off"\) offEpoch\+\+;/);
  assert.match(loop, /const switchedOff = \(\) =>\s*getVoiceMode\(\) === "off" \|\| voiceOffEpoch\(\) !== offAtOpen \|\| epoch !== heldEpoch;/);
});

test("speech heard after the chat model went away is never sent", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const submit = engine.slice(engine.indexOf("const submitTranscript = useCallback("));
  assert.match(submit.slice(0, 500), /if \(!hasChatModelRef\.current\) return;/);
  assert.match(engine, /if \(!hasChatModel\) \{\s*const composer = auiRef\.current\.composer\(\);\s*if \(composer\.getState\(\)\.dictation\) composer\.stopDictation\(\);/);
});

test("a re-arm strips only the voice fragment after a typed draft", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /const typed = withoutVoiceText\(fresh\.getState\(\)\.text\);\s*if \(typed !== null\) fresh\.setText\(typed\);/);
  assert.match(loop, /if \(!said \|\| !box\.endsWith\(said\)\) return null;[\s\S]{0,200}if \(before && !\/\\s\$\/\.test\(before\)\) return null;/);
});

test("Compare's read-only threads never mount the voice orb", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /\{voiceRequested && !hideComposer && \(\s*<Suspense fallback=\{null\}>\s*<VoiceOrb \/>/);
});

test("the mic re-arms for every reply stream, and a deleted saved voice falls back to the model's own", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /streamBegunRef\.current = true;\s*\/\/[^\n]*\n\s*setStreamSeq\(\(n\) => n \+ 1\);/);
  assert.match(engine, /\}, \[isSpeaking, streamSeq\]\);/);
  const player = readSrc("features/chat/hooks/use-tts-player.ts");
  assert.match(player, /if \(\/saved voice no longer exists\/i\.test\([^)]*\)\) \{[\s\S]{0,200}useChatRuntimeStore\.getState\(\)\.setSelectedVoiceId\(""\);\s*url = await request\(\);/);
});

test("a thread switch drops the old chat's pending transcript and its claim on the box", () => {
  assert.match(loop, /export function clearHeldTurn\(\): void \{\s*heldEpoch \+= 1;[\s\S]{0,250}voiceBoxText = "";/);
  assert.match(loop, /const epoch = heldEpoch;/);
});

test("a phrase still in VAD/STT holds the coalesced reply; an unmounted engine's mic retries and held phrase go stale", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(loop, /teardown\(\);\s*utterancesPending \+= 1;\s*try \{\s*await finalizeHeard\(\);\s*\} finally \{\s*utterancesPending -= 1;/);
  assert.match(engine, /if \(\(rt\.voiceHearing \|\| rt\.voiceTranscribing \|\| voiceUtterancePending\(\)\) && Date\.now\(\)/);
  assert.match(engine, /return \(\) => \{\s*rearmSeqRef\.current \+= 1;\s*(?:\/\/[^\n]*\n\s*)?clearHeldTurn\(\);\s*if \(voiceCoalesceTimerRef\.current\) clearTimeout/);
});

test("a deferred submit is dropped when the turn was reset first; a sent transcript stops claiming later drafts", () => {
  assert.doesNotMatch(loop, /setTimeout\(\(\) => requestVoiceSubmit\(\), 0\)/);
  assert.match(loop, /const submitLater = \(\) =>\s*setTimeout\(\(\) => \{\s*if \(!switchedOff\(\)\) requestVoiceSubmit\(\);\s*\}, 0\);/);
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const clear = engine.slice(engine.indexOf("const releaseVoiceText = voiceTextReleaser();"));
  assert.ok(clear.length > 0 && clear.indexOf("releaseVoiceText();") < clear.indexOf("// First message"));
});

test("the barge-in bar applies only while the reply is audible, not while it is being thought", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /registerVoicePlayback\(\s*\(\) => isPlayingRef\.current \|\| performance\.now\(\) - playedAtRef\.current < PLAYBACK_TAIL_MS,\s*\)/);
  assert.match(engine, /if \(isPlayingRef\.current && !isPlaying\) playedAtRef\.current = performance\.now\(\);/);
});

test("speech already underway when the reply starts playing keeps recording as a barge-in", () => {
  const frame = loop.slice(loop.indexOf("processor.onaudioprocess = (event) => {"));
  const latch = frame.indexOf("if (heardSpeech && !bargedIn && isVoicePlaybackActive()) {\n            bargedIn = true;\n            requestVoiceBargeIn();");
  assert.ok(latch > 0 && latch < frame.indexOf("if (!bargedIn && (isVoicePlaybackActive() || bargeConfirming)) {"));
});

test("a message sent from the composer drops a voice fragment still waiting for its continuation", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const submit = thread.slice(thread.indexOf("const handleSubmit = useCallback("));
  assert.ok(submit.indexOf("noteTypedSend();") > 0 && submit.indexOf("noteTypedSend();") < submit.indexOf("isRewindCommand(composerText)"));
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /registerVoiceTypedSend\(\(\) => \{\s*if \(voiceModeRef\.current === "active"\) clearHeldTurn\(\);\s*\}\);/);
});

test("a clip dropped by a reset re-arms the mic while voice stays on", () => {
  assert.match(loop, /const dropSwitched = \(\) => \{\s*finish\("cancelled"\);\s*if \(getVoiceMode\(\) === "active"\) setTimeout\(\(\) => requestVoiceResume\(\), 0\);\s*\};/);
  assert.equal((loop.match(/if \(switchedOff\(\)\) \{\s*dropSwitched\(\);/g) ?? []).length, 6);
});

test("a voice turn whose attachment send is slow still gets its reply", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /if \(msgs\.length <= before && Date\.now\(\) - armedAt < VOICE_APPEND_WAIT_MAX_MS\) \{/);
  assert.doesNotMatch(engine, /msgs\.length <= before && attempt < VOICE_COALESCE_MAX_RETRIES/);
});

test("turning voice on stops a read-aloud already playing before the mic opens", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const act = engine.slice(engine.indexOf("const activateLoop = useCallback("));
  const stopAt = act.indexOf("thread.stopSpeaking();");
  assert.ok(stopAt > 0 && stopAt < act.indexOf("startDictation();"));
  assert.match(act.slice(0, stopAt), /if \(thread\.getState\(\)\.speech\) \{/);
});

test("a clip heard across voice off runs no VAD or STT and raises no error or download prompt", () => {
  const fin = loop.slice(loop.indexOf("const finalizeHeard = async () => {"));
  const firstOff = fin.indexOf("if (switchedOff()) {\n        dropSwitched();");
  assert.ok(firstOff > 0 && firstOff < fin.indexOf("await clipHasSpeech("));
  assert.match(fin, /if \(error instanceof SttModelNotDownloadedError && !switchedOff\(\)\) \{/);
  assert.match(fin, /\} catch \(error\) \{\s*if \(ended\) return;\s*if \(switchedOff\(\)\) \{\s*dropSwitched\(\);\s*return;\s*\}\s*console\.error\("Voice transcription error:"/);
});

test("the orb's Escape stays out of other pages while Chat is hidden; speech owns the voice slot first", () => {
  const orb = readSrc("components/assistant-ui/voice-orb.tsx");
  const listener = orb.slice(orb.indexOf("const onKeyDown = (e: KeyboardEvent) => {"));
  assert.ok(listener.indexOf("root.closest('[aria-hidden=\"true\"], [inert]') || root.getClientRects().length === 0) return;") < listener.indexOf("e.preventDefault();"));
  assert.match(orb, /<div\s+ref=\{rootRef\}/);
  const player = readSrc("features/chat/hooks/use-tts-player.ts");
  const synth = player.slice(player.indexOf("const synthesize = useCallback("));
  const own = synth.indexOf("if (tries >= 3 || !(await ensure())) {");
  assert.ok(own > 0 && own < synth.indexOf("url = await request();"));
  assert.ok(synth.indexOf("if (slotFailedReqRef.current === reqId) return null;") < own);
  // Residency is rechecked for every sentence, before the cached flag is trusted.
  const check = synth.indexOf("if (check && (slotFailedReqRef.current === reqId || !(await check()))) {");
  assert.ok(check > 0 && check < synth.indexOf("!useChatRuntimeStore.getState().voiceSlotLoaded; tries++"));
});

test("voice off during mic startup shows no microphone error", () => {
  assert.match(loop, /teardown\(\);\s*\/\/[^\n]*\n\s*if \(ended\) return;\s*console\.error\("Voice mic error:"/);
});

test("the dictation mic stays a direct child of the composer controls", () => {
  // The narrow-composer CSS shrinks `.aui-composer-action-wrapper > button`; a wrapper div would skip it.
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /\{!voiceEngaged && \(\s*<ComposerPrimitive\.If dictation=\{false\}>/);
  assert.doesNotMatch(thread, /voiceEngaged \? "hidden" : "contents"/);
});

test("signing out ends voice; a held continuation keeps the coalesced reply waiting", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /voiceSlot\.close\(\);\s*if \(prevVoiceModeRef\.current !== "off"\) \{\s*releaseVoiceDetector\(\);\s*endVoiceOnSignOut\(\);\s*useChatRuntimeStore\.getState\(\)\.setVoiceMode\("off"\);/);
  assert.match(loop, /registerVoiceSessionEnd\(\(\) => openSession\?\.cancel\(\)\);/);
  assert.match(loop, /if \(openSession === session\) openSession = null;/);
  assert.match(loop, /return utterancesPending > 0 \|\| heldTurn !== null;/);
});
