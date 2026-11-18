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
  const submit = reveal.indexOf("submitLater(transcript)");
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
  assert.match(commit, /if \(msgs\.length <= before && Date\.now\(\) - armedAt < VOICE_APPEND_WAIT_MAX_MS\) \{/);
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
  assert.match(engine, /const typed = isComposerHeldForEdit\(\) \? null : withoutVoiceText\(fresh\.getState\(\)\.text\);/);
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
  assert.match(clear.slice(0, 400), /if \(voiceWroteBox\(fresh\.getState\(\)\.text\)\) \{\s*fresh\.setText\(""\);/);
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
  assert.match(effect.slice(0, 500), /if \(useChatRuntimeStore\.getState\(\)\.voiceSlotLoaded\) void warmAsides\(asidePhrases\(\)\);/);
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

test("turning voice off mid-reveal stops the transcript reaching the composer", () => {
  // Part C: turnOff clears the held turn, which bumps heldEpoch; the reveal stops on that epoch.
  const reveal = loop.slice(loop.indexOf("const stepChars = Math.max("));
  assert.match(reveal.slice(0, 900), /if \(ended \|\| epoch !== heldEpoch\) break;[\s\S]*if \(epoch !== heldEpoch\) \{\s*voiceTrace\("stt\.drop"/);
  const off = engine.slice(engine.indexOf("const turnOff = useCallback("));
  assert.match(off.slice(0, off.indexOf("}, [stop")), /clearHeldTurn\(\);/);
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

test("the delayed stream close yields to a run that started inside the beat", () => {
  assert.match(engine, /streamBegunRef\.current = true;\s*streamGenRef\.current\+\+;/);
  assert.match(engine, /const gen = streamGenRef\.current;\s*setTimeout\(\(\) => \{[\s\S]{0,200}if \(gen !== streamGenRef\.current \|\| voiceModeRef\.current !== "active"\) return;/);
});

test("a recording that crossed a voice off stays dropped after voice comes back", () => {
  const bridge = readSrc("features/chat/voice/voice-loop-bridge.ts");
  assert.match(bridge, /if \(next === "off" && voiceMode !== "off"\) offEpoch\+\+;/);
  assert.match(loop, /const switchedOff = \(\) => getVoiceMode\(\) === "off" \|\| voiceOffEpoch\(\) !== offAtOpen;/);
});

test("speech heard after the chat model went away is never sent", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const submit = engine.slice(engine.indexOf("const submitTranscript = useCallback("));
  assert.match(submit.slice(0, 900), /if \(!hasChatModelRef\.current\) \{\s*if \(text && !typed\) composer\.setText\(text\);\s*return;/);
  assert.match(engine, /if \(!hasChatModel\) \{\s*const composer = auiRef\.current\.composer\(\);\s*if \(composer\.getState\(\)\.dictation\) composer\.stopDictation\(\);/);
});

test("a re-arm strips only the voice fragment after a typed draft", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /withoutVoiceText\(fresh\.getState\(\)\.text\);\s*if \(typed !== null\) fresh\.setText\(typed\);/);
  assert.match(loop, /if \(!said \|\| !box\.endsWith\(said\)\) return null;[\s\S]{0,200}if \(before && !\/\\s\$\/\.test\(before\)\) return null;/);
});

test("a long utterance the interject model would ignore is queued instead", () => {
  const override = engine.slice(engine.indexOf("function overrideInterject("));
  assert.match(override.slice(0, 1400), /const tooLongToIgnore = text\.trim\(\)\.split\(\/\\s\+\/\)\.length > IGNORE_MAX_WORDS;/);
  assert.match(override.slice(0, 1400), /\(continuing \|\| tooLongToIgnore\) && verdict === "ignore"\s*\? "queue"/);
});

test("a research run keeps its own Stop while voice mode is on", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /\{voiceActive && !isResearchActive \? \(\s*<VoicePauseStopButton \/>/);
  assert.match(thread, /\{isResearchActive \? \(\s*<Button[\s\S]{0,200}aria-label=\{researchStopping \? "Stopping research" : "Stop research"\}/);
  assert.match(thread, /\) : voiceActive \? null : \(\s*<AuiIf condition=\{\(\{ thread \}\) => thread\.isRunning\}>/);
});

test("a noise word continuing a held turn is kept, and an edit claimed mid-judgment is never auto-sent", () => {
  assert.match(loop, /const noise =\s*isSttNoise\(transcript\) && !heldTurn && !answersClosingQuestion\(transcript, getVoiceReplyText\(\)\);/);
  assert.match(loop, /\(await turnUnfinished\(transcript\)\);[\s\S]{0,300}if \(!editing && !heldEdit\.autoSend\(\)\) \{\s*editing = true;\s*transcript = spokenNow;/);
  assert.match(loop, /let editing = !heldEdit\.autoSend\(\);\s*const spokenNow = transcript;/);
  assert.match(loop, /unfinished &&\s*!editing &&\s*performance\.now\(\)/);
  assert.match(loop, /if \(editing \|\| !heldEdit\.autoSend\(\)\) \{\s*voiceTrace\("hold\.userEditAppend"/);
});

test("speech before the reply's first word is judged, not taken as a barge-in", () => {
  assert.match(engine, /thread\.getState\(\)\.isRunning && \(latestAssistantIsReasoning\(\) \|\| replyNotStarted\(\)\)/);
  assert.match(engine, /if \(!latestAssistantIsReasoning\(\) && !replyNotStarted\(\)\) return \{ verdict: "restart", kind: null \};/);
});

test("Compare's read-only threads never mount the voice orb", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /\{voiceRequested && !hideComposer && \(\s*<Suspense fallback=\{null\}>\s*<VoiceOrb \/>/);
});

test("the judge falls back to the local unfinished check, and parked speech waits for a loaded voice", () => {
  assert.match(loop, /const unfinished = r === null \? looksUnfinished\(text\) : r\.p < JUDGE_SEND_THRESHOLD;/);
  assert.match(engine, /if \(pendingForVoiceRef\.current\.length && !rt\.voiceSlotLoading && rt\.voiceSlotLoaded\) \{/);
});

test("the mic re-arms for every reply stream, and a deleted saved voice falls back to the model's own", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /streamBegunRef\.current = true;[\s\S]{0,1200}setStreamSeq\(\(n\) => n \+ 1\);/);
  assert.match(engine, /\}, \[isSpeaking, streamSeq\]\);/);
  const player = readSrc("features/chat/hooks/use-tts-player.ts");
  assert.match(player, /if \(\/saved voice no longer exists\/i\.test\([^)]*\)\) \{[\s\S]{0,200}useChatRuntimeStore\.getState\(\)\.setSelectedVoiceId\(""\);\s*url = await request\(\);/);
});

test("a thread switch drops the old chat's pending transcript and its claim on the box", () => {
  // Part C: clearHeldTurn bumps heldEpoch (the reveal and judge check it); the box claim goes on switch/off only.
  assert.match(loop, /export function clearHeldTurn\(\): void \{\s*heldEpoch \+= 1;/);
  assert.match(loop, /export function forgetVoiceBox\(\): void \{\s*voiceBoxText = "";/);
  for (const name of ["const turnOff = useCallback(", "const threadReset = useCallback("]) {
    const body = engine.slice(engine.indexOf(name));
    assert.match(body.slice(0, 2500), /clearHeldTurn\(\);\s*forgetVoiceBox\(\);/, name);
  }
});

test("a phrase still in VAD/STT holds the coalesced reply; an unmounted engine's mic retries and held phrase go stale", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(loop, /teardown\(\);\s*utterancesPending \+= 1;\s*try \{\s*await finalizeHeard\(\);\s*\} finally \{\s*utterancesPending -= 1;/);
  assert.match(engine, /if \(\(rt\.voiceHearing \|\| rt\.voiceTranscribing \|\| voiceUtterancePending\(\)\) && Date\.now\(\)/);
  assert.match(engine, /return \(\) => \{\s*rearmSeqRef\.current \+= 1;\s*(?:\/\/[^\n]*\n\s*)?clearHeldTurn\(\);\s*forgetVoiceBox\(\);\s*if \(voiceCoalesceTimerRef\.current\) clearTimeout/);
});

test("opening and dismissing voice setup keeps typed Send/Stop and the running reply", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /const voiceActive = useChatRuntimeStore\(\(s\) => s\.voiceMode === "active"\);/);
  assert.match(thread, /\{voiceActive && !isResearchActive \? \(\s*<VoicePauseStopButton \/>/);
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  const off = engine.slice(engine.indexOf("const turnOff = useCallback("));
  assert.match(off.slice(0, 300), /const wasActive = voiceModeRef\.current === "active";/);
  assert.match(off.slice(0, 2500), /if \(wasActive && thread\.getState\(\)\.isRunning\) \{\s*try \{\s*thread\.cancelRun\(\);/);
});

test("a filler that finishes synthesizing after the reply ended is dropped", () => {
  const player = readSrc("features/chat/hooks/use-tts-player.ts");
  const aside = player.slice(player.indexOf("const sayAside = useCallback("));
  assert.match(aside.slice(0, 1500), /const play = \([^)]*\) => \{\s*(?:\/\/[^\n]*\n\s*)?if \(reqRef\.current !== reqId \|\| streamRef\.current !== st \|\| st\.final\) return;/);
});

test("the since-last-export cutoff is taken before the request, not after the save dialog", () => {
  const section = readSrc("features/settings/components/voice-decision-log-section.tsx");
  const run = section.slice(section.indexOf("const doExport = async () => {"));
  assert.ok(run.indexOf("const cutoff = Date.now();") < run.indexOf("await exportLayaLog("));
  assert.match(run.slice(0, 900), /setExportedAt\(cutoff\);/);
});

test("a deferred submit is dropped when the turn was reset first; a sent transcript stops claiming later drafts", () => {
  assert.doesNotMatch(loop, /setTimeout\(\(\) => requestVoiceSubmit\(\), 0\)/);
  assert.match(loop, /const submitLater = \(text: string\) =>\s*setTimeout\(\(\) => \{\s*if \(!switchedOff\(\) && epoch === heldEpoch\) requestVoiceSubmit\(text\);\s*\}, 0\);/);
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
  assert.match(engine, /registerVoiceTypedSend\(\(\) => \{\s*if \(voiceModeRef\.current !== "active"\) return;\s*clearHeldTurn\(\);\s*(?:\/\/[^\n]*\n\s*)?clearPendingVoiceNotes\(\);\s*\}\);/);
});

test("a clip dropped by a reset re-arms the mic while voice stays on", () => {
  assert.match(loop, /const dropSwitched = \(\) => \{\s*finish\("cancelled"\);\s*if \(getVoiceMode\(\) === "active"\) setTimeout\(\(\) => requestVoiceResume\(\), 0\);\s*\};/);
  // Part C drops a cleared reveal through stt.drop, which resumes on its own; the pre-VAD drops also cancel their work.
  assert.equal((loop.match(/if \(switchedOff\(\)\) \{\s*dropSwitched\(\);/g) ?? []).length, 2);
});

test("a voice turn whose attachment send is slow still gets its reply", () => {
  const engine = readSrc("features/chat/voice/voice-engine.tsx");
  assert.match(engine, /if \(msgs\.length <= before && Date\.now\(\) - armedAt < VOICE_APPEND_WAIT_MAX_MS\) \{/);
  assert.doesNotMatch(engine, /msgs\.length <= before && attempt < VOICE_COALESCE_MAX_RETRIES/);
});

test("assistive tech's bare click pauses voice; a handled press is not tapped twice", () => {
  const button = readSrc("components/assistant-ui/voice-pause-stop-button.tsx");
  assert.match(button, /const release = \(\) => \{\s*releasedAtRef\.current = performance\.now\(\);/);
  assert.match(button, /if \(e\.detail === 0 && performance\.now\(\) - releasedAtRef\.current > 500\) requestVoicePauseToggle\(\);/);
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
  // Part C analyses the clip (VAD + STT) in one step, possibly started early: the drop also cancels it.
  const firstOff = fin.indexOf("if (switchedOff()) {\n        early.clear();\n        dropSwitched();");
  assert.ok(firstOff > 0 && firstOff < fin.indexOf("const analysis = ready ?? analyseClip(span);"));
  assert.match(fin, /if \(error instanceof SttModelNotDownloadedError && !switchedOff\(\)\) \{/);
  assert.match(fin, /\} catch \(error\) \{\s*if \(ended\) return;\s*if \(switchedOff\(\)\) \{\s*dropSwitched\(\);\s*return;\s*\}\s*console\.error\("Voice transcription error:"/);
});

test("the orb's Escape stays out of other pages while Chat is hidden; speech owns the voice slot first", () => {
  const orb = readSrc("components/assistant-ui/voice-orb.tsx");
  const listener = orb.slice(orb.indexOf("const onKeyDown = (e: KeyboardEvent) => {"));
  assert.ok(listener.indexOf("root.closest('[aria-hidden=\"true\"], [inert]') || root.getClientRects().length === 0) return;") < listener.indexOf("e.preventDefault();"));
  assert.match(orb, /\sref=\{rootRef\}/);
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
  assert.match(page, /useChatRuntimeStore\.getState\(\)\.setVadSlotLoaded\(false\);\s*(?:\/\/[^\n]*\n\s*)?endVoiceOnSignOut\(\);\s*useChatRuntimeStore\.getState\(\)\.setVoiceMode\("off"\);/);
  assert.match(loop, /registerVoiceSessionEnd\(\(\) => liveSession\?\.cancel\(\)\);/);
  assert.match(loop, /return utterancesPending > 0 \|\| heldTurn !== null;/);
});
