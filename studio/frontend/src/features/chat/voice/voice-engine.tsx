// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The voice conversation loop: continuous speech to speech.
 *
 * VoiceEngine is headless. It owns the turn-taking -- listen, send, generate,
 * speak, listen again -- plus barge-in, and publishes the orb's state to the chat
 * runtime store. VoiceControlButton is the small composer control that mirrors that
 * state. The full-screen orb itself is voice-orb.tsx, and the module-level registry
 * both of these talk through is voice-loop-bridge.ts.
 *
 * It sits beside the composer in thread.tsx rather than inside it because the loop
 * has to keep running while the composer remounts (see _prevRunning below and the
 * bridge's own header).
 */

import { resetPromptQueues } from "@/components/assistant-ui/thread";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { ORB_IDLE_GRADIENT, orbConfig } from "@/components/assistant-ui/voice-orb";
import { StudioWebSpeechDictationAdapter } from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
import {
  TTS_AUDIO_TYPES,
  useTtsPlayer,
} from "@/features/chat/hooks/use-tts-player";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { deriveOrbState } from "@/features/chat/voice/orb-state";
import {
  getVoiceMode,
  registerVoiceResume,
  registerVoiceToggle,
  setVoiceMode as setLoopVoiceMode,
} from "@/features/chat/voice/voice-loop-bridge";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { cn } from "@/lib/utils";
import { useAui, useAuiState } from "@assistant-ui/react";
import { MessageSquareIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type FC } from "react";

// Tracks the previous thread-running state across the first-send remount, so the
// run-lifecycle effect doesn't lose the first turn's true->false transition. The
// loop's mode itself lives in the bridge, which is what the rest of the app reads.
let _prevRunning = false;

// Silence debounce: how long the transcript must stop growing before we treat
// the user as done and send. Reset on every transcript update so a mid-sentence
// pause never clips speech; only a real pause this long auto-sends.
const VOICE_SILENCE_DEBOUNCE_MS = 1500;
// Barge-in debounce: when speech is heard while TTS is playing, wait this long
// and only interrupt if the transcript kept growing across the window. A cough
// or stray blip produces one fragment that doesn't sustain, so it's ignored;
// real "wait, stop" speech keeps adding words and cuts the model off.
const VOICE_BARGE_IN_DEBOUNCE_MS = 300;
// Voice coalescing window: after a finished utterance is appended as a user
// bubble, wait this long for another utterance before generating. Utterances that
// arrive within the window fold into the SAME prompt (several bubbles, one reply)
// instead of stacking as separate turns. Tunable: lower = snappier single turns,
// higher = groups slower/longer multi-part barge-ins.
const VOICE_COALESCE_MS = 650;
// How long playback may pause between sentences before the orb admits a real
// synth stall and turns lilac ("Generating voice"). The swap between two already
// synthesized clips is well under this, so ordinary sentence boundaries hold the
// blue "Speaking" state instead of flickering to lilac; only a gap that persists
// past this (synthesis genuinely behind playback) shows lilac.
const SYNTH_GAP_MS = 350;

/**
 * Whether the plus menu may offer Voice at all. It needs BOTH halves, and the
 * listening half is the narrower one. Speaking: the browser's speechSynthesis, or a
 * loaded TTS codec. Listening: the streaming Web Speech engine specifically,
 * because the loop's turn-taking watches the transcript grow. A batch engine
 * (local or custom transcription) holds its transcript until end-of-utterance, so
 * the orb would sit on "listening" and nothing would ever send.
 *
 * Gating on TTS alone offered Voice in browsers with speechSynthesis but no
 * SpeechRecognition, where it could be switched on and never produce a word.
 */
export function useVoiceAvailable(): boolean {
  const activeAudioType = useChatRuntimeStore((s) => {
    const m = s.models.find((mm) => mm.id === s.params.checkpoint);
    return m?.audioType ?? null;
  });
  const dictationEngine = useVoiceSettingsStore((s) => s.dictationEngine);
  return (
    ((typeof window !== "undefined" && "speechSynthesis" in window) ||
      TTS_AUDIO_TYPES.has(activeAudioType ?? "")) &&
    dictationEngine === "browser" &&
    StudioWebSpeechDictationAdapter.isSupported()
  );
}

export const VoiceEngine: FC = () => {
  const aui = useAui();
  const [voiceMode, setVoiceModeState] = useState(getVoiceMode());
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Barge-in debounce: timer that confirms sustained speech, and the latest
  // transcript so the timer can check whether it grew across the window.
  const bargeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Coalesce debounce: after appending a spoken utterance as a user bubble, this
  // fires the single run once the user pauses; reset each time another utterance
  // folds into the same prompt.
  const voiceCoalesceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestTranscriptRef = useRef("");
  // voiceModeRef is only written in toggle/activate — never in the render body.
  const voiceModeRef = useRef(getVoiceMode());
  const isSpeakingRef = useRef(false);
  // Mirrors isPlaying (a clip is audibly playing) separately from isSpeaking (the
  // TTS session is streaming). Barge-in must cut in BOTH cases -- including the
  // tail where the last clip is still playing after the session already ended.
  const isPlayingRef = useRef(false);
  // Debounce timer for the speaking -> synthesizing (lilac) transition, so a brief
  // inter-sentence playback gap doesn't flicker the orb to lilac.
  const synthGapTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const auiRef = useRef(aui);
  auiRef.current = aui;

  // Read from the store keyed by activeThreadId, not useAuiState(thread.isRunning):
  // the store value survives the composer remount on first send, so the first
  // turn's running transition isn't missed (same reason the prompt queue uses
  // runningByThreadId instead of aui.thread()).
  const isThreadRunning = useChatRuntimeStore((s) =>
    s.activeThreadId ? Boolean(s.runningByThreadId[s.activeThreadId]) : false,
  );
  const dictationStatusType = useAuiState(
    ({ composer }) => composer.dictation?.status.type,
  );
  const dictationTranscript = useAuiState(
    ({ composer }) => composer.dictation?.transcript ?? "",
  );
  const activeAudioType = useChatRuntimeStore((s) => {
    const m = s.models.find((m) => m.id === s.params.checkpoint);
    return m?.audioType ?? null;
  });
  // The store field, synced from /voice/status -- not "a voice is selected", which
  // is true from the moment the picker changes and stays true while the slot is
  // still loading. The player keys both isTtsModel and streamMode off this, so the
  // selection alone would POST the first synth into an empty slot.
  const voiceSlotLoaded = useChatRuntimeStore((s) => s.voiceSlotLoaded);

  // Called after speaking ends (or immediately if there's nothing to speak).
  const resumeListen = useCallback(() => {
    // After a session ends (no-speech finish, silence timer), the composer's
    // dictation field can lag a few frames before clearing. Clicking Dictate
    // while it is still set would toggle dictation OFF and kill the loop, so
    // poll briefly for it to clear before re-arming; as a last resort, re-arm
    // anyway rather than leave the loop dead.
    const MAX_ATTEMPTS = 5;
    const RETRY_MS = 50;

    const clickDictate = () => {
      const fresh = auiRef.current.composer();
      // Already listening (e.g. the mic was armed during TTS for barge-in): do
      // NOT restart it. Calling startDictation() on a live session tears the
      // current mic down -- that's what broke the continuous loop (turn 1 heard
      // you, turn 2 went deaf). The old button-click was a no-op here because the
      // Dictate button isn't rendered while dictating; mirror that no-op.
      if (fresh.getState().dictation) return;
      // Proactively clear any stale text on a FRESH handle before re-arming, in
      // case the deferred post-send clear hadn't landed before this re-arm got
      // here. Without it, turn 2's dictation appends onto turn 1's leftover text.
      if (fresh.getState().text) {
        fresh.setText("");
      }
      // Start dictation via the composer runtime, NOT by clicking the Dictate
      // button: in voice mode that button is hidden/replaced by the orb, so a
      // DOM query returns null and the mic never opens (silent green orb).
      fresh.startDictation();
    };

    const attempt = (n: number) => {
      // Voice turned off while we were waiting — abort the re-arm.
      if (voiceModeRef.current !== "active") return;
      const hasDictation = Boolean(
        auiRef.current.composer().getState().dictation,
      );
      if (!hasDictation) {
        clickDictate();
        return;
      }
      if (n < MAX_ATTEMPTS) {
        setTimeout(() => attempt(n + 1), RETRY_MS);
        return;
      }
      // Exhausted ~250ms of retries; assume the state is stale and re-arm.
      clickDictate();
    };

    attempt(1);
  }, []);

  const { isSpeaking, isPlaying, beginStream, feedText, endStream, stop, primeAudio } =
    useTtsPlayer(activeAudioType, resumeListen, voiceSlotLoaded);
  isSpeakingRef.current = isSpeaking;
  isPlayingRef.current = isPlaying;
  // Streaming TTS handles (refs so the run-lifecycle effect never goes stale).
  const beginStreamRef = useRef(beginStream);
  beginStreamRef.current = beginStream;
  const feedTextRef = useRef(feedText);
  feedTextRef.current = feedText;
  const endStreamRef = useRef(endStream);
  endStreamRef.current = endStream;
  // Interval that feeds the growing assistant reply into the TTS stream.
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Latest assistant reply text (concatenated text parts of the newest assistant
  // message). Used both to stream during generation and to flush on completion.
  const latestAssistantText = useCallback((): string => {
    const messages = auiRef.current.thread().getState().messages;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role !== "assistant") continue;
      let text = "";
      for (const part of msg.content as Array<{ type: string; text?: string }>) {
        if (part.type === "text" && part.text) text += part.text;
      }
      return text;
    }
    return "";
  }, []);

  // Submit a batch-STT (Whisper) transcript. The adapter has already committed
  // the final transcript into the composer via its onSpeech(isFinal) callback,
  // so here we just end dictation and send it — the run-lifecycle effect then
  // speaks the reply and re-arms the mic, same as the streaming path. Mirrors
  // the silence-timer send block (clear-on-turn-1 race guard included).
  const submitTranscript = useCallback(() => {
    if (voiceModeRef.current !== "active") return;
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
    const text = composer.getState().text.trim();
    // Barge-in: the instant the user speaks, cut the audio and the in-flight run.
    // This utterance supersedes whatever the model was saying. Cover the tail
    // clip (isPlaying) too, not just an active streaming session (isSpeaking).
    if (isSpeakingRef.current || isPlayingRef.current) stop();
    if (!text) {
      resumeListen();
      return;
    }
    const thread = auiRef.current.thread();
    if (thread.getState().isRunning) {
      try {
        thread.cancelRun();
      } catch {
        // Run may have ended between the check and here.
      }
    }
    // Voice coalescing: several barge-ins in a row should read as ONE prompt --
    // a user bubble each, then a single reply -- NOT a stack of separate turns
    // like the text-chat prompt queue. Append this utterance WITHOUT running,
    // clear that queue, keep the mic live, and (re)arm the coalesce debounce; the
    // single run fires only once the user has actually paused. Another utterance
    // arriving first appends another bubble and resets the debounce, folding into
    // the same prompt.
    resetPromptQueues();
    const DEFERRED_CLEAR_MS = 50;
    const deferredClear = (n: number) => {
      if (voiceModeRef.current !== "active") return;
      const fresh = auiRef.current.composer();
      if (!fresh.getState().text) return;
      fresh.setText("");
      if (n < 5) setTimeout(() => deferredClear(n + 1), DEFERRED_CLEAR_MS);
    };

    // First message of a thread: the append + delayed-startRun coalescing path
    // can't reliably drive a brand-new thread -- its runtime remounts on the first
    // send, so the captured handle goes stale and the run never fires (the bubble
    // shows but gets no reply). Use the normal composer.send() for the very first
    // utterance; it creates the thread and runs. Coalescing kicks in from turn 2 on.
    if (thread.getState().messages.length === 0) {
      composer.send();
      composer.setText("");
      setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
      return;
    }

    let appended = false;
    try {
      auiRef.current.thread().append({
        role: "user",
        content: [{ type: "text", text }],
        createdAt: new Date(),
        startRun: false,
      } as never);
      appended = true;
    } catch {
      // Runtime without append({startRun:false}) support -- fall back below.
    }

    if (!appended) {
      // Fallback: old immediate-send behavior (text is still in the composer).
      composer.send();
      composer.setText("");
      setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
      return;
    }

    composer.setText("");
    setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
    // Keep listening so the user can chain more utterances into this same prompt.
    resumeListen();
    if (voiceCoalesceTimerRef.current) clearTimeout(voiceCoalesceTimerRef.current);
    voiceCoalesceTimerRef.current = setTimeout(() => {
      voiceCoalesceTimerRef.current = null;
      if (voiceModeRef.current !== "active") return;
      const t = auiRef.current.thread();
      if (t.getState().isRunning) return;
      const msgs = t.getState().messages;
      const lastId = msgs.length > 0 ? msgs[msgs.length - 1].id : null;
      try {
        t.startRun({ parentId: lastId });
      } catch {
        // startRun unsupported -- the bubbles are already shown; nothing to do.
      }
    }, VOICE_COALESCE_MS);
  }, [resumeListen, stop]);

  // Sync derived orb state to the store so VoiceOrb can read it without prop-drilling.
  // The priority order itself lives in deriveOrbState, which is pure and tested.
  const setVoiceOrbState = useChatRuntimeStore((s) => s.setVoiceOrbState);
  const voiceSlotLoading = useChatRuntimeStore((s) => s.voiceSlotLoading);
  const voiceTranscribing = useChatRuntimeStore((s) => s.voiceTranscribing);
  const voiceHearing = useChatRuntimeStore((s) => s.voiceHearing);
  useEffect(() => {
    const clearSynthGap = () => {
      if (synthGapTimerRef.current) {
        clearTimeout(synthGapTimerRef.current);
        synthGapTimerRef.current = null;
      }
    };
    const decision = deriveOrbState({
      voiceMode,
      voiceSlotLoading,
      voiceHearing,
      isPlaying,
      voiceTranscribing,
      isThreadRunning,
      isSpeaking,
    });
    if (decision.kind === "set") {
      clearSynthGap();
      setVoiceOrbState(decision.state);
      return;
    }
    // A TTS session with nothing playing. Only show "Generating voice" if the gap
    // outlasts SYNTH_GAP_MS -- playback genuinely behind synthesis -- rather than
    // flickering at every sentence boundary. If a clip starts inside the window,
    // isPlaying flips true, this effect re-runs into the "set" arm above, and
    // clearSynthGap cancels the pending switch.
    if (synthGapTimerRef.current) return;
    synthGapTimerRef.current = setTimeout(() => {
      synthGapTimerRef.current = null;
      // Re-check live via refs at fire time: only go lilac if still stalled.
      if (
        voiceModeRef.current === "active" &&
        isSpeakingRef.current &&
        !isPlayingRef.current
      ) {
        setVoiceOrbState("synthesizing");
      }
    }, SYNTH_GAP_MS);
  }, [voiceMode, voiceSlotLoading, voiceHearing, voiceTranscribing, isThreadRunning, isSpeaking, isPlaying, setVoiceOrbState]);

  // Clear any pending synth-gap timer on unmount so it can't fire after teardown.
  useEffect(() => () => {
    if (synthGapTimerRef.current) clearTimeout(synthGapTimerRef.current);
  }, []);

  // Helper: transition to "active" and start the mic.
  const activateLoop = useCallback(() => {
    setLoopVoiceMode("active");
    voiceModeRef.current = "active";
    setVoiceModeState("active");
    if (!auiRef.current.thread().getState().isRunning && !isSpeakingRef.current) {
      auiRef.current.composer().startDictation();
    }
  }, []);

  // On remount: restore "active" only — "configuring" stays as-is (no mic).
  // Deferred one tick: on a New Chat the Thread remounts here with getVoiceMode() still
  // "active", but the parent's thread-switch reset (which flips voice OFF) runs in
  // the SAME commit, after this child effect. Starting the mic synchronously would
  // arm dictation on the fresh chat that voice is being turned off for (voice reads
  // off, yet the red "stop dictation" square is showing). By deferring and
  // re-checking getVoiceMode(), the reset settles first and we skip the restore. The
  // first-send remount keeps getVoiceMode() "active", so dictation still resumes there.
  useEffect(() => {
    if (getVoiceMode() !== "active" || isThreadRunning) return;
    const id = setTimeout(() => {
      if (getVoiceMode() !== "active") return;
      if (auiRef.current.thread().getState().isRunning || isSpeakingRef.current) return;
      auiRef.current.composer().startDictation();
    }, 0);
    return () => clearTimeout(id);
  }, []); // mount only

  // Watch the store: when it transitions from "configuring" → "active"
  // (triggered externally by the voice-model dropdown pick), activate the loop.
  const storeVoiceMode = useChatRuntimeStore((s) => s.voiceMode);
  useEffect(() => {
    if (
      storeVoiceMode === "active" &&
      voiceModeRef.current === "configuring"
    ) {
      activateLoop();
    }
  }, [storeVoiceMode, activateLoop]);

  // Run lifecycle: while the model streams, synthesize each finished sentence so
  // the first plays fast; on completion flush the rest and re-arm the mic.
  useEffect(() => {
    if (isThreadRunning) {
      _prevRunning = true;
      if (voiceModeRef.current === "active") {
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
        const composer = auiRef.current.composer();
        if (composer.getState().dictation) composer.stopDictation();
        // Begin streaming TTS and feed the reply as it generates so audio starts
        // on the first complete sentence, not after the whole reply.
        beginStreamRef.current();
        if (streamPollRef.current) clearInterval(streamPollRef.current);
        streamPollRef.current = setInterval(() => {
          feedTextRef.current(latestAssistantText());
        }, 150);
        // The mic is not re-armed during generation here: on the streaming engine
        // the silence timer would send a second, concurrent run mid-generation.
        // armDuringTts below arms it for the playback window instead. A batch STT
        // engine wants the opposite (arm here, let its own VAD supersede the run),
        // which is part of wiring that engine into the loop.
      }
      return;
    }
    if (!_prevRunning) return;
    _prevRunning = false;

    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    if (voiceModeRef.current !== "active") return;
    // Note: no isSpeaking guard here. If a stale utterance is still speaking when
    // a newer reply lands, the newest wins — beginStream above supersedes via
    // stop(). Skipping would silently drop the reply and dead-end the loop.

    const text = latestAssistantText();
    if (!text) {
      resumeListen();
      return;
    }
    // Flush the remaining sentences (incl. the trailing one) and finish; the
    // stream's consumer calls resumeListen when playback drains.
    endStreamRef.current(text);
    // Arm the mic DURING TTS so barge-in works: the streaming engine needs a live
    // session for its transcript to grow. The just-ended turn's dictation can read
    // as "set" for a few frames, and a single check would skip arming (leaving no
    // mic to barge with), so retry until it clears, then click Dictate.
    const armDuringTts = (n: number) => {
      if (voiceModeRef.current !== "active") return;
      if (auiRef.current.composer().getState().dictation) {
        if (n < 6) setTimeout(() => armDuringTts(n + 1), 60);
        return;
      }
      auiRef.current.composer().startDictation();
    };
    armDuringTts(0);
  }, [isThreadRunning, resumeListen, latestAssistantText]);

  // Silence timer: only fires in "active" state. This watches the transcript
  // grow, so it needs the streaming (Web Speech) engine; VoiceControlButton
  // refuses to enter voice mode on any other, because a batch engine holds its
  // transcript until end-of-utterance and this would stop the mic after 1.5s and
  // find an empty composer every time.
  useEffect(() => {
    if (dictationStatusType !== "running") return;
    if (voiceModeRef.current !== "active") return;

    // Barge-in (debounced): speech while TTS is playing only interrupts if it's
    // sustained. On the first fragment, open a window and snapshot the transcript
    // length; when it elapses, barge only if the transcript grew (the user kept
    // talking). A cough/blip doesn't grow, so it's ignored.
    latestTranscriptRef.current = dictationTranscript;
    if (isSpeakingRef.current && dictationTranscript.trim()) {
      if (bargeTimerRef.current === null) {
        const baseline = dictationTranscript.trim().length;
        bargeTimerRef.current = setTimeout(() => {
          bargeTimerRef.current = null;
          if (!isSpeakingRef.current) return;
          if (latestTranscriptRef.current.trim().length > baseline) stop();
        }, VOICE_BARGE_IN_DEBOUNCE_MS);
      }
    } else if (bargeTimerRef.current !== null) {
      clearTimeout(bargeTimerRef.current);
      bargeTimerRef.current = null;
    }

    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    silenceTimerRef.current = setTimeout(() => {
      silenceTimerRef.current = null;
      if (voiceModeRef.current !== "active") return;
      const composer = auiRef.current.composer();
      composer.stopDictation();
      // Still audible? Then whatever the mic picked up is the model's own voice
      // coming back through the speakers. Real speech over the top gets through a
      // different door: it grows the transcript, which cuts the TTS above, and by
      // the time this window expires nothing is playing. Sending here instead
      // would post a fragment of the model's own sentence as the user's next
      // message. Drop it and keep listening.
      if (isSpeakingRef.current || isPlayingRef.current) {
        composer.setText("");
        setTimeout(() => resumeListen(), 0);
        return;
      }
      if (composer.getState().isEditing && composer.getState().text.trim()) {
        composer.send();
        // On the first turn the send-reset can race with the new-thread bind /
        // composer remount, leaving the utterance as a prefix on the next
        // message. Clear explicitly to guard against that.
        composer.setText("");
        // The synchronous clear above lands on the pre-remount composer instance,
        // which the FIRST send unmounts a few ms later (ComposerActionWrapper
        // remount) — so on turn 1 it has no effect. Re-clear against a FRESH
        // composer handle (auiRef.current.composer(), never the captured `composer`)
        // across a few frames until it takes, or voice turns off.
        const DEFERRED_CLEAR_MAX = 5;
        const DEFERRED_CLEAR_MS = 50;
        const deferredClear = (n: number) => {
          if (voiceModeRef.current !== "active") return;
          const fresh = auiRef.current.composer();
          if (!fresh.getState().text) {
            return;
          }
          fresh.setText("");
          if (n < DEFERRED_CLEAR_MAX) setTimeout(() => deferredClear(n + 1), DEFERRED_CLEAR_MS);
        };
        setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
      } else {
        // No speech this window: stopDictation ends the session but nothing
        // re-arms it, so the loop would die while voiceMode stays "active"
        // (orb stuck, mic dead). Re-arm so we keep listening indefinitely.
        // Deferred so assistant-ui clears the ended session before resumeListen
        // re-checks it.
        setTimeout(() => resumeListen(), 0);
      }
    }, VOICE_SILENCE_DEBOUNCE_MS);

    return () => {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
    };
  }, [dictationTranscript, dictationStatusType, stop, resumeListen]);

  const toggle = useCallback(() => {
    // OFF → CONFIGURING (show dropdown, don't start mic)
    // CONFIGURING → OFF (user cancelled before picking)
    // ACTIVE → OFF (turn off the loop)
    const next: "off" | "configuring" =
      voiceModeRef.current === "off" ? "configuring" : "off";
    setLoopVoiceMode(next);
    voiceModeRef.current = next;
    setVoiceModeState(next);
    useChatRuntimeStore.getState().setVoiceMode(next);

    if (next === "configuring") {
      // Open each voice session with no TTS pre-selected (Browser voice) so a
      // previously-remembered pick doesn't silently auto-load; the user chooses
      // a voice explicitly from the "Speak with" dropdown.
      useChatRuntimeStore.getState().setSelectedVoiceModelId(null);
      primeAudio();
    }

    if (next === "off") {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      if (bargeTimerRef.current) {
        clearTimeout(bargeTimerRef.current);
        bargeTimerRef.current = null;
      }
      if (voiceCoalesceTimerRef.current) {
        clearTimeout(voiceCoalesceTimerRef.current);
        voiceCoalesceTimerRef.current = null;
      }
      stop();
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
    }
    // "configuring": dropdown appears via store; mic stays off.
  }, [stop, primeAudio]);

  // When voice mode turns off, make sure no loop-owned dictation session lingers.
  // Otherwise the manual Dictate button (un-hidden the instant voice is off) shows
  // its red pulsing "stop dictation" square because the composer still thinks it's
  // recording. stopDictation can lag a frame or a late re-arm can slip in, so retry
  // until the session actually clears.
  useEffect(() => {
    if (voiceMode !== "off") return;
    let n = 0;
    const clear = () => {
      if (voiceModeRef.current !== "off") return;
      const c = auiRef.current.composer();
      if (!c.getState().dictation) return;
      c.stopDictation();
      if (n++ < 8) setTimeout(clear, 60);
    };
    clear();
  }, [voiceMode]);

  useEffect(() => {
    return () => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      if (bargeTimerRef.current) clearTimeout(bargeTimerRef.current);
      if (voiceCoalesceTimerRef.current) clearTimeout(voiceCoalesceTimerRef.current);
      if (streamPollRef.current) clearInterval(streamPollRef.current);
    };
  }, []);

  // Expose the toggle so the plus-menu Voice item can drive it (and run
  // primeAudio inside that click gesture) from a different component.
  useEffect(() => {
    registerVoiceToggle(toggle);
    return () => {
      registerVoiceToggle(null);
    };
  }, [toggle]);

  // Expose resumeListen so the dictation adapter can re-arm the mic after a
  // recoverable no-speech timeout. It self-guards: no-op unless voice is active.
  useEffect(() => {
    registerVoiceResume(resumeListen);
    return () => {
      registerVoiceResume(null);
    };
  }, [resumeListen]);

  // Thread-switch voice reset moved OUT of VoiceEngine: this component remounts
  // across the ThreadWelcome → ThreadComposerDock (first-send) boundary and loses
  // its prev-thread subscription, so it never sees null → __LOCALID_xxx. The reset
  // now lives in SingleContent (chat-page.tsx), which is stable across that remount,
  // and drives requestVoiceToggle() via the module-level bridge.

  // Headless: the visible control now lives in the plus menu (ComposerToolsMenu).
  // This component stays mounted only to keep the voice loop's hooks/effects alive.
  return null;
};

// Composer voice control, shown once voice is opened from the + menu. It is a
// mini version of the orb whose color mirrors the full orb's state:
//   - configuring (loop not started): grey mini orb; click starts the loop.
//   - active + minimized (chat visible): colored mini orb (listening/thinking/
//     speaking); click re-opens the full orb.
//   - active + full orb showing: a "back to chat" icon; click minimizes the orb
//     so you can read/use the chat while speech-to-speech keeps running.
export const VoiceControlButton: FC = () => {
  const voiceMode = useChatRuntimeStore((s) => s.voiceMode);
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceMode = useChatRuntimeStore((s) => s.setVoiceMode);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);

  if (voiceMode === "off") return null;

  // Full orb is showing: offer to minimize it back to the chat view.
  if (voiceMode === "active" && !collapsed) {
    return (
      <TooltipIconButton
        tooltip="Back to chat"
        aria-label="Back to chat"
        variant="ghost"
        className="size-8 rounded-full text-foreground"
        onClick={() => setVoiceOrbCollapsed(true)}
      >
        <MessageSquareIcon className="size-5" />
      </TooltipIconButton>
    );
  }

  const active = voiceMode === "active";
  const gradient =
    active && orbState ? orbConfig[orbState].gradient : ORB_IDLE_GRADIENT;
  const tooltip = active ? "Open voice orb" : "Start voice mode";

  return (
    <TooltipIconButton
      tooltip={tooltip}
      aria-label={tooltip}
      variant="ghost"
      className="size-8 rounded-full"
      onClick={() => {
        setVoiceOrbCollapsed(false);
        if (!active) setVoiceMode("active");
      }}
    >
      <span
        aria-hidden
        className={cn(
          "size-4 rounded-full transition-colors",
          active && orbState === "speaking" && "animate-pulse",
        )}
        style={{ background: gradient }}
      />
    </TooltipIconButton>
  );
};
