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

import { resetPromptQueuesForThread } from "@/components/assistant-ui/thread";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  ORB_IDLE_GRADIENT,
  orbConfig,
  VoiceOrb,
} from "@/components/assistant-ui/voice-orb";
import {
  StudioVoiceLoopAdapter,
  clearHeldTurn,
  getHeldTurnText,
  setMicMuted,
} from "@/features/chat/adapters/studio-voice-loop-adapter";
import { MicIcon as DictateMicIcon } from "@/lib/mic-icon";
import { useTtsPlayer } from "@/features/chat/hooks/use-tts-player";
import {
  fillerDelayMs,
  pickStage2,
  STAGE2_AFTER_MS,
  setPendingFiller,
  toolKindOf,
  type FillerDecision,
  type ToolKind,
} from "@/features/chat/voice/thinking-filler";
import {
  fillerFromModel,
  interjectAction,
} from "@/features/chat/voice/voice-models-client";
import { clearVoiceDecisions } from "@/features/chat/voice/voice-decisions";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { deriveOrbState } from "@/features/chat/voice/orb-state";
import {
  getVoiceMode,
  registerVoiceBargeAbort,
  registerVoiceBargeIn,
  registerVoiceListen,
  registerVoicePlayback,
  registerVoiceReplyText,
  registerVoiceResume,
  registerVoiceSubmit,
  registerVoiceThreadReset,
  registerVoiceToggle,
  setVoiceMode as setLoopVoiceMode,
} from "@/features/chat/voice/voice-loop-bridge";
import { cn } from "@/lib/utils";
import { useAui } from "@assistant-ui/react";
import { MessageSquareIcon, MicOffIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type FC } from "react";

// Tracks the previous thread-running state across the first-send remount, so the
// run-lifecycle effect doesn't lose the first turn's true->false transition. The
// loop's mode itself lives in the bridge, which is what the rest of the app reads.
let _prevRunning = false;

// Voice coalescing window: after a finished utterance is appended as a user
// bubble, wait this long for another utterance before generating. Utterances that
// arrive within the window fold into the SAME prompt (several bubbles, one reply)
// instead of stacking as separate turns. Tunable: lower = snappier single turns,
// higher = groups slower/longer multi-part barge-ins.
const VOICE_COALESCE_MS = 650;
// thread.cancelRun() is asynchronous, so the old run can still read as running
// when the coalesce window closes. Rather than give up and leave the appended
// bubbles unanswered, re-check every VOICE_COALESCE_MS up to this many times
// (~10s) before concluding the run will never settle.
const VOICE_COALESCE_MAX_RETRIES = 15;
// How long playback may pause between sentences before the orb admits a real
// synth stall and turns lilac ("Generating voice"). The swap between two already
// synthesized clips is well under this, so ordinary sentence boundaries hold the
// blue "Speaking" state instead of flickering to lilac; only a gap that persists
// past this (synthesis genuinely behind playback) shows lilac.
const SYNTH_GAP_MS = 350;

// Re-exported so voice mode is one lazy entry (loadVoiceMode).
export { VoiceOrb };
export { SttModelSelector } from "@/components/assistant-ui/stt-model-selector";
export { VoiceModelSelector } from "@/components/assistant-ui/voice-model-selector";
export { VadModelSelector } from "@/components/assistant-ui/vad-model-selector";
export { loadVoice, unloadVoice, voiceServing } from "@/features/chat/voice/voice-slot";

// The dictation dispatcher reaches the loop's adapter through the bridge, so the
// adapter loads with voice mode instead of at startup.
registerVoiceListen(() =>
  StudioVoiceLoopAdapter.isSupported() ? new StudioVoiceLoopAdapter().listen() : null,
);

export const VoiceEngine: FC = () => {
  const aui = useAui();
  const [voiceMode, setVoiceModeState] = useState(getVoiceMode());
  // Coalesce debounce: after appending a spoken utterance as a user bubble, this
  // fires the single run once the user pauses; reset each time another utterance
  // folds into the same prompt.
  const voiceCoalesceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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
  // Voice can be opened before a chat model is picked -- the orb is reachable from
  // the + menu at any time. Without one there is nothing to answer a spoken turn,
  // so the mic stays shut and the orb says so, rather than capturing into a dead
  // end. The ref is what the async re-arm retries read.
  const hasChatModel = useChatRuntimeStore((s) => Boolean(s.params.checkpoint));
  const hasChatModelRef = useRef(hasChatModel);
  hasChatModelRef.current = hasChatModel;
  // Supersedes an in-flight re-arm poll, so a later resumeListen can't leave two
  // of them racing to open the mic.
  const rearmSeqRef = useRef(0);
  // Called after speaking ends (or immediately if there's nothing to speak).
  const resumeListen = useCallback(() => {
    // every re-arm (a turn ending, a chat model arriving) waits for a voice still loading;
    // the load finishing calls back in here
    if (useChatRuntimeStore.getState().voiceSlotLoading) return;
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
      // A fragment the judge is holding stays on screen while the mic re-arms;
      // anything else in the box is turn 1's leftover and goes.
      const held = getHeldTurnText();
      if (held) {
        if (fresh.getState().text !== held) fresh.setText(held);
      } else if (fresh.getState().text) {
        fresh.setText("");
      }
      // Start dictation via the composer runtime, NOT by clicking the Dictate
      // button: in voice mode that button is hidden/replaced by the orb, so a
      // DOM query returns null and the mic never opens (silent green orb).
      fresh.startDictation();
    };

    const seq = ++rearmSeqRef.current;

    const attempt = (n: number) => {
      // Voice turned off while we were waiting — abort the re-arm.
      if (voiceModeRef.current !== "active") return;
      // No chat model: don't open the mic. Not a retry — waiting here would spin
      // for as long as the orb is up. Picking a model re-arms through its own
      // effect below.
      if (!hasChatModelRef.current) return;
      if (rearmSeqRef.current !== seq) return; // superseded by a newer re-arm
      // A run with no voice session (voice switched on mid-run) has nothing to
      // barge in on, and a mic open now would fold the next turn into this one.
      // Wait it out rather than dropping the re-arm, so the loop cannot dead-end.
      // With a session up, the mic opens in the adapter's barge-in mode.
      if (auiRef.current.thread().getState().isRunning && !isSpeakingRef.current) {
        setTimeout(() => attempt(n), RETRY_MS);
        return;
      }
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

  // Speech that landed while the model was still thinking and the interject model
  // called a separate question: held here and sent once the reply has been heard.
  const queuedUtteranceRef = useRef<string | null>(null);
  const submitRef = useRef<() => void>(() => {});
  const runStartedAtRef = useRef(0);
  const onPlaybackEnd = useCallback(() => {
    const queued = queuedUtteranceRef.current;
    if (queued && voiceModeRef.current === "active") {
      queuedUtteranceRef.current = null;
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
      composer.setText(queued);
      submitRef.current();
      return;
    }
    resumeListen();
  }, [resumeListen]);
  const {
    isSpeaking,
    isPlaying,
    speak,
    beginStream,
    feedText,
    endStream,
    sayAside,
    stop,
    primeAudio,
    pausePlayback,
    resumePlayback,
  } = useTtsPlayer(onPlaybackEnd);
  isSpeakingRef.current = isSpeaking;
  isPlayingRef.current = isPlaying;
  useEffect(() => {
    registerVoicePlayback(() => isSpeakingRef.current || isPlayingRef.current);
    return () => registerVoicePlayback(null);
  }, []);
  // Streaming TTS handles (refs so the run-lifecycle effect never goes stale).
  const beginStreamRef = useRef(beginStream);
  beginStreamRef.current = beginStream;
  const feedTextRef = useRef(feedText);
  feedTextRef.current = feedText;
  const endStreamRef = useRef(endStream);
  endStreamRef.current = endStream;
  const speakRef = useRef(speak);
  speakRef.current = speak;
  // Whether the run-start branch opened a TTS session for the run in flight. It
  // only does so while voice mode is already active, so a run that started before
  // the user switched voice mode on has no session for endStream to flush.
  const streamBegunRef = useRef(false);
  // Interval that feeds the growing assistant reply into the TTS stream.
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sayAsideRef = useRef(sayAside);
  sayAsideRef.current = sayAside;
  // Thinking filler for the run in flight: decided at submit (the model is told
  // about it), spoken after a short delay once the run starts, then at most one
  // more line if the model is still thinking. Text is held back until the filler
  // has been queued so it is always the first thing heard.
  const fillerRef = useRef<{
    phrase: string;
    timer: ReturnType<typeof setTimeout> | null;
    queued: boolean;
    spokenAt: number | null;
    stage2Done: boolean;
  } | null>(null);
  const clearFiller = useCallback(() => {
    const f = fillerRef.current;
    if (f?.timer) clearTimeout(f.timer);
    fillerRef.current = null;
  }, []);
  const latestAssistantToolKinds = useCallback((): ToolKind[] => {
    const messages = auiRef.current.thread().getState().messages;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role !== "assistant") continue;
      const kinds: ToolKind[] = [];
      for (const part of msg.content as ReadonlyArray<{ type: string; toolName?: string }>) {
        if (part.type === "tool-call") kinds.push(toolKindOf(part.toolName ?? ""));
      }
      return kinds;
    }
    return [];
  }, []);

  // Latest assistant reply text (concatenated text parts of the newest assistant
  // message). Used both to stream during generation and to flush on completion.
  // Whether the newest assistant message is still only reasoning: a reasoning
  // part exists and no text part has content yet. Drives the orb's "thinking".
  const latestAssistantIsReasoning = useCallback((): boolean => {
    const messages = auiRef.current.thread().getState().messages;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role !== "assistant") continue;
      let reasoning = false;
      for (const part of msg.content as Array<{ type: string; text?: string }>) {
        if (part.type === "text" && part.text) return false;
        if (part.type === "reasoning") reasoning = true;
      }
      return reasoning;
    }
    return false;
  }, []);
  const [isReasoning, setIsReasoning] = useState(false);

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

  // Send an utterance: cut whatever is playing, cancel the run it supersedes, and
  // fold it into one coalesced prompt with any other bubbles still unanswered.
  const commitUtterance = useCallback((text: string) => {
    const composer = auiRef.current.composer();
    // Barge-in: the instant the user speaks, cut the audio and the in-flight run.
    // This utterance supersedes whatever the model was saying. Cover the tail
    // clip (isPlaying) too, not just an active streaming session (isSpeaking).
    if (isSpeakingRef.current || isPlayingRef.current) stop();
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
    // clear THIS thread's queue (prompts queued in other chats are not ours to
    // drop), keep the mic live, and (re)arm the coalesce debounce; the single run
    // fires only once the user has actually paused. Another utterance arriving
    // first appends another bubble and resets the debounce, folding into the same
    // prompt.
    const activeThreadId = useChatRuntimeStore.getState().activeThreadId;
    if (activeThreadId) resetPromptQueuesForThread(activeThreadId);
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
    const fireCoalescedRun = (attempt: number) => {
      voiceCoalesceTimerRef.current = null;
      if (voiceModeRef.current !== "active") return;
      const t = auiRef.current.thread();
      if (t.getState().isRunning) {
        // The barge-in's cancelRun() has not settled yet. Wait for the running
        // transition instead of returning for good; the timer ref stays pointed at
        // the retry so a fresh utterance still clears and restarts the window.
        if (attempt < VOICE_COALESCE_MAX_RETRIES) {
          voiceCoalesceTimerRef.current = setTimeout(
            () => fireCoalescedRun(attempt + 1),
            VOICE_COALESCE_MS,
          );
        }
        return;
      }
      const msgs = t.getState().messages;
      const last = msgs.length > 0 ? msgs[msgs.length - 1] : null;
      // Only a trailing user bubble is unanswered. If a reply has landed since
      // (something else answered while we waited), starting another run here
      // would regenerate on top of it.
      if (!last || last.role !== "user") return;
      try {
        t.startRun({ parentId: last.id });
      } catch {
        // startRun unsupported -- the bubbles are already shown; nothing to do.
      }
    };
    if (voiceCoalesceTimerRef.current) clearTimeout(voiceCoalesceTimerRef.current);
    voiceCoalesceTimerRef.current = setTimeout(
      () => fireCoalescedRun(0),
      VOICE_COALESCE_MS,
    );
  }, [resumeListen, stop]);

  useEffect(() => {
    registerVoiceReplyText(() => latestAssistantText());
    return () => registerVoiceReplyText(null);
  }, [latestAssistantText]);

  // Text of the trailing user bubbles (one coalesced prompt may be several).
  const lastUserPrompt = useCallback((): string => {
    const messages = auiRef.current.thread().getState().messages;
    const parts: string[] = [];
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role !== "user") {
        if (parts.length > 0) break;
        continue;
      }
      let text = "";
      for (const part of msg.content as ReadonlyArray<{ type: string; text?: string }>) {
        if (part.type === "text" && part.text) text += part.text;
      }
      if (text) parts.unshift(text);
    }
    return parts.join(" ");
  }, []);

  // Submit a transcript. The adapter has already committed it into the composer
  // via its onSpeech(isFinal) callback, so here we just end dictation and send it;
  // the run-lifecycle effect then speaks the reply and re-arms the mic.
  const submitTranscript = useCallback(() => {
    if (voiceModeRef.current !== "active") return;
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
    const text = composer.getState().text.trim();
    if (!text) {
      // Nothing was said: a no-speech finish, or the half-duplex close the
      // run-start effect does. Neither is a barge-in, so leave a playing reply
      // alone -- cutting here is what truncated it after the first word. Just
      // re-arm; resumeListen defers that until the reply has finished speaking.
      resumeListen();
      return;
    }
    // Decide the thinking filler for this turn now, so the run's request can tell
    // the model what was said. A reply that ended in a question wants a short
    // answer, so no filler there.
    const prevReply = latestAssistantText().trimEnd();
    const prevAskedQuestion = prevReply.endsWith("?");
    const applyFiller = (decision: FillerDecision) => {
      setPendingFiller(decision);
      clearFiller();
      if (decision.say) {
        fillerRef.current = {
          phrase: decision.phrase,
          timer: null,
          queued: false,
          spokenAt: null,
          stage2Done: false,
        };
      }
    };
    setPendingFiller(null);
    clearFiller();
    // The filler model answers inside the coalesce window, before the run starts.
    void fillerFromModel(text, prevAskedQuestion).then((fromModel) => {
      if (fromModel && !auiRef.current.thread().getState().isRunning) applyFiller(fromModel);
    });
    const thread = auiRef.current.thread();
    // Speech while the model is still thinking (no reply text yet) is not always a
    // barge-in: the interject model says whether it changes the request, is a
    // separate question to hold for after the reply, or is nothing to act on.
    if (thread.getState().isRunning && latestAssistantIsReasoning()) {
      const elapsedS = (Date.now() - runStartedAtRef.current) / 1000;
      void interjectAction(lastUserPrompt(), text, elapsedS).then((action) => {
        if (voiceModeRef.current !== "active") return;
        if (action === "ignore" || action === "queue") {
          if (action === "queue") queuedUtteranceRef.current = text;
          composer.setText("");
          resumeListen();
          return;
        }
        commitUtterance(text);
      });
      return;
    }
    commitUtterance(text);
  }, [resumeListen, stop, latestAssistantText, latestAssistantIsReasoning, lastUserPrompt, clearFiller, commitUtterance]);


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
      hasChatModel,
      voiceSlotLoading,
      voiceHearing,
      isPlaying,
      voiceTranscribing,
      isThreadRunning,
      isReasoning,
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
  }, [voiceMode, hasChatModel, voiceSlotLoading, voiceHearing, voiceTranscribing, isThreadRunning, isReasoning, isSpeaking, isPlaying, setVoiceOrbState]);

  // The voice finishing its load is what opens the mic when activateLoop held it shut.
  const wasSlotLoadingRef = useRef(voiceSlotLoading);
  useEffect(() => {
    const was = wasSlotLoadingRef.current;
    wasSlotLoadingRef.current = voiceSlotLoading;
    if (!was || voiceSlotLoading) return;
    if (voiceModeRef.current !== "active" || !hasChatModelRef.current) return;
    resumeListen();
  }, [voiceSlotLoading, resumeListen]);

  // Clear any pending synth-gap timer on unmount so it can't fire after teardown.
  useEffect(() => () => {
    if (synthGapTimerRef.current) clearTimeout(synthGapTimerRef.current);
  }, []);

  // Helper: transition to "active" and start the mic.
  const activateLoop = useCallback(() => {
    setLoopVoiceMode("active");
    voiceModeRef.current = "active";
    setVoiceModeState("active");
    if (!hasChatModelRef.current) return; // orb shows "Select a model"; mic stays shut
    // a reply sent before the voice loads would go unspoken; the slot finishing opens the mic
    if (useChatRuntimeStore.getState().voiceSlotLoading) return;
    if (!auiRef.current.thread().getState().isRunning && !isSpeakingRef.current) {
      auiRef.current.composer().startDictation();
    }
  }, []);

  // A chat model arriving while the orb is up is what unblocks the loop, so it is
  // also what opens the mic: the activate path already ran and returned without
  // arming. Skipped on the first render so a model loaded before voice was opened
  // doesn't double-arm on top of activateLoop.
  const armedForModelRef = useRef(hasChatModel);
  useEffect(() => {
    if (armedForModelRef.current === hasChatModel) return;
    armedForModelRef.current = hasChatModel;
    if (!hasChatModel) return;
    if (voiceModeRef.current !== "active") return;
    resumeListen();
  }, [hasChatModel, resumeListen]);

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
      if (!hasChatModelRef.current) return;
      if (useChatRuntimeStore.getState().voiceSlotLoading) return;
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
        const composer = auiRef.current.composer();
        if (composer.getState().dictation) composer.stopDictation();
        // Begin streaming TTS and feed the reply as it generates so audio starts
        // on the first complete sentence, not after the whole reply.
        beginStreamRef.current();
        streamBegunRef.current = true;
        runStartedAtRef.current = Date.now();
        const filler = fillerRef.current;
        if (filler && !filler.queued && !filler.timer) {
          filler.timer = setTimeout(() => {
            const f = fillerRef.current;
            if (!f || f !== filler) return;
            f.timer = null;
            f.queued = true;
            f.spokenAt = Date.now();
            sayAsideRef.current(f.phrase);
          }, fillerDelayMs());
        }
        if (streamPollRef.current) clearInterval(streamPollRef.current);
        streamPollRef.current = setInterval(() => {
          const f = fillerRef.current;
          const text = latestAssistantText();
          if (f && !f.queued) {
            // Hold the reply until the filler is in the queue.
          } else {
            feedTextRef.current(text);
          }
          if (
            f?.queued &&
            !f.stage2Done &&
            f.spokenAt !== null &&
            !text &&
            Date.now() - f.spokenAt >= STAGE2_AFTER_MS
          ) {
            f.stage2Done = true;
            sayAsideRef.current(pickStage2(latestAssistantToolKinds()));
          }
          setIsReasoning(latestAssistantIsReasoning());
        }, 150);
        // The session starting is what reopens the mic, in barge-in mode (below).
      }
      return;
    }
    if (!_prevRunning) return;
    _prevRunning = false;
    setIsReasoning(false);
    const streamBegun = streamBegunRef.current;
    streamBegunRef.current = false;

    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    // A reply that landed inside the delay: the model was told about the filler,
    // so it still goes first, just without the wait.
    const fillerAtEnd = fillerRef.current;
    if (fillerAtEnd && !fillerAtEnd.queued) sayAsideRef.current(fillerAtEnd.phrase);
    clearFiller();
    if (voiceModeRef.current !== "active") return;
    // Note: no isSpeaking guard here. If a stale utterance is still speaking when
    // a newer reply lands, the newest wins — beginStream above supersedes via
    // stop(). Skipping would silently drop the reply and dead-end the loop.

    const text = latestAssistantText();
    if (!text) {
      resumeListen();
      return;
    }
    if (streamBegun) {
      // Flush the remaining sentences (incl. the trailing one) and finish; the
      // stream's consumer calls resumeListen when playback drains.
      endStreamRef.current(text);
    } else {
      // Voice mode was switched on while this run was already generating, so the
      // run-start branch never opened a session. Speak the finished reply whole;
      // speak() ends the turn through onPlaybackEnd -> resumeListen like a stream.
      speakRef.current(text);
    }
  }, [
    isThreadRunning,
    resumeListen,
    latestAssistantText,
    latestAssistantIsReasoning,
    latestAssistantToolKinds,
    clearFiller,
  ]);

  // Barge-in: while the reply is being spoken the mic stays open. The adapter sees
  // playback (registerVoicePlayback) and only listens for the user cutting in over
  // the echo-cancelled leak; when playback ends it simply carries on as the next
  // turn's listener. Retried because the session the run start closed can read as
  // live for a few frames.
  useEffect(() => {
    if (!isSpeaking) return;
    let cancelled = false;
    const arm = (n: number) => {
      if (cancelled || voiceModeRef.current !== "active") return;
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) {
        if (n < 8) setTimeout(() => arm(n + 1), 60);
        return;
      }
      composer.startDictation();
    };
    arm(0);
    return () => {
      cancelled = true;
    };
  }, [isSpeaking]);

  // No session-end watcher here on purpose. The adapter raises requestVoiceSubmit
  // itself once it has a transcript, on its own end-of-turn AND on a manual stop,
  // so submission is already covered; the session ending is not a turn ending.

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
      // The pick is left alone. Clearing it here used to be safe because null
      // meant "browser voice" -- a working fallback. With the browser voice gone
      // null means silence, so voice would open mute every time. ChatPage seeds a
      // default when nothing is remembered; the slot still loads on activate, not
      // here, so opening the picker costs nothing.
      primeAudio();
    }

    if (next === "off") {
      if (voiceCoalesceTimerRef.current) {
        clearTimeout(voiceCoalesceTimerRef.current);
        voiceCoalesceTimerRef.current = null;
      }
      clearHeldTurn();
      clearFiller();
      setPendingFiller(null);
      queuedUtteranceRef.current = null;
      clearVoiceDecisions();
      stop();
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
    }
    // "configuring": dropdown appears via store; mic stays off.
  }, [stop, primeAudio, clearFiller]);

  // Switching threads carries the loop over instead of ending it: cut the old
  // thread's reply and mic, drop its pending timers, then re-arm on the new one.
  // Voice mode, the picked voice and the loaded slot all survive, the same as the
  // chat model does.
  const threadReset = useCallback(() => {
    if (voiceModeRef.current === "off") return;
    if (voiceCoalesceTimerRef.current) {
      clearTimeout(voiceCoalesceTimerRef.current);
      voiceCoalesceTimerRef.current = null;
    }
    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    streamBegunRef.current = false;
    clearHeldTurn();
    clearFiller();
    setPendingFiller(null);
    queuedUtteranceRef.current = null;
    stop();
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
    // Only "active" re-arms. In "configuring" the mic was never open, and the
    // re-arm would start a loop the user has not started yet.
    if (voiceModeRef.current === "active") resumeListen();
  }, [stop, resumeListen, clearFiller]);

  useEffect(() => {
    registerVoiceThreadReset(threadReset);
    return () => {
      registerVoiceThreadReset(null);
    };
  }, [threadReset]);

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

  // The adapter owns end-of-utterance. A burst over the reply only holds the audio:
  // the run is cut when the utterance proves real (submitTranscript), and a burst
  // that was a cough or a hum lets the reply carry on from where it was held.
  useEffect(() => {
    registerVoiceBargeIn(() => {
      if (voiceModeRef.current !== "active") return;
      if (!isSpeakingRef.current && !isPlayingRef.current) return;
      pausePlayback();
    });
    registerVoiceBargeAbort(() => {
      if (voiceModeRef.current !== "active") return;
      resumePlayback();
    });
    return () => {
      registerVoiceBargeIn(null);
      registerVoiceBargeAbort(null);
    };
  }, [pausePlayback, resumePlayback]);

  submitRef.current = submitTranscript;
  useEffect(() => {
    registerVoiceSubmit(submitTranscript);
    return () => {
      registerVoiceSubmit(null);
    };
  }, [submitTranscript]);

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
// Mute: the loop stays up, the mic frames are dropped. Ctrl+M toggles it (Cmd+M
// is the window's Minimize on macOS, so it is not used).
export const VoiceMuteButton: FC = () => {
  const voiceMode = useChatRuntimeStore((s) => s.voiceMode);
  const muted = useChatRuntimeStore((s) => s.voiceMuted);
  const setVoiceMuted = useChatRuntimeStore((s) => s.setVoiceMuted);
  useEffect(() => {
    setMicMuted(muted);
  }, [muted]);
  useEffect(() => {
    if (voiceMode !== "active") return;
    const onKey = (event: KeyboardEvent) => {
      if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      if (event.key.toLowerCase() !== "m") return;
      event.preventDefault();
      setVoiceMuted(!useChatRuntimeStore.getState().voiceMuted);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [voiceMode, setVoiceMuted]);
  if (voiceMode !== "active") return null;
  const label = muted ? "Unmute microphone (Ctrl+M)" : "Mute microphone (Ctrl+M)";
  return (
    <TooltipIconButton
      tooltip={label}
      aria-label={label}
      aria-pressed={muted}
      variant="ghost"
      className={cn("size-8 rounded-full", muted ? "text-destructive" : "text-foreground")}
      onClick={() => setVoiceMuted(!muted)}
    >
      {muted ? (
        <MicOffIcon className="size-5" />
      ) : (
        <DictateMicIcon className="unsloth-dictate-icon size-5" />
      )}
    </TooltipIconButton>
  );
};

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
