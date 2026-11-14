// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { resetPromptQueuesForThread } from "@/components/assistant-ui/thread";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  VoiceOrb
} from "@/components/assistant-ui/voice-orb";
import {
  StudioVoiceLoopAdapter,
  cancelVoiceSession,
  claimHeldTurnForEdit,
  clearHeldTurn,
  endHeldEdit,
  getHeldTurnText,
  isComposerHeldForEdit,
  isUtteranceInFlight,
  registerComposerTextReader,
  rushVoiceReveal,
  setMicMuted,
  voiceRevealText,
} from "@/features/chat/adapters/studio-voice-loop-adapter";
import { getActiveChatGenerationRuns } from "@/features/chat/api/chat-generation-api";
import { splitStreaming, useTtsPlayer } from "@/features/chat/hooks/use-tts-player";
import { useChatRuntimeStore
} from "@/features/chat/stores/chat-runtime-store";
import { stopChatThread
} from "@/features/chat/utils/stop-chat-thread";
import {
  REPEAT_WINDOW_MS,
  interjectOutcomeFor,
  interjectUnderApproval,
  isHoldOn,
  replyEchoesFiller,
  sharesMostWords,
} from "@/features/chat/voice/laya-outcomes";
import { ORB_IDLE_GRADIENT, orbConfig } from "@/features/chat/voice/orb-config";
import { deriveOrbState } from "@/features/chat/voice/orb-state";
import {
  ASIDE_MIN_GAP_MS, type FillerDecision,
  QUIET_FALLBACK_AFTER_MS,
  SILENT_VERDICT_LIMIT_MS,
  STAGE2_AFTER_MS,
  type ToolKind,
  WRITING_AFTER_MS,
  asidePhrases,
  fillerDelayMs,
  fillerStandIns,
  pickApproval,
  pickQuietFallback,
  pickStage2,
  pickWriting,
  clearPendingVoiceNotes,
  setPendingFiller,
  setPendingReplace,
  toolKindOf,
  voiceNotesEpoch, } from "@/features/chat/voice/thinking-filler";
import { clearVoiceDecisions } from "@/features/chat/voice/voice-decisions";
import {
  describeToolCall,
  getVoiceMode,
  registerToolCallDescriber,
  registerVoiceBargeAbort,
  registerVoiceBargeIn,
  registerVoiceEarlyInterject,
  registerVoiceListen,
  registerVoiceParkedText,
  registerVoicePauseToggle,
  registerVoicePlayback,
  registerVoiceReplyText,
  registerVoiceResume,
  registerVoiceStopSpeaking,
  registerVoiceSubmit,
  registerVoiceThreadReset,
  registerVoiceToggle,
  registerVoiceTypedSubmit,
  setVoiceMode as setLoopVoiceMode,
  voiceRunToStop,
} from "@/features/chat/voice/voice-loop-bridge";
import {
  type InterjectAction,
  fillerFromModel,
  interjectAction,
  reportLayaOutcome,
} from "@/features/chat/voice/voice-models-client";
import { voiceServing } from "@/features/chat/voice/voice-slot";
import { pauseOutputSpectrum } from "@/features/chat/voice/voice-spectrum";
import { voiceTrace } from "@/features/chat/voice/voice-trace";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { MicIcon as DictateMicIcon } from "@/lib/mic-icon";
import { cn } from "@/lib/utils";
import { useAui } from "@assistant-ui/react";
import { MessageSquareIcon, MicOffIcon } from "lucide-react";
import { type FC, useCallback, useEffect, useRef, useState } from "react";
import {
  type EarlyAnswer,
  type EarlyInterject,
  createEarlyInterject,
} from "./early-interject";
import { type HoldToStop, createHoldToStop } from "./hold-to-stop";
import { type OrbDwell, createOrbDwell } from "./orb-dwell";
import { ORB_WIDGET_LONG_PRESS_MS, pendingApproval } from "./orb-widget";
import { isRevealPrefix, isTypedVoiceSubmit } from "./typed-send";

let _prevRunning = false;
// The thread whose run the loop is voicing. Module-level for the same reason: New Chat
// remounts VoiceEngine, and the reset that follows must still find the run to stop.
let _voicedThreadId: string | null = null;

// Tool calls of the voiced chat waiting on Allow / Deny, filtered as the floating orb does: a
// Compare pane or a background chat raising its own card is not voice mode's to speak about.
function voicedApprovalCount(s: ReturnType<typeof useChatRuntimeStore.getState>): number {
  const pending = pendingApproval(s.toolConfirmations, s.activeThreadId, (id) => describeToolCall(id) !== null);
  return pending ? pending.more + 1 : 0;
}

const VOICE_COALESCE_MS = 650;
// After a barge-in cut a run that had not answered yet: the user tends to keep
// going ("can you hear me?" ... "hello?"), and each restart threw the model's
// thinking away, so the next run waits longer for the rest to stack up.
const VOICE_RESTART_COALESCE_MS = 1800;
// A bare "hold on" / "wait" says more is coming: give the person time to think it.
const VOICE_HOLD_COALESCE_MS = 6000;
// The window never closes on someone mid-sentence; past this it gives up waiting.
const VOICE_HEARING_HOLD_MAX_MS = 30000;
// thread.cancelRun() is asynchronous, so the old run can still read as running
// when the coalesce window closes. Rather than give up and leave the appended
// bubbles unanswered, re-check every VOICE_COALESCE_MS up to this many times
// (~10s) before concluding the run will never settle.
const VOICE_COALESCE_MAX_RETRIES = 15;
// How long to wait for the backend to settle a cancelled run before starting the next.
const VOICE_RUN_SETTLE_MAX_MS = 6000;
const VOICE_RUN_SETTLE_POLL_MS = 150;
// How long playback may pause between sentences before the orb admits a real
// synth stall and turns lilac ("Generating voice"). The swap between two already
// synthesized clips is well under this, so ordinary sentence boundaries hold the
// blue "Speaking" state instead of flickering to lilac; only a gap that persists
// past this (synthesis genuinely behind playback) shows lilac.
const SYNTH_GAP_MS = 350;

export { VoiceOrb };
export { VoicePanel } from "@/components/assistant-ui/voice-panel";
export { loadVoice, servedVoice, unloadVoice, voiceServing } from "@/features/chat/voice/voice-slot";

registerVoiceListen(() =>
  StudioVoiceLoopAdapter.isSupported() ? new StudioVoiceLoopAdapter().listen() : null,
);

export const VoiceEngine: FC = () => {
  const aui = useAui();
  const [voiceMode, setVoiceModeState] = useState(getVoiceMode());
  const voiceCoalesceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const voiceModeRef = useRef(getVoiceMode());
  const isSpeakingRef = useRef(false);
  const isPlayingRef = useRef(false);
  const synthGapTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Holds each displayed orb state for its minimum dwell before the next one lands.
  const orbDwellRef = useRef<OrbDwell | null>(null);
  const auiRef = useRef(aui);
  auiRef.current = aui;

  // The store, not useAuiState: it survives the composer remount on first send.
  const isThreadRunning = useChatRuntimeStore((s) =>
    s.activeThreadId ? Boolean(s.runningByThreadId[s.activeThreadId]) : false,
  );
  const hasChatModel = useChatRuntimeStore((s) => Boolean(s.params.checkpoint));
  const hasChatModelRef = useRef(hasChatModel);
  hasChatModelRef.current = hasChatModel;
  const rearmSeqRef = useRef(0);
  const resumeListen = useCallback(() => {
    // A voice still loading no longer shuts the mic: listening never needed the
    // voice, and a shut mic mid-swap lost what was said and flushed a held
    // fragment as a whole turn. What waits for the voice is the SEND (below).
    // After a session ends (no-speech finish, silence timer), the composer's
    // dictation field can lag a few frames before clearing. Clicking Dictate
    // while it is still set would toggle dictation OFF and kill the loop, so
    // poll briefly for it to clear before re-arming; as a last resort, re-arm
    // anyway rather than leave the loop dead.
    const MAX_ATTEMPTS = 5;
    const RETRY_MS = 50;

    const clickDictate = () => {
      const fresh = auiRef.current.composer();
      // Already listening: startDictation() on a live session tears the mic down.
      if (fresh.getState().dictation) return;
      // Start from an empty box: assistant-ui appends every transcript onto the
      // text present at startDictation, and the adapter emits the whole turn so
      // far (a held fragment included), so leftover text here would double up.
      // A turn the user clicked into stays: what they say next is appended to it.
      if (fresh.getState().text && !isComposerHeldForEdit()) fresh.setText("");
      // Start dictation via the composer runtime, NOT by clicking the Dictate
      // button: in voice mode that button is hidden/replaced by the orb, so a
      // DOM query returns null and the mic never opens (silent green orb).
      fresh.startDictation();
    };

    const seq = ++rearmSeqRef.current;

    const attempt = (n: number) => {
      if (voiceModeRef.current !== "active") return;
      if (!hasChatModelRef.current) return;
      if (rearmSeqRef.current !== seq) return;
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
      clickDictate();
    };

    attempt(1);
  }, []);

  // A click on the composer while a fragment is held means the user is about to edit
  // it: from then on it waits for their send. Pointerdown only, since programmatic
  // focus and caret moves must not count, and one passive listener for the page.
  useEffect(() => {
    if (voiceMode !== "active") return;
    registerComposerTextReader(() => auiRef.current.composer().getState().text);
    const onPointerDown = (e: PointerEvent) => {
      if (!getHeldTurnText()) return;
      if (!(e.target instanceof Element) || !e.target.closest(".aui-composer-input")) return;
      const composer = auiRef.current.composer();
      // The fragment is the live session's interim text; make it the box's own so
      // the user's edits and later speech build on it. setText skips an equal value.
      const text = composer.getState().text;
      composer.setText("");
      composer.setText(text);
      claimHeldTurnForEdit();
    };
    document.addEventListener("pointerdown", onPointerDown, { capture: true, passive: true });
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, { capture: true });
      registerComposerTextReader(null);
    };
  }, [voiceMode]);

  // Speech that landed while the model was still thinking and the interject model
  // called a separate question: held here and sent, as one, once the reply has been heard.
  const queuedUtteranceRef = useRef<string[]>([]);
  // Turns finished while the voice was still loading; sent as one once it is up.
  const pendingForVoiceRef = useRef<string[]>([]);
  const submitRef = useRef<(text?: string) => void>(() => {});
  const runStartedAtRef = useRef(0);
  const warmAsidesRef = useRef<(texts: readonly string[]) => Promise<void>>(async () => {});
  const onPlaybackEnd = useCallback(() => {
    const queued = queuedUtteranceRef.current;
    if (queued.length && voiceModeRef.current === "active") {
      queuedUtteranceRef.current = [];
      submitRef.current(queued.join(" "));
      return;
    }
    // Idle again: top up the aside cache (a stop mid-warm leaves phrases missing).
    void warmAsidesRef.current(asidePhrases());
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
    warmAsides,
    stop,
    primeAudio,
    pausePlayback,
    resumePlayback,
    holdPlayback,
  } = useTtsPlayer(onPlaybackEnd);
  isSpeakingRef.current = isSpeaking;
  isPlayingRef.current = isPlaying;
  useEffect(() => {
    registerVoicePlayback(() => isSpeakingRef.current || isPlayingRef.current);
    return () => registerVoicePlayback(null);
  }, []);
  const beginStreamRef = useRef(beginStream);
  beginStreamRef.current = beginStream;
  const feedTextRef = useRef(feedText);
  feedTextRef.current = feedText;
  const endStreamRef = useRef(endStream);
  endStreamRef.current = endStream;
  const speakRef = useRef(speak);
  speakRef.current = speak;
  const streamBegunRef = useRef(false);
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastFallbackAtRef = useRef(0);
  // The filler model answered (say or quiet). A quiet vote is the answer, so the
  // quiet fallback only bridges a turn the model never answered at all.
  const fillerAnsweredRef = useRef(false);
  // The reply last handed to the voice, so a thread that blinks back to "running"
  // with no new message cannot cut it and speak it again from the top.
  const spokenReplyRef = useRef<{ id: string; text: string } | null>(null);
  const runFlickerRef = useRef(false);
  const lastAsideAtRef = useRef(0);
  const sayAsideRef = useRef(sayAside);
  sayAsideRef.current = (...args: Parameters<typeof sayAside>) => {
    lastAsideAtRef.current = Date.now();
    return sayAside(...args);
  };
  // Whether anything was said aloud for the current turn: the first aside must not be "still on it".
  const spokenThisTurnRef = useRef(false);
  warmAsidesRef.current = warmAsides;
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
    /** The decision log row, cleared once its outcome is reported. */
    logId: string | null;
  } | null>(null);
  const clearFiller = useCallback(() => {
    const f = fillerRef.current;
    if (f?.timer) clearTimeout(f.timer);
    fillerRef.current = null;
  }, []);
  // An ignored or queued utterance, watched for the user saying it again inside
  // REPEAT_WINDOW_MS: that is the one signal the interrupt verdict was wrong.
  const pendingInterjectRef = useRef<{
    id: string;
    verdict: "ignore" | "queue";
    text: string;
    at: number;
    timer: ReturnType<typeof setTimeout>;
  } | null>(null);
  const settleInterject = useCallback((later: string | null) => {
    const p = pendingInterjectRef.current;
    if (!p) return;
    pendingInterjectRef.current = null;
    clearTimeout(p.timer);
    const gapMs = Date.now() - p.at;
    reportLayaOutcome(p.id, interjectOutcomeFor(p.verdict, p.text, later, gapMs), { gap_ms: gapMs });
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
  const commitUtterance = useCallback((said: string) => {
    const composer = auiRef.current.composer();
    // Utterances the interrupt model held for after the reply go with this one:
    // a barge-in cuts playback without its end ever firing, so they were turning
    // up minutes later as a stale prompt of their own. Repeats collapse.
    const held = queuedUtteranceRef.current.filter((q) => q !== said);
    queuedUtteranceRef.current = [];
    const text = held.length ? [...held, said].join(" ") : said;
    if (held.length) voiceTrace("engine.foldHeld", { held, text });
    // Barge-in: the instant the user speaks, cut the audio and the in-flight run.
    // This utterance supersedes whatever the model was saying. Cover the tail
    // clip (isPlaying) too, not just an active streaming session (isSpeaking).
    voiceTrace("engine.commitUtterance", { text, speaking: isSpeakingRef.current, playing: isPlayingRef.current });
    if (isSpeakingRef.current || isPlayingRef.current) stop();
    const thread = auiRef.current.thread();
    const cutRun = thread.getState().isRunning;
    if (cutRun) {
      // The barge-in cuts the filler with the reply.
      const f = fillerRef.current;
      if (f?.logId) {
        reportLayaOutcome(f.logId, "cut", { queued: f.queued });
        f.logId = null;
      }
      try {
        voiceTrace("engine.cancelRun", {});
        thread.cancelRun();
      } catch {
      }
    }
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

    // First message: the append + startRun path goes stale across the first-send remount.
    if (thread.getState().messages.length === 0) {
      // send() takes the box, not `text`: parked turns were never in it (the re-arm
      // clears it), so an empty box here sent nothing at all.
      composer.setText(text);
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
    }

    if (!appended) {
      // Fallback: old immediate-send behavior.
      composer.setText(text);
      composer.send();
      composer.setText("");
      setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
      return;
    }

    composer.setText("");
    setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
    resumeListen();
    const armedAt = Date.now();
    const fireCoalescedRun = (attempt: number) => {
      voiceCoalesceTimerRef.current = null;
      if (voiceModeRef.current !== "active") return;
      // Still talking (or the last words are being transcribed): the next bubble is
      // on its way, so starting now would run without it and be cut again.
      const rt = useChatRuntimeStore.getState();
      if ((rt.voiceHearing || rt.voiceTranscribing) && Date.now() - armedAt < VOICE_HEARING_HOLD_MAX_MS) {
        voiceTrace("engine.coalesceWaitHearing", {});
        voiceCoalesceTimerRef.current = setTimeout(() => fireCoalescedRun(attempt), VOICE_COALESCE_MS);
        return;
      }
      const t = auiRef.current.thread();
      if (t.getState().isRunning) {
        // cancelRun() has not settled yet: retry rather than leave the bubbles unanswered.
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
      if (!last || last.role !== "user") return;
      // The barge-in's cancel is acknowledged here before the backend has settled
      // the old run, and a new run on the thread while the old one is still
      // "cancelling" is refused outright. Ask the server, not the local state.
      const threadId = useChatRuntimeStore.getState().activeThreadId;
      const turn = voiceNotesEpoch();
      const start = () => {
        if (voiceModeRef.current !== "active") return;
        // A thread switch, Stop or typed send while the old run settled let this turn go.
        if (voiceNotesEpoch() !== turn) return;
        if (useChatRuntimeStore.getState().activeThreadId !== threadId) return;
        try {
          t.startRun({ parentId: last.id });
        } catch {
          // startRun unsupported -- the bubbles are already shown; nothing to do.
        }
      };
      if (!threadId) {
        start();
        return;
      }
      void (async () => {
        const deadline = Date.now() + VOICE_RUN_SETTLE_MAX_MS;
        for (;;) {
          let active = 0;
          try {
            active = (await getActiveChatGenerationRuns(threadId)).length;
          } catch {
            break;
          }
          if (active === 0 || Date.now() >= deadline) {
            if (active) voiceTrace("engine.runStillActive", { threadId, active });
            break;
          }
          await new Promise((r) => setTimeout(r, VOICE_RUN_SETTLE_POLL_MS));
        }
        start();
      })();
    };
    if (voiceCoalesceTimerRef.current) clearTimeout(voiceCoalesceTimerRef.current);
    const holdOn = isHoldOn(said);
    if (holdOn) voiceTrace("engine.holdOn", { text: said });
    voiceCoalesceTimerRef.current = setTimeout(
      () => fireCoalescedRun(0),
      holdOn ? VOICE_HOLD_COALESCE_MS : cutRun ? VOICE_RESTART_COALESCE_MS : VOICE_COALESCE_MS,
    );
  }, [resumeListen, stop]);

  useEffect(() => {
    registerVoiceReplyText(() => latestAssistantText());
    registerVoiceParkedText(() => pendingForVoiceRef.current.join(" "));
    return () => {
      registerVoiceReplyText(null);
      registerVoiceParkedText(null);
    };
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
    // One bubble per line, the shape the interject model was trained on.
    return parts.join("\n");
  }, []);

  const sendTranscriptRef = useRef<(text: string, typed?: boolean) => void>(() => {});
  const sendTranscript = useCallback(
    (text: string, typed?: boolean) => sendTranscriptRef.current(text, typed),
    [],
  );
  // Submit a transcript: the adapter's own when it passes one (the composer can be
  // mid-reveal or already cleared by then), else whatever is in the box. Ends
  // dictation and sends; the run-lifecycle effect then speaks the reply and
  // re-arms the mic.
  const submitTranscript = useCallback((spoken?: string, typed?: boolean) => {
    if (voiceModeRef.current !== "active") return;
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
    let text = (spoken ?? composer.getState().text).trim();
    voiceTrace("engine.submit", { text });
    const pending = pendingForVoiceRef.current;
    if (useChatRuntimeStore.getState().voiceSlotLoading) {
      // A reply sent now would land before the voice can speak it. Park the turn;
      // the slot finishing its load sends everything parked as one prompt.
      if (text) {
        voiceTrace("engine.holdForVoice", { text });
        pending.push(text);
      }
      resumeListen();
      return;
    }
    if (pending.length) {
      text = [...pending, text].filter(Boolean).join(" ");
      pendingForVoiceRef.current = [];
    }
    if (!text) {
      // Nothing was said: a no-speech finish, or the half-duplex close the
      // run-start effect does. Neither is a barge-in, so leave a playing reply
      // alone -- cutting here is what truncated it after the first word. Just
      // re-arm; resumeListen defers that until the reply has finished speaking.
      resumeListen();
      return;
    }
    // The store can believe the voice is loaded after the backend has dropped it
    // (an idle unload, a restart): the reply then ran before the voice came back.
    // One status read settles it; a missing voice parks the turn and reloads.
    const store = useChatRuntimeStore.getState();
    const ensure = store.ensureVoiceSlotLoaded;
    const sendNow = () => sendTranscript(text, typed);
    if (!ensure || isSpeakingRef.current) {
      sendNow();
      return;
    }
    const sent = text;
    // Voice off, a thread switch, Stop or a typed send while the status is read let this turn go.
    const turn = voiceNotesEpoch();
    void voiceServing(store.selectedVoiceModelId).then(
      (serving) => {
        if (voiceModeRef.current !== "active" || voiceNotesEpoch() !== turn) return;
        if (serving) {
          sendNow();
          return;
        }
        voiceTrace("engine.holdForVoice", { text: sent, why: "not serving" });
        pendingForVoiceRef.current.push(sent);
        useChatRuntimeStore.getState().setVoiceSlotLoaded(false);
        void ensure();
        resumeListen();
      },
      () => {
        if (voiceModeRef.current === "active" && voiceNotesEpoch() === turn) sendNow();
      },
    );
  }, [resumeListen, sendTranscript]);

  // The send itself, once the text is final and the voice is known to be up.
  function sendTranscriptImpl(text: string, typed?: boolean) {
    const composer = auiRef.current.composer();
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
          logId: decision.logId ?? null,
        };
      }
    };
    // Any new utterance settles the interrupt verdict before it.
    settleInterject(text);
    // A fragment of this already cut the run: this is that restart, with every word.
    const early = earlyInterjectRef.current?.complete();
    setPendingFiller(null);
    clearFiller();
    // The filler model answers inside the coalesce window, before the run starts.
    fillerAnsweredRef.current = false;
    const notesEpoch = voiceNotesEpoch();
    void fillerFromModel(text, prevAskedQuestion).then((fromModel) => {
      // Voice off, a thread switch or a typed send since: the verdict is for a turn that's gone.
      if (notesEpoch !== voiceNotesEpoch()) return;
      if (fromModel) fillerAnsweredRef.current = true;
      if (fromModel && !auiRef.current.thread().getState().isRunning) applyFiller(fromModel);
    });
    const thread = auiRef.current.thread();
    if (early?.restarted) {
      voiceTrace("engine.earlyRestart", { text, kind: early.kind });
      setPendingReplace(early.kind === "replace");
      commitUtterance(text);
      return;
    }
    // Speech while the model is still thinking (no reply text yet) is not always a
    // barge-in: the interject model says whether it changes the request, is a
    // separate question to hold for after the reply, or is nothing to act on.
    // Typed text is meant: never held or ignored by the interject model.
    if (!typed && thread.getState().isRunning && latestAssistantIsReasoning()) {
      const elapsedS = (Date.now() - runStartedAtRef.current) / 1000;
      const fromThread = useChatRuntimeStore.getState().activeThreadId;
      const turn = voiceNotesEpoch();
      void interjectAction(lastUserPrompt(), text, elapsedS).then((answer) => {
        const verdict = answer?.action ?? null;
        const { action, sameAsk, continuing, awaitingApproval } = overrideInterject(text, verdict);
        voiceTrace("engine.interject", { action, verdict, text, sameAsk, continuing, awaitingApproval });
        if (sameAsk) {
          sayAsideRef.current(pickStage2(latestAssistantToolKinds(), Math.random, !spokenThisTurnRef.current));
          spokenThisTurnRef.current = true;
        }
        if (answer?.id) {
          if (action === "ignore" || action === "queue") {
            settleInterject(null);
            const id = answer.id;
            pendingInterjectRef.current = {
              id,
              verdict: action,
              text,
              at: Date.now(),
              timer: setTimeout(() => {
                if (pendingInterjectRef.current?.id === id) settleInterject(null);
              }, REPEAT_WINDOW_MS),
            };
          } else {
            reportLayaOutcome(answer.id, "ok", { overridden: action !== verdict });
          }
        }
        if (voiceModeRef.current !== "active") return;
        // A thread switch, Stop or typed send while this was decided let the turn go.
        if (voiceNotesEpoch() !== turn) return;
        if (useChatRuntimeStore.getState().activeThreadId !== fromThread) return;
        if (action === "ignore" || action === "queue") {
          if (action === "queue" && !queuedUtteranceRef.current.includes(text)) {
            queuedUtteranceRef.current.push(text);
          }
          composer.setText("");
          // The barge-in held the reply; nothing cut it, so let it carry on. Without
          // this the playback context stayed suspended and the loop looked stuck.
          resumePlayback();
          resumeListen();
          return;
        }
        setPendingReplace(answer?.kind === "replace");
        commitUtterance(text);
      });
      return;
    }
    commitUtterance(text);
  }
  sendTranscriptRef.current = sendTranscriptImpl;

  // The loop's own overrides on the interject verdict, the same for a fragment and a
  // whole utterance.
  function overrideInterject(text: string, verdict: InterjectAction | null) {
    // Saying the held thing again is insisting on it, not a second hold.
    const repeated = queuedUtteranceRef.current.includes(text);
    // Asking the pending question again ("can you hear me?" over its own
    // reply being thought out) is impatience, not a new request: restarting
    // threw the thinking away and made the wait longer every time.
    const sameAsk = sharesMostWords(lastUserPrompt(), text);
    // Something is already held for after this reply, so this is the rest of it
    // ("How many?" then "Too many cannolis"): it joins the hold, never dropped
    // on its own, and the two go as one when the reply ends.
    const continuing = queuedUtteranceRef.current.length > 0 && !repeated;
    const awaitingApproval = voicedApprovalCount(useChatRuntimeStore.getState()) > 0;
    const action = interjectUnderApproval(
      sameAsk
        ? "ignore"
        : verdict !== "restart" && repeated
          ? "restart"
          : continuing && verdict === "ignore"
            ? "queue"
            : verdict,
      awaitingApproval,
    );
    return { action, sameAsk, continuing, awaitingApproval };
  }

  // A fragment the turn judge is holding, over a run that is thinking or writing: asked
  // about now rather than once the whole utterance is in, so a restart cuts the run
  // seconds earlier. The sameAsk aside and the queue wait for the whole utterance.
  async function askEarly(fragment: string): Promise<EarlyAnswer | null> {
    if (voiceModeRef.current !== "active") return null;
    if (!auiRef.current.thread().getState().isRunning) return null;
    // Writing the answer: a real utterance cuts it, as the finished one would.
    if (!latestAssistantIsReasoning()) return { verdict: "restart", kind: null };
    const elapsedS = (Date.now() - runStartedAtRef.current) / 1000;
    const answer = await interjectAction(lastUserPrompt(), fragment, elapsedS);
    if (voiceModeRef.current !== "active" || !auiRef.current.thread().getState().isRunning) {
      return null;
    }
    const verdict = answer?.action ?? null;
    const { action } = overrideInterject(fragment, verdict);
    if (action === "restart" && answer?.id) {
      reportLayaOutcome(answer.id, "ok", { overridden: action !== verdict, early: true });
    }
    return { verdict: action, kind: answer?.kind ?? null };
  }
  function cutEarly({ gapMs, kind }: { fragment: string; kind: string | null; gapMs: number }) {
    if (voiceModeRef.current !== "active") return;
    voiceTrace("engine.cancelRun", { early: true, gapMs, kind });
    if (isSpeakingRef.current || isPlayingRef.current) stop();
    const f = fillerRef.current;
    if (f?.logId) {
      reportLayaOutcome(f.logId, "cut", { queued: f.queued });
      f.logId = null;
    }
    try {
      auiRef.current.thread().cancelRun();
    } catch {
      // Run may have ended between the check and here.
    }
  }
  const earlyAskRef = useRef(askEarly);
  earlyAskRef.current = askEarly;
  const earlyCutRef = useRef(cutEarly);
  earlyCutRef.current = cutEarly;
  const earlyInterjectRef = useRef<EarlyInterject | null>(null);
  if (!earlyInterjectRef.current) {
    earlyInterjectRef.current = createEarlyInterject({
      ask: (fragment) => earlyAskRef.current(fragment),
      cancel: (info) => earlyCutRef.current(info),
      now: () => Date.now(),
      trace: voiceTrace,
    });
  }
  useEffect(() => {
    registerVoiceEarlyInterject((fragment) => earlyInterjectRef.current?.fragment(fragment));
    return () => registerVoiceEarlyInterject(null);
  }, []);

  // The floating orb's approval popup names the waiting call from the thread, newest first.
  useEffect(() => {
    registerToolCallDescriber((toolCallId) => {
      const messages = auiRef.current.thread().getState().messages;
      for (let i = messages.length - 1; i >= 0; i--) {
        for (const part of messages[i].content as ReadonlyArray<{
          type: string;
          toolCallId?: string;
          toolName?: string;
          args?: unknown;
        }>) {
          if (part.type === "tool-call" && part.toolCallId === toolCallId) {
            return { toolName: part.toolName ?? "", args: part.args };
          }
        }
      }
      return null;
    });
    return () => registerToolCallDescriber(null);
  }, []);
  const commitUtteranceRef = useRef(commitUtterance);
  commitUtteranceRef.current = commitUtterance;


  // Sync derived orb state to the store so VoiceOrb can read it without prop-drilling.
  // The priority order itself lives in deriveOrbState, which is pure and tested.
  // A tool call paused for Allow/Deny: say so once per card, after whatever is playing.
  const confirmationCount = useChatRuntimeStore(voicedApprovalCount);
  const confirmationsSeenRef = useRef(0);
  useEffect(() => {
    const before = confirmationsSeenRef.current;
    confirmationsSeenRef.current = confirmationCount;
    if (confirmationCount <= before || voiceModeRef.current !== "active") return;
    // Speech held for after the reply would sit behind the card until someone taps it.
    // Send it now: the cancel denies the parked call and the new run sees both asks.
    const held = queuedUtteranceRef.current;
    if (held.length) {
      voiceTrace("engine.heldAtApproval", { held: [...held] });
      commitUtteranceRef.current(held[held.length - 1]);
      return;
    }
    const line = pickApproval();
    let tries = 0;
    const timer = setInterval(() => {
      tries++;
      if (voicedApprovalCount(useChatRuntimeStore.getState()) === 0) {
        clearInterval(timer);
        return;
      }
      // Not over the user either: the line landing mid-sentence cut them off.
      const { voiceHearing, voiceTranscribing } = useChatRuntimeStore.getState();
      if ((isPlayingRef.current || voiceHearing || voiceTranscribing) && tries < 40) return;
      clearInterval(timer);
      voiceTrace("engine.approvalAside", { text: line, waitedMs: tries * 250 });
      spokenThisTurnRef.current = true;
      sayAsideRef.current(line, { patient: true });
    }, 250);
    return () => clearInterval(timer);
  }, [confirmationCount]);

  const setVoiceOrbState = useChatRuntimeStore((s) => s.setVoiceOrbState);
  const voiceSlotLoading = useChatRuntimeStore((s) => s.voiceSlotLoading);
  const voiceTranscribing = useChatRuntimeStore((s) => s.voiceTranscribing);
  const voiceHearing = useChatRuntimeStore((s) => s.voiceHearing);
  const voicePaused = useChatRuntimeStore((s) => s.voicePaused);
  useEffect(() => {
    const dwell = (orbDwellRef.current ??= createOrbDwell((state) => {
      // Nothing may light the orb once the loop is off, whatever was queued before.
      const shown = state !== null && voiceModeRef.current === "off" ? null : state;
      voiceTrace("orb.state", { state: shown, raw: state });
      setVoiceOrbState(shown);
    }));
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
      voicePaused,
    });
    if (decision.kind === "set") {
      clearSynthGap();
      dwell.push(decision.state);
      return;
    }
    if (synthGapTimerRef.current) return;
    synthGapTimerRef.current = setTimeout(() => {
      synthGapTimerRef.current = null;
      if (
        voiceModeRef.current === "active" &&
        isSpeakingRef.current &&
        !isPlayingRef.current
      ) {
        dwell.push("synthesizing");
      }
    }, SYNTH_GAP_MS);
  }, [voiceMode, hasChatModel, voiceSlotLoading, voiceHearing, voiceTranscribing, isThreadRunning, isReasoning, isSpeaking, isPlaying, voicePaused, setVoiceOrbState]);

  const wasSlotLoadingRef = useRef(voiceSlotLoading);
  useEffect(() => {
    const was = wasSlotLoadingRef.current;
    wasSlotLoadingRef.current = voiceSlotLoading;
    if (!was || voiceSlotLoading) return;
    if (voiceModeRef.current !== "active" || !hasChatModelRef.current) return;
    // The voice is up and idle: cache the fixed asides so the first one is instant.
    void warmAsides(asidePhrases());
    // A failed load also ends loading: re-sending would re-park and reload in a loop.
    if (pendingForVoiceRef.current.length && useChatRuntimeStore.getState().voiceSlotLoaded) {
      // Something is being said right now: cutting the session here lost a turn
      // mid-transcription. Its own finish (or no-speech resume) sends the lot.
      if (isUtteranceInFlight()) return;
      submitRef.current("");
      return;
    }
    resumeListen();
  }, [voiceSlotLoading, resumeListen, warmAsides]);

  // The slot can already be serving when voice mode starts (loaded from the picker
  // or the API first), so the loading transition above never fires: warm here too.
  const voiceSlotLoaded = useChatRuntimeStore((s) => s.voiceSlotLoaded);
  useEffect(() => {
    if (voiceMode !== "active" || !voiceSlotLoaded || voiceSlotLoading) return;
    void warmAsides(asidePhrases());
  }, [voiceMode, voiceSlotLoaded, voiceSlotLoading, warmAsides]);

  // Clear any pending synth-gap or dwell timer on unmount so it can't fire after teardown.
  useEffect(() => () => {
    if (synthGapTimerRef.current) clearTimeout(synthGapTimerRef.current);
    orbDwellRef.current?.dispose();
  }, []);

  const activateLoop = useCallback(() => {
    // Turning voice on starts clean: nothing held from a session before it carries in.
    if (voiceModeRef.current !== "active") clearHeldTurn();
    setLoopVoiceMode("active");
    voiceModeRef.current = "active";
    setVoiceModeState("active");
    if (!hasChatModelRef.current) return;
    if (useChatRuntimeStore.getState().voiceSlotLoading) return;
    if (!auiRef.current.thread().getState().isRunning && !isSpeakingRef.current) {
      auiRef.current.composer().startDictation();
    }
  }, []);

  const armedForModelRef = useRef(hasChatModel);
  useEffect(() => {
    if (armedForModelRef.current === hasChatModel) return;
    armedForModelRef.current = hasChatModel;
    if (!hasChatModel) return;
    if (voiceModeRef.current !== "active") return;
    resumeListen();
  }, [hasChatModel, resumeListen]);

  // Deferred a tick: a New Chat's voice reset runs in the same commit, after this effect.
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
  }, []);

  const storeVoiceMode = useChatRuntimeStore((s) => s.voiceMode);
  useEffect(() => {
    if (
      storeVoiceMode === "active" &&
      voiceModeRef.current === "configuring"
    ) {
      activateLoop();
    }
  }, [storeVoiceMode, activateLoop]);

  useEffect(() => {
    if (isThreadRunning) {
      _prevRunning = true;
      // The user sent the turn they were editing: the next one auto-sends again.
      endHeldEdit();
      earlyInterjectRef.current?.reset();
      const msgs = auiRef.current.thread().getState().messages;
      const tail = msgs.length ? msgs[msgs.length - 1] : null;
      const spoken = spokenReplyRef.current;
      if (spoken && tail?.role === "assistant" && tail.id === spoken.id && latestAssistantText() === spoken.text) {
        voiceTrace("engine.runFlicker", { id: tail.id });
        runFlickerRef.current = true;
        return;
      }
      runFlickerRef.current = false;
      if (voiceModeRef.current === "active") {
        const composer = auiRef.current.composer();
        if (composer.getState().dictation) composer.stopDictation();
        beginStreamRef.current();
        streamBegunRef.current = true;
        _voicedThreadId = useChatRuntimeStore.getState().activeThreadId;
        runStartedAtRef.current = Date.now();
        spokenThisTurnRef.current = false;
        const filler = fillerRef.current;
        if (filler && !filler.queued && !filler.timer) {
          filler.timer = setTimeout(() => {
            const f = fillerRef.current;
            if (!f || f !== filler) return;
            f.timer = null;
            f.queued = true;
            f.spokenAt = Date.now();
            spokenThisTurnRef.current = true;
            sayAsideRef.current(f.phrase, { standIns: fillerStandIns() });
          }, fillerDelayMs());
        }
        if (streamPollRef.current) clearInterval(streamPollRef.current);
        // Per run: the bridge said when the classifier chose silence, and the one
        // said while the answer is being written.
        let quietFallbackAt: number | null = null;
        let writingDone = false;
        // One bridge per stretch of restarts, not one per restart ("one sec" three times).
        const FALLBACK_REPEAT_GAP_MS = 8000;
        streamPollRef.current = setInterval(() => {
          const f = fillerRef.current;
          const text = latestAssistantText();
          if (f && !f.queued) {
            // Hold the reply until the filler is in the queue.
          } else {
            feedTextRef.current(text);
          }
          const now = Date.now();
          // Waiting on the Allow/Deny card: the approval line said what is needed, so no
          // "working on it" style asides on top of it.
          const awaitingApproval = voicedApprovalCount(useChatRuntimeStore.getState()) > 0;
          const asideGapOk = now - lastAsideAtRef.current >= ASIDE_MIN_GAP_MS;
          if (
            !awaitingApproval &&
            asideGapOk &&
            f?.queued &&
            !f.stage2Done &&
            f.spokenAt !== null &&
            !text &&
            now - f.spokenAt >= STAGE2_AFTER_MS
          ) {
            f.stage2Done = true;
            sayAsideRef.current(pickStage2(latestAssistantToolKinds()));
          }
          // The classifier chose silence and nothing has played: the quiet reads
          // as a miss after a couple of seconds, so bridge it. A "none" verdict holds
          // only while the reply is quick; a tool call or a long think still gets one line.
          const toolKinds = latestAssistantToolKinds();
          const silenceOverrun =
            !text && (toolKinds.length > 0 || now - runStartedAtRef.current >= SILENT_VERDICT_LIMIT_MS);
          if (
            !awaitingApproval &&
            asideGapOk &&
            !f &&
            (!fillerAnsweredRef.current || silenceOverrun) &&
            quietFallbackAt === null &&
            !isPlayingRef.current &&
            now - runStartedAtRef.current >= QUIET_FALLBACK_AFTER_MS &&
            now - lastFallbackAtRef.current >= FALLBACK_REPEAT_GAP_MS
          ) {
            quietFallbackAt = now;
            lastFallbackAtRef.current = now;
            spokenThisTurnRef.current = true;
            voiceTrace("engine.fillerFallback", {
              sinceStart: now - runStartedAtRef.current,
              overrode: fillerAnsweredRef.current,
              tools: toolKinds.length,
            });
            sayAsideRef.current(toolKinds.length > 0 ? pickStage2(toolKinds) : pickQuietFallback());
          }
          // Text is arriving (the writing step) but its first sentence is still on
          // the way, and the last aside was a while ago: "almost there" is true now,
          // and the aside lands ahead of that first sentence.
          const lastAsideAt = quietFallbackAt ?? f?.spokenAt ?? runStartedAtRef.current;
          if (
            !awaitingApproval &&
            asideGapOk &&
            !writingDone &&
            text &&
            splitStreaming(text).complete.length === 0 &&
            !isPlayingRef.current &&
            now - lastAsideAt >= WRITING_AFTER_MS
          ) {
            writingDone = true;
            voiceTrace("engine.fillerWriting", { sinceAside: now - lastAsideAt });
            sayAsideRef.current(pickWriting());
          }
          setIsReasoning(latestAssistantIsReasoning());
        }, 150);
      }
      return;
    }
    if (!_prevRunning) return;
    _prevRunning = false;
    if (_voicedThreadId && !useChatRuntimeStore.getState().runningByThreadId[_voicedThreadId]) {
      _voicedThreadId = null;
    }
    if (runFlickerRef.current) {
      runFlickerRef.current = false;
      const spoken = spokenReplyRef.current;
      const now = latestAssistantText();
      if (!spoken || now === spoken.text) return;
      // The same message grew (a tool result, a resumed run): speak only what is new.
      const added = now.startsWith(spoken.text) ? now.slice(spoken.text.length) : now;
      spokenReplyRef.current = { ...spoken, text: now };
      if (voiceModeRef.current === "active" && added.trim()) speakRef.current(added);
      return;
    }
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
    if (fillerAtEnd && !fillerAtEnd.queued) sayAsideRef.current(fillerAtEnd.phrase, { standIns: fillerStandIns() });
    if (fillerAtEnd?.logId) {
      const outcome = !fillerAtEnd.queued
        ? "too_late"
        : replyEchoesFiller(latestAssistantText(), fillerAtEnd.phrase)
          ? "echoed"
          : "spoken";
      reportLayaOutcome(fillerAtEnd.logId, outcome);
    }
    clearFiller();
    if (voiceModeRef.current !== "active") return;
    // No isSpeaking guard: beginStream supersedes, and skipping would dead-end the loop.

    const text = latestAssistantText();
    if (!text) {
      // A run with no answer text (reasoning only, an error) still closes the stream it opened:
      // playback end resumes listening and sends queued utterances.
      if (streamBegun) endStreamRef.current("");
      else resumeListen();
      return;
    }
    const endMsgs = auiRef.current.thread().getState().messages;
    const endTail = endMsgs.length ? endMsgs[endMsgs.length - 1] : null;
    spokenReplyRef.current = endTail?.role === "assistant" ? { id: endTail.id, text } : null;
    if (streamBegun) {
      // Flush the remaining sentences (incl. the trailing one) and finish; the
      // stream's consumer calls resumeListen when playback drains. Read again
      // after a beat: the last token ("?") landed after isRunning flipped once.
      setTimeout(() => {
        const final = latestAssistantText();
        const said = final.length >= text.length ? final : text;
        if (spokenReplyRef.current) spokenReplyRef.current = { ...spokenReplyRef.current, text: said };
        endStreamRef.current(said);
      }, 150);
    } else {
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

  // No session-end watcher: the adapter raises requestVoiceSubmit itself.

  // End the loop (the plus-menu toggle; ChatPage unloads the voice models on it).
  const turnOff = useCallback(() => {
    _voicedThreadId = null;
    setLoopVoiceMode("off");
    voiceModeRef.current = "off";
    setVoiceModeState("off");
    useChatRuntimeStore.getState().setVoiceMode("off");
    if (voiceCoalesceTimerRef.current) {
      clearTimeout(voiceCoalesceTimerRef.current);
      voiceCoalesceTimerRef.current = null;
    }
    clearHeldTurn();
    clearPendingVoiceNotes();
    earlyInterjectRef.current?.reset();
    clearFiller();
    setPendingFiller(null);
    queuedUtteranceRef.current = [];
    pendingForVoiceRef.current = [];
    clearVoiceDecisions();
    holdPlayback(false);
    stop();
    pauseOutputSpectrum();
    const thread = auiRef.current.thread();
    if (thread.getState().isRunning) {
      try {
        thread.cancelRun();
      } catch {
        // Run may have ended between the check and here.
      }
    }
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
  }, [stop, clearFiller, holdPlayback]);

  const toggle = useCallback(() => {
    // OFF → CONFIGURING (show dropdown, don't start mic)
    // CONFIGURING → OFF (user cancelled before picking)
    // ACTIVE → OFF (turn off the loop, models unloaded)
    if (voiceModeRef.current !== "off") {
      turnOff();
      return;
    }
    setLoopVoiceMode("configuring");
    voiceModeRef.current = "configuring";
    setVoiceModeState("configuring");
    useChatRuntimeStore.getState().setVoiceMode("configuring");
    // The pick is left alone. Clearing it here used to be safe because null
    // meant "browser voice" -- a working fallback. With the browser voice gone
    // null means silence, so voice would open mute every time. ChatPage seeds a
    // default when nothing is remembered; the slot still loads on activate, not
    // here, so opening the picker costs nothing.
    primeAudio();
    // "configuring": dropdown appears via store; mic stays off.
  }, [turnOff, primeAudio]);


  // Switching threads carries the loop over instead of ending it: cut the old
  // thread's reply, run and mic, drop its pending timers, then re-arm on the new one.
  // Voice mode, the picked voice, the loaded slot and a pause all survive, the same
  // as the chat model does; a paused loop stays paused with nothing stale to resume.
  const threadReset = useCallback((prevThreadId: string | null) => {
    if (voiceModeRef.current === "off") return;
    // The aui handle already points at the new thread, so stop the old run by id.
    const oldRun = voiceRunToStop(
      prevThreadId,
      _voicedThreadId,
      useChatRuntimeStore.getState().runningByThreadId,
    );
    voiceTrace("engine.threadReset", { prevThreadId, stopped: oldRun, paused: useChatRuntimeStore.getState().voicePaused });
    if (oldRun) stopChatThread(oldRun);
    _voicedThreadId = null;
    settleInterject(null);
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
    clearPendingVoiceNotes();
    earlyInterjectRef.current?.reset();
    clearFiller();
    setPendingFiller(null);
    queuedUtteranceRef.current = [];
    pendingForVoiceRef.current = [];
    stop();
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
    if (voiceModeRef.current === "active") resumeListen();
  }, [stop, resumeListen, clearFiller, settleInterject]);

  useEffect(() => {
    registerVoiceThreadReset(threadReset);
    return () => {
      registerVoiceThreadReset(null);
    };
  }, [threadReset]);

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

  useEffect(() => {
    registerVoiceToggle(toggle);
    return () => {
      registerVoiceToggle(null);
    };
  }, [toggle]);

  // The composer's stop square: everything in flight ends (reply audio, the run,
  // held and queued utterances, a pending coalesced send) and the loop goes back
  // to listening. Voice mode and its models stay as they are.
  useEffect(() => {
    registerVoiceStopSpeaking(() => {
      if (voiceModeRef.current !== "active") return;
      voiceTrace("engine.stopSpeaking", { speaking: isSpeakingRef.current, playing: isPlayingRef.current });
      // A hard stop ends a pause too: the loop is listening again afterwards.
      if (useChatRuntimeStore.getState().voicePaused) {
        useChatRuntimeStore.getState().setVoicePaused(false);
        holdPlayback(false);
      }
      if (voiceCoalesceTimerRef.current) {
        clearTimeout(voiceCoalesceTimerRef.current);
        voiceCoalesceTimerRef.current = null;
      }
      clearHeldTurn();
      clearPendingVoiceNotes();
      earlyInterjectRef.current?.reset();
      clearFiller();
      setPendingFiller(null);
      queuedUtteranceRef.current = [];
      pendingForVoiceRef.current = [];
      stop();
      const thread = auiRef.current.thread();
      if (thread.getState().isRunning) {
        try {
          thread.cancelRun();
        } catch {
          // Run may have ended between the check and here.
        }
      }
      // The open barge-in session holds the epoch the clear just retired: its next turn would be dropped.
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
      resumeListen();
    });
    return () => registerVoiceStopSpeaking(null);
  }, [stop, resumeListen, clearFiller, holdPlayback]);

  // The composer's tap: pause (mic muted through the mute path, playback frozen, the
  // run left to finish into the thread) or resume from exactly where it was held.
  useEffect(() => {
    registerVoicePauseToggle(() => {
      if (voiceModeRef.current !== "active") return;
      const store = useChatRuntimeStore.getState();
      const next = !store.voicePaused;
      voiceTrace("engine.pause", { paused: next, speaking: isSpeakingRef.current, playing: isPlayingRef.current });
      store.setVoicePaused(next);
      holdPlayback(next);
    });
    return () => registerVoicePauseToggle(null);
  }, [holdPlayback]);

  // Expose resumeListen so the dictation adapter can re-arm the mic after a
  // recoverable no-speech timeout. It self-guards: no-op unless voice is active.
  // A no-speech end with turns parked for a voice that has since loaded sends
  // them instead of just re-arming.
  const resumeOrFlush = useCallback(() => {
    if (
      pendingForVoiceRef.current.length &&
      !useChatRuntimeStore.getState().voiceSlotLoading
    ) {
      submitRef.current("");
      return;
    }
    resumeListen();
  }, [resumeListen]);
  useEffect(() => {
    registerVoiceResume(resumeOrFlush);
    return () => {
      registerVoiceResume(null);
    };
  }, [resumeOrFlush]);

  useEffect(() => {
    registerVoiceBargeIn(() => {
      voiceTrace("engine.bargeIn", { speaking: isSpeakingRef.current, playing: isPlayingRef.current });
      if (voiceModeRef.current !== "active") return;
      if (!isSpeakingRef.current && !isPlayingRef.current) return;
      pausePlayback();
    });
    registerVoiceBargeAbort(() => {
      voiceTrace("engine.bargeAbort", { speaking: isSpeakingRef.current, playing: isPlayingRef.current });
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

  // Enter or Send with typed text in the box: send the box as the turn, paused,
  // listening, holding a fragment or speaking. The box already shows the parked
  // turns and the held fragment, so it replaces them, and the session is cancelled
  // rather than finished so nothing it captured goes out a second time.
  useEffect(() => {
    registerVoiceTypedSubmit((box) => {
      if (voiceModeRef.current !== "active") return false;
      const shown = [pendingForVoiceRef.current.join(" "), getHeldTurnText()]
        .filter(Boolean)
        .join(" ");
      const paused = useChatRuntimeStore.getState().voicePaused;
      const revealing = voiceRevealText();
      if (isRevealPrefix(box, revealing)) {
        // The loop sends the whole transcript itself once the reveal stops typing.
        voiceTrace("engine.revealRush", { box });
        rushVoiceReveal();
        return true;
      }
      if (!isTypedVoiceSubmit({ box, shown, paused, revealing })) return false;
      voiceTrace("engine.typedSubmit", { text: box, paused });
      clearHeldTurn();
      clearPendingVoiceNotes();
      pendingForVoiceRef.current = [];
      cancelVoiceSession();
      submitTranscript(box, true);
      return true;
    });
    return () => registerVoiceTypedSubmit(null);
  }, [submitTranscript]);

  // Thread-switch voice reset moved OUT of VoiceEngine: this component remounts
  // across the ThreadWelcome → ThreadComposerDock (first-send) boundary and loses
  // its prev-thread subscription, so it never sees null → __LOCALID_xxx. The reset
  // now lives in SingleContent (chat-page.tsx), which is stable across that remount,
  // and drives requestVoiceToggle() via the module-level bridge.

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
  const paused = useChatRuntimeStore((s) => s.voicePaused);
  const setVoiceMuted = useChatRuntimeStore((s) => s.setVoiceMuted);
  const t = useT();
  useEffect(() => {
    setMicMuted(muted || paused);
  }, [muted, paused]);
  useEffect(() => {
    if (voiceMode === "off") return;
    const onKey = (event: KeyboardEvent) => {
      if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      if (event.key.toLowerCase() !== "m") return;
      event.preventDefault();
      setVoiceMuted(!useChatRuntimeStore.getState().voiceMuted);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [voiceMode, setVoiceMuted]);
  // Shown from the moment voice mode is armed, so the mic can be muted while the models load.
  if (voiceMode === "off") return null;
  const label = muted ? t("voiceControls.unmute") : t("voiceControls.mute");
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

// Desktop only: holding the mini orb swaps the window for the floating orb
// (orb-widget-host.ts). It opens as the hold arms, and the click that follows the
// release is swallowed so a hold never also opens the full orb.
function useOrbWidgetHold() {
  const firedRef = useRef(false);
  const holdRef = useRef<HoldToStop | null>(null);
  // Screen coordinates are already logical (CSS) pixels, so no scale factor applies.
  const cursorRef = useRef<{ x: number; y: number } | null>(null);
  if (isTauri && !holdRef.current) {
    holdRef.current = createHoldToStop(
      (phase) => {
        if (phase !== "armed") return;
        firedRef.current = true;
        // The window hides under the held pointer, so its pointerup may never come.
        queueMicrotask(() => holdRef.current?.cancel());
        const store = useChatRuntimeStore.getState();
        if (store.voiceMode !== "active") store.setVoiceMode("active");
        const cursor = cursorRef.current;
        void import("./orb-widget-host").then((m) => m.openOrbWidget(cursor));
      },
      undefined,
      ORB_WIDGET_LONG_PRESS_MS,
      ORB_WIDGET_LONG_PRESS_MS,
    );
  }
  useEffect(() => () => holdRef.current?.dispose(), []);
  const cancel = () => holdRef.current?.cancel();
  return {
    handlers: {
      onPointerDown: (e: React.PointerEvent) => {
        if (e.button !== 0) return;
        firedRef.current = false;
        cursorRef.current = { x: e.screenX, y: e.screenY };
        holdRef.current?.press();
      },
      onPointerMove: (e: React.PointerEvent) => {
        cursorRef.current = { x: e.screenX, y: e.screenY };
      },
      onPointerUp: () => {
        holdRef.current?.release();
      },
      onPointerLeave: cancel,
      onPointerCancel: cancel,
    },
    /** True once for the click that ends a hold that opened the widget. */
    consumeClick: () => {
      if (!firedRef.current) return false;
      firedRef.current = false;
      return true;
    },
  };
}

export const VoiceControlButton: FC = () => {
  const hold = useOrbWidgetHold();
  const voiceMode = useChatRuntimeStore((s) => s.voiceMode);
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceMode = useChatRuntimeStore((s) => s.setVoiceMode);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);
  const t = useT();

  if (voiceMode === "off") return null;

  if (voiceMode === "active" && !collapsed) {
    return (
      <TooltipIconButton
        tooltip={t("voiceControls.backToChat")}
        aria-label={t("voiceControls.backToChat")}
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
  const tooltip = active ? t("voiceControls.openOrb") : t("voiceControls.start");

  return (
    <TooltipIconButton
      tooltip={isTauri ? t("voiceControls.holdToFloat", { action: tooltip }) : tooltip}
      aria-label={tooltip}
      variant="ghost"
      className="size-8 rounded-full"
      {...hold.handlers}
      onClick={() => {
        if (hold.consumeClick()) return;
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
