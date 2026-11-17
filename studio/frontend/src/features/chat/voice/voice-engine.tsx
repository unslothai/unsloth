// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { resetPromptQueuesForThread } from "@/components/assistant-ui/thread";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { VoicePanel as VoiceSetupPanel } from "@/components/assistant-ui/voice-panel";
import {
  VoiceOrb
} from "@/components/assistant-ui/voice-orb";
import {
  StudioVoiceLoopAdapter,
  cancelVoiceSession,
  claimHeldTurnForEdit,
  clearHeldTurn,
  dropVoiceUtteranceAudio,
  forgetVoiceBox,
  endHeldEdit,
  getHeldTurnText,
  isComposerHeldForEdit,
  isUtteranceInFlight,
  registerComposerTextReader,
  rushVoiceReveal,
  setMicMuted,
  parkVoiceTurnAudio,
  takeVoiceTurnAudio,
  voiceRevealText,
  takeTypedDraft,
  voiceWroteBox,
  voiceTextReleaser,
  voiceUtterancePending,
  withoutVoiceText,
} from "@/features/chat/adapters/studio-voice-loop-adapter";
import { getActiveChatGenerationRuns } from "@/features/chat/api/chat-generation-api";
import { splitStreaming, useTtsPlayer } from "@/features/chat/hooks/use-tts-player";
import { codeToolsOn, useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { stopChatThread
} from "@/features/chat/utils/stop-chat-thread";
import {
  REPEAT_WINDOW_MS,
  interjectOutcomeFor,
  interjectUnderApproval,
  isHoldOn,
  replyEchoesFiller,
  isSameAsk,
} from "@/features/chat/voice/laya-outcomes";
import { ORB_IDLE_GRADIENT, orbConfig } from "@/features/chat/voice/orb-config";
import { useChatModelStatus } from "@/features/chat/voice/chat-model-ready";
import { deriveOrbState } from "@/features/chat/voice/orb-state";
import { stillInTurn } from "@/features/chat/voice/turn-end";
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
  clearPendingHeardCut,
  pendingHeardCut,
  setPendingHeardCut,
  registerVoiceEarlyInterject,
  registerVoiceListen,
  registerVoiceParkedText,
  registerVoicePauseToggle,
  registerVoicePlayback,
  registerVoiceReplyText,
  registerVoiceResume,
  registerVoiceStopSpeaking,
  registerVoiceSpokenAudio,
  registerVoiceSubmit,
  registerVoiceThreadReset,
  registerVoiceToggle,
  registerVoiceTypedSubmit,
  registerVoiceTypedSend,
  setVoiceMode as setLoopVoiceMode,
  type VoiceSubmitMeta,
  type VoiceTurnEnd,
  dropUnsentVoiceTurn,
  dropVoiceTurn,
  endVoiceTurns,
  liveVoiceTurn,
  noteVoiceTurn,
  sendVoiceTurn,
  voiceRunToStop,
} from "@/features/chat/voice/voice-loop-bridge";
import {
  type InterjectAction,
  fillerFromModel,
  interjectAction,
  reportLayaOutcome,
} from "@/features/chat/voice/voice-models-client";
import { chatModelSpeaks, isChatModelVoice, voiceHearsAudio } from "@/features/chat/voice/voice-models";
import { voiceServing } from "@/features/chat/voice/voice-slot";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { pauseOutputSpectrum } from "@/features/chat/voice/voice-spectrum";
import { voiceTrace } from "@/features/chat/voice/voice-trace";
import { ChatMessageProtectedError } from "../api/chat-api";
import { setThreadMessageVoiceHeard } from "../utils/update-thread-message";
import { type PlayedChunk, heardCut, joinTextParts, laterCut, savedHeardCut } from "./heard-cut";
import { splitSpoken } from "./speech-text";
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
import {
  coalescedTurnAudio,
  lastUserTurnHasAudio,
  voiceTurnAttachment,
  withoutAudioPlayer,
} from "./turn-audio";
import { isRevealPrefix, isTypedVoiceSubmit } from "./typed-send";

let _prevRunning = false;
// The first reply token seen for this turn id, so it is marked once.
let _firstTokenTurn: string | null = null;

function endTurns(reason: VoiceTurnEnd, liveOnly = false): void {
  for (const turnId of endVoiceTurns(liveOnly)) voiceTrace("turn.end", { turnId, reason });
}

/** No run will answer the waiting utterance: its trace ends, and the next utterance starts a turn of its own. */
function dropUnsentTurn(): void {
  const turnId = dropUnsentVoiceTurn();
  if (turnId) voiceTrace("turn.end", { turnId, reason: "cancelled" });
}

/** A run is about to answer: the waiting spoken turn becomes the live one. */
function markChatSent(): void {
  const before = liveVoiceTurn();
  const turnId = sendVoiceTurn();
  if (!turnId) return;
  // A turn still live here never got its first audio: closed, so every traced turn ends once.
  if (before && before !== turnId) voiceTrace("turn.end", { turnId: before, reason: "superseded" });
  voiceTrace("turn.chatSent", { turnId });
}
// Module-level: New Chat remounts VoiceEngine, and the reset after it must still find the run to stop.
let _voicedThreadId: string | null = null;
// Replies with a heard-cut save loop running.
const _heardSaving = new Set<string>();

// Filtered as the floating orb does: a Compare pane or background chat's card is not voice mode's to speak.
function voicedApprovalCount(s: ReturnType<typeof useChatRuntimeStore.getState>): number {
  const pending = pendingApproval(s.toolConfirmations, s.activeThreadId, (id) => describeToolCall(id) !== null);
  return pending ? pending.more + 1 : 0;
}

function speaksWithChatModel(): boolean {
  const s = useChatRuntimeStore.getState();
  return isChatModelVoice(s.selectedVoiceModelId) && chatModelSpeaks(s);
}

const IGNORE_MAX_WORDS = 3;
const PLAYBACK_TAIL_MS = 500;
const VOICE_APPEND_WAIT_MAX_MS = 120_000;
const VOICE_COALESCE_MS = 250;
const VOICE_COALESCE_POLL_MS = 250;
// After a barge-in cut an unanswered run the user tends to keep going; each restart threw the thinking away.
const VOICE_RESTART_COALESCE_MS = 1800;
const VOICE_HOLD_COALESCE_MS = 6000;
// The window never closes on someone mid-sentence; past this it gives up waiting.
const VOICE_HEARING_HOLD_MAX_MS = 30000;
// cancelRun() is async, so the old run can still read as running when the window closes: re-check for ~10 s.
const VOICE_COALESCE_MAX_RETRIES = 40;
const VOICE_RUN_SETTLE_MAX_MS = 6000;
const VOICE_RUN_SETTLE_POLL_MS = 150;
// Clip swaps at sentence boundaries stay under this, so only a real synth stall turns the orb lilac.
const SYNTH_GAP_MS = 350;
// How long a heard cut keeps trying to reach the server: its thread back on screen, or a failed save.
const HEARD_SAVE_PATIENCE_MS = 30 * 60_000;

export { VoiceOrb };
export { VoicePanel } from "@/components/assistant-ui/voice-panel";
export { loadVoice, servedVoice, unloadVoice, voiceServing } from "@/features/chat/voice/voice-slot";

registerVoiceListen(() =>
  StudioVoiceLoopAdapter.isSupported() ? new StudioVoiceLoopAdapter().listen() : null,
);

type HeardSnapshot = { messageId: string; position: number; log: PlayedChunk[]; base: number };

export const VoiceEngine: FC = () => {
  const aui = useAui();
  const [voiceMode, setVoiceModeState] = useState(getVoiceMode());
  const voiceCoalesceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // A chat's first message, held for the coalesce window before it goes through the composer.
  const firstTurnHeldRef = useRef<string | null>(null);
  // Bumped per armed coalesced run: a start still settling after a newer utterance must not fire.
  const coalesceSeqRef = useRef(0);
  const voiceModeRef = useRef(getVoiceMode());
  const isSpeakingRef = useRef(false);
  const isPlayingRef = useRef(false);
  const playedAtRef = useRef(-Infinity);
  const synthGapTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const orbDwellRef = useRef<OrbDwell | null>(null);
  const auiRef = useRef(aui);
  auiRef.current = aui;

  // The store, not useAuiState: it survives the composer remount on first send.
  const isThreadRunning = useChatRuntimeStore((s) =>
    s.activeThreadId ? Boolean(s.runningByThreadId[s.activeThreadId]) : false,
  );
  const chatModelStatus = useChatModelStatus();
  const hasChatModel = chatModelStatus === "ready";
  const sttDownload = useChatRuntimeStore((s) => s.sttSlotDownload);
  // The chat model the loop last had to talk to: a swap's "changed" is judged against it.
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint ?? null);
  const readyCheckpointRef = useRef(checkpoint);
  if (hasChatModel) readyCheckpointRef.current = checkpoint;
  const hasChatModelRef = useRef(hasChatModel);
  hasChatModelRef.current = hasChatModel;
  const rearmSeqRef = useRef(0);
  const resumeListen = useCallback(() => {
    // Listening never waits for a loading voice (a shut mic lost speech); only the send does.
    // Dictation can lag clearing after a session ends; clicking Dictate while it is set would toggle it OFF.
    const MAX_ATTEMPTS = 5;
    const RETRY_MS = 50;

    const clickDictate = () => {
      const fresh = auiRef.current.composer();
      // Already listening: startDictation() on a live session tears the mic down.
      if (fresh.getState().dictation) return;
      // assistant-ui appends transcripts onto the box text and the adapter emits the whole turn, so leftovers double up.
      // A typed draft ahead of a held fragment stays: only the fragment goes.
      const typed = isComposerHeldForEdit() ? null : withoutVoiceText(fresh.getState().text);
      if (typed !== null) fresh.setText(typed);
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

  const queuedUtteranceRef = useRef<string[]>([]);
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
    feedClip,
    endClips,
    sayAside,
    warmAsides,
    stop,
    primeAudio,
    pausePlayback,
    resumePlayback,
    holdPlayback,
    playbackPosition,
    playedLog,
  } = useTtsPlayer(onPlaybackEnd);
  const pausedForHold = useChatRuntimeStore((s) => s.voicePaused);
  isSpeakingRef.current = isSpeaking;
  if (isPlayingRef.current && !isPlaying) playedAtRef.current = performance.now();
  isPlayingRef.current = isPlaying;
  useEffect(() => {
    // Audible output only (plus its echo tail and sentence gaps): while the reply is still being thought
    // or synthesized nothing plays, so a normal-volume interjection is heard, not held to the barge-in bar.
    registerVoicePlayback(
      () => isPlayingRef.current || performance.now() - playedAtRef.current < PLAYBACK_TAIL_MS,
    );
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
  const feedClipRef = useRef(feedClip);
  feedClipRef.current = feedClip;
  const endClipsRef = useRef(endClips);
  endClipsRef.current = endClips;
  const playbackPositionRef = useRef(playbackPosition);
  playbackPositionRef.current = playbackPosition;
  const playedLogRef = useRef(playedLog);
  playedLogRef.current = playedLog;
  const heardSnapshotRef = useRef<HeardSnapshot | null>(null);
  const heardCutRecorderRef = useRef<() => void>(() => {});
  const spokenRunRef = useRef(false);
  const spokenClipsRef = useRef(0);
  const streamBegunRef = useRef(false);
  const streamGenRef = useRef(0);
  const [streamSeq, setStreamSeq] = useState(0);
  const streamPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastFallbackAtRef = useRef(0);
  // The filler model answered (say or quiet). A quiet vote is the answer, so the
  // quiet fallback only bridges a turn the model never answered at all.
  const fillerAnsweredRef = useRef(false);
  // So a thread blinking back to running with no new message cannot cut the reply and speak it again.
  const spokenReplyRef = useRef<{ id: string; text: string } | null>(null);
  const runFlickerRef = useRef(false);
  const lastAsideAtRef = useRef(0);
  const sayAsideRef = useRef(sayAside);
  sayAsideRef.current = (...args: Parameters<typeof sayAside>) => {
    // Speaking with the chat model, an aside would queue behind the reply on that same model.
    if (speaksWithChatModel()) return;
    lastAsideAtRef.current = Date.now();
    return sayAside(...args);
  };
  // Whether anything was said aloud for the current turn: the first aside must not be "still on it".
  const spokenThisTurnRef = useRef(false);
  const warmAsidesIfTts = useCallback(
    async (texts: readonly string[]) => {
      if (!speaksWithChatModel()) await warmAsides(texts);
    },
    [warmAsides],
  );
  warmAsidesRef.current = warmAsidesIfTts;
  // Reply text is held back until the filler is queued, so the filler is always heard first.
  const fillerRef = useRef<{
    phrase: string;
    timer: ReturnType<typeof setTimeout> | null;
    queued: boolean;
    spokenAt: number | null;
    stage2Done: boolean;
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
  // No answer text yet: prefill, or thinking with reasoning off. Speech then is judged like speech during reasoning.
  const replyNotStarted = useCallback((): boolean => {
    const messages = auiRef.current.thread().getState().messages;
    const last = messages[messages.length - 1];
    if (!last || last.role === "user") return true;
    if (last.role !== "assistant") return false;
    return !(last.content as Array<{ type: string; text?: string }>).some((p) => p.type === "text" && p.text);
  }, []);
  const [isReasoning, setIsReasoning] = useState(false);

  const latestAssistantText = useCallback((): string => {
    const messages = auiRef.current.thread().getState().messages;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role !== "assistant") continue;
      // "\n" between parts, so "check" + tool + "The…" is not read as "checkThe…"; trimmed at the
      // end only, so offsets into the reply stay valid.
      const { text } = joinTextParts(msg.content as Array<{ type: string; text?: string }>);
      return withoutAudioPlayer(text).trimEnd();
    }
    return "";
  }, []);

  const commitUtterance = useCallback((said: string, confirmedEarly = false) => {
    const composer = auiRef.current.composer();
    // A barge-in cuts playback without its end firing, so held utterances go now or surface later as a stale prompt.
    // A first message still inside its window is the start of this one. A cleared timer means voice
    // mode stopped since, so that text is stale.
    const first = voiceCoalesceTimerRef.current !== null ? firstTurnHeldRef.current : null;
    firstTurnHeldRef.current = null;
    const held = [
      ...(first !== null && first !== said ? [first] : []),
      ...queuedUtteranceRef.current.filter((q) => q !== said),
    ];
    queuedUtteranceRef.current = [];
    const text = held.length ? [...held, said].join(" ") : said;
    if (held.length) voiceTrace("engine.foldHeld", { held, text });
    // Drained on every send, so speech kept while the model could not answer in it never rides along later.
    const taken = takeVoiceTurnAudio();
    // Non-GGUF backends refuse audio beside an image or video, so such a turn goes as its transcript.
    const visual = composer
      .getState()
      .attachments.some((a) => a.type === "image" || /^(image|video)\//.test(a.contentType ?? ""));
    // A spoken reply (and MLX) takes one clip: beside an attached audio file the question goes as its transcript.
    const attachedAudio = composer
      .getState()
      .attachments.some((a) => a.type === "audio" || /^audio\//.test(a.contentType ?? ""));
    const rt = useChatRuntimeStore.getState();
    // The audio-input route skips Studio's tool loop, so with tools on the turn goes as its transcript.
    const toolsOn =
      rt.supportsTools &&
      (rt.toolsEnabled || codeToolsOn(rt) || rt.mcpEnabledForChat || rt.ragEnabled || rt.deepResearchEnabled);
    const turnAudio =
      !visual &&
      !toolsOn &&
      !(attachedAudio && (speaksWithChatModel() || rt.loadedIsMlx === true)) &&
      voiceHearsAudio(rt, useVoiceSettingsStore.getState().micTarget)
        ? taken
        : null;
    // Cover the tail clip (isPlaying) too, not just an active streaming session (isSpeaking).
    voiceTrace("engine.commitUtterance", { text, speaking: isSpeakingRef.current, playing: isPlayingRef.current });
    const thread = auiRef.current.thread();
    const cutRun = thread.getState().isRunning;
    // A barge-in takes effect here, so an ignored or queued one cuts nothing: the cut is in memory before the
    // request is built. A send that cuts nothing drops a snapshot an earlier barge-in left.
    if (cutRun || isSpeakingRef.current || isPlayingRef.current) heardCutRecorderRef.current();
    else heardSnapshotRef.current = null;
    // The reply this cuts never got its first audio out.
    if (cutRun || isSpeakingRef.current || isPlayingRef.current) endTurns("barged", true);
    if (isSpeakingRef.current || isPlayingRef.current) stop();
    if (cutRun) {
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
    const releaseVoiceText = voiceTextReleaser();
    const deferredClear = (n: number) => {
      if (voiceModeRef.current !== "active") return;
      const fresh = auiRef.current.composer();
      // Only the transcript the composer refills: a draft typed since, or another chat's, stays.
      if (voiceWroteBox(fresh.getState().text)) {
        fresh.setText("");
        if (n < 5) {
          setTimeout(() => deferredClear(n + 1), DEFERRED_CLEAR_MS);
          return;
        }
      }
      releaseVoiceText();
    };
    const sendNow = () => {
      composer.setText(text);
      markChatSent();
      composer.send();
      composer.setText("");
      setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
    };
    const sendThroughComposer = () => {
      if (!turnAudio) {
        sendNow();
        return;
      }
      const bytes = Uint8Array.from(atob(turnAudio), (c) => c.charCodeAt(0));
      // A chat switch while the clip attaches resets the turn: the old words never go to the new chat.
      const turn = voiceNotesEpoch();
      const fromThread = useChatRuntimeStore.getState().activeThreadId;
      void composer
        .addAttachment(new File([bytes], "voice.wav", { type: "audio/wav" }))
        .catch((error: unknown) => voiceTrace("engine.turnAudioAttachFailed", { message: String(error) }))
        .then(() => {
          if (voiceModeRef.current !== "active" || voiceNotesEpoch() !== turn) return;
          if (useChatRuntimeStore.getState().activeThreadId !== fromThread) return;
          sendNow();
        });
    };

    // First message: the append + startRun path goes stale across the first-send remount.
    if (thread.getState().messages.length === 0) {
      // send() takes the box, not `text`: parked turns were never in it (the re-arm
      // clears it), so an empty box here sent nothing at all.
      // It waits out the window a later turn gets, with the mic open: sent at once, the reply began
      // over a sentence that went on after an early end, and the rest of it was lost.
      resumeListen();
      const armedAt = Date.now();
      let heardAt = 0;
      const sendFirst = () => {
        voiceCoalesceTimerRef.current = null;
        if (voiceModeRef.current !== "active") return;
        const rt = useChatRuntimeStore.getState();
        const hearing = rt.voiceHearing || rt.voiceTranscribing;
        if (hearing) heardAt = Date.now();
        if (stillInTurn(hearing, heardAt, Date.now()) && Date.now() - armedAt < VOICE_HEARING_HOLD_MAX_MS) {
          voiceTrace("engine.coalesceWaitHearing", { first: true });
          voiceCoalesceTimerRef.current = setTimeout(sendFirst, VOICE_COALESCE_POLL_MS);
          return;
        }
        firstTurnHeldRef.current = null;
        sendThroughComposer();
      };
      firstTurnHeldRef.current = text;
      if (voiceCoalesceTimerRef.current) clearTimeout(voiceCoalesceTimerRef.current);
      voiceCoalesceTimerRef.current = setTimeout(sendFirst, VOICE_COALESCE_MS);
      return;
    }

    const before = thread.getState().messages.length;
    let appended = false;
    if (composer.getState().attachments.length > 0) {
      // The composer's send carries its attachments; it lands asynchronously, so the window waits for it.
      const send = () => {
        composer.setText(text);
        composer.send({ startRun: false });
      };
      if (turnAudio) {
        const bytes = Uint8Array.from(atob(turnAudio), (c) => c.charCodeAt(0));
        const turn = voiceNotesEpoch();
        void composer
          .addAttachment(new File([bytes], "voice.wav", { type: "audio/wav" }))
          .catch((error: unknown) => voiceTrace("engine.turnAudioAttachFailed", { message: String(error) }))
          .then(() => {
            if (voiceModeRef.current !== "active" || voiceNotesEpoch() !== turn) return;
            send();
            // The window may have given up waiting for this message: arm it again.
            if (!voiceCoalesceTimerRef.current)
              voiceCoalesceTimerRef.current = setTimeout(() => fireCoalescedRun(0), VOICE_COALESCE_MS);
          });
      } else {
        send();
      }
      appended = true;
    } else {
      try {
        auiRef.current.thread().append({
          role: "user",
          content: [{ type: "text", text }],
          ...(turnAudio
            ? {
                attachments: [
                  voiceTurnAttachment(
                    coalescedTurnAudio(auiRef.current.thread().getState().messages, turnAudio),
                    `voice-${Date.now()}`,
                  ),
                ],
              }
            : {}),
          createdAt: new Date(),
          startRun: false,
        } as never);
        appended = true;
      } catch {
      }
    }

    if (!appended) {
      sendThroughComposer();
      return;
    }

    composer.setText("");
    setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
    resumeListen();
    const armedAt = Date.now();
    let heardAt = 0;
    const fireCoalescedRun = (attempt: number) => {
      voiceCoalesceTimerRef.current = null;
      if (voiceModeRef.current !== "active") return;
      const rt = useChatRuntimeStore.getState();
      const hearing = rt.voiceHearing || rt.voiceTranscribing || voiceUtterancePending();
      if (hearing) heardAt = Date.now();
      if (stillInTurn(hearing, heardAt, Date.now()) && Date.now() - armedAt < VOICE_HEARING_HOLD_MAX_MS) {
        voiceTrace("engine.coalesceWaitHearing", {});
        voiceCoalesceTimerRef.current = setTimeout(() => fireCoalescedRun(attempt), VOICE_COALESCE_POLL_MS);
        return;
      }
      const t = auiRef.current.thread();
      if (t.getState().isRunning) {
        // cancelRun() has not settled yet: retry rather than leave the bubbles unanswered.
        if (attempt < VOICE_COALESCE_MAX_RETRIES) {
          voiceCoalesceTimerRef.current = setTimeout(
            () => fireCoalescedRun(attempt + 1),
            VOICE_COALESCE_POLL_MS,
          );
        } else {
          dropUnsentTurn();
        }
        return;
      }
      const msgs = t.getState().messages;
      // An attachment's send uploads its original first: wait for the message on the clock, not a tick budget.
      if (msgs.length <= before && Date.now() - armedAt < VOICE_APPEND_WAIT_MAX_MS) {
        voiceCoalesceTimerRef.current = setTimeout(() => fireCoalescedRun(attempt + 1), VOICE_COALESCE_POLL_MS);
        return;
      }
      const last = msgs.length > 0 ? msgs[msgs.length - 1] : null;
      if (!last || last.role !== "user") {
        dropUnsentTurn();
        return;
      }
      // A new run while the old one is still "cancelling" is refused outright: ask the server, not the local state.
      const threadId = useChatRuntimeStore.getState().activeThreadId;
      const turn = voiceNotesEpoch();
      const start = () => {
        if (voiceModeRef.current !== "active") return;
        if (voiceNotesEpoch() !== turn || coalesceSeqRef.current !== seq) return;
        if (useChatRuntimeStore.getState().activeThreadId !== threadId) return;
        const now = t.getState();
        // Another run took the thread, or the tail moved: this utterance is not answered here.
        if (now.isRunning || now.messages[now.messages.length - 1]?.id !== last.id) {
          dropUnsentTurn();
          return;
        }
        try {
          markChatSent();
          t.startRun({ parentId: last.id });
        } catch {
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
    const seq = ++coalesceSeqRef.current;
    const holdOn = isHoldOn(said);
    if (holdOn) voiceTrace("engine.holdOn", { text: said });
    voiceCoalesceTimerRef.current = setTimeout(
      () => fireCoalescedRun(0),
      // An early-confirmed turn already answered "did they keep talking?"; restart and hold keep their windows.
      holdOn
        ? VOICE_HOLD_COALESCE_MS
        : cutRun
          ? VOICE_RESTART_COALESCE_MS
          : confirmedEarly
            ? 0
            : VOICE_COALESCE_MS,
    );
  }, [resumeListen, stop]);

  useEffect(() => {
    registerVoiceSpokenAudio((threadId, wav) => {
      if (voiceModeRef.current !== "active" || !spokenRunRef.current) return;
      const active = useChatRuntimeStore.getState().activeThreadId;
      if (threadId && active && threadId !== active) return;
      spokenClipsRef.current += 1;
      feedClipRef.current(wav);
    });
    return () => registerVoiceSpokenAudio(null);
  }, []);

  useEffect(() => {
    registerVoiceReplyText(() => latestAssistantText());
    registerVoiceParkedText(() => pendingForVoiceRef.current.join(" "));
    return () => {
      registerVoiceReplyText(null);
      registerVoiceParkedText(null);
    };
  }, [latestAssistantText]);

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
    return parts.join("\n");
  }, []);

  const sendTranscriptRef = useRef<(text: string, typed?: boolean, meta?: VoiceSubmitMeta) => void>(() => {});
  const sendTranscript = useCallback(
    (text: string, typed?: boolean, meta?: VoiceSubmitMeta) => sendTranscriptRef.current(text, typed, meta),
    [],
  );
  const submitTranscript = useCallback((spoken?: string, typed?: boolean, meta?: VoiceSubmitMeta) => {
    if (voiceModeRef.current !== "active") return;
    const composer = auiRef.current.composer();
    // A typed send is the whole box already, draft included.
    const draft = spoken && !typed ? takeTypedDraft(composer.getState().text) : "";
    if (composer.getState().dictation) composer.stopDictation();
    let text = (spoken ?? composer.getState().text).trim();
    if (draft) text = `${draft} ${text}`;
    // The chat model went away while this was heard: keep it as a draft, never auto-load a model to send it.
    if (!hasChatModelRef.current) {
      if (text && !typed) composer.setText(text);
      return;
    }
    voiceTrace("engine.submit", { text });
    if (typed) endTurns("typed");
    else if (text) noteVoiceTurn(meta?.turnId);
    const pending = pendingForVoiceRef.current;
    // A reply sent now would land before the voice can speak it: park the turn until the slot loads.
    if (useChatRuntimeStore.getState().voiceSlotLoading) {
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
    // Nothing said is not a barge-in: cutting a playing reply here truncated it after the first word.
    if (!text) {
      resumeListen();
      return;
    }
    // The store can think the voice is loaded after the backend dropped it (idle unload): one status read settles it.
    const store = useChatRuntimeStore.getState();
    const ensure = store.ensureVoiceSlotLoaded;
    const sendNow = () => sendTranscript(text, typed, meta);
    if (!ensure || isSpeakingRef.current) {
      sendNow();
      return;
    }
    const sent = text;
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

  function sendTranscriptImpl(text: string, typed?: boolean, meta?: VoiceSubmitMeta) {
    const composer = auiRef.current.composer();
    // Decided now so the run's request can tell the model; a reply ending in a question gets no filler.
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
    settleInterject(text);
    const early = earlyInterjectRef.current?.complete();
    setPendingFiller(null);
    clearFiller();
    fillerAnsweredRef.current = false;
    const notesEpoch = voiceNotesEpoch();
    // No thinking filler when the chat model speaks: it would have to wait for the reply itself.
    const fillerVerdict = speaksWithChatModel()
      ? Promise.resolve(null)
      : fillerFromModel(text, prevAskedQuestion);
    void fillerVerdict.then((fromModel) => {
      // Voice off, a thread switch or a typed send since: the verdict is for a turn that's gone.
      if (notesEpoch !== voiceNotesEpoch()) return;
      // A decision landing after the run started is too late for the request's note: the quiet fallback covers the wait.
      if (!fromModel || auiRef.current.thread().getState().isRunning) return;
      fillerAnsweredRef.current = true;
      applyFiller(fromModel);
    });
    const thread = auiRef.current.thread();
    if (early?.restarted) {
      voiceTrace("engine.earlyRestart", { text, kind: early.kind });
      setPendingReplace(early.kind === "replace");
      commitUtterance(text);
      return;
    }
    // Speech during thinking is not always a barge-in: the interject model decides. Typed text is never held.
    if (!typed && thread.getState().isRunning && (latestAssistantIsReasoning() || replyNotStarted())) {
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
        if (voiceNotesEpoch() !== turn) return;
        if (useChatRuntimeStore.getState().activeThreadId !== fromThread) return;
        // The reply ended while the model decided: no playback end is left to send a queued turn.
        const replyGone =
          !auiRef.current.thread().getState().isRunning && !isSpeakingRef.current && !isPlayingRef.current;
        if (action === "ignore" || (action === "queue" && !replyGone)) {
          if (action === "queue" && !queuedUtteranceRef.current.includes(text)) {
            queuedUtteranceRef.current.push(text);
          }
          if (action === "queue") parkVoiceTurnAudio();
          else {
            dropVoiceUtteranceAudio();
            if (dropVoiceTurn(meta?.turnId)) voiceTrace("turn.end", { turnId: meta?.turnId, reason: "cancelled" });
          }
          composer.setText("");
          // The reply plays on, so the barge-in heard nothing to cut.
          heardSnapshotRef.current = null;
          // Nothing cut the reply the barge-in held: without this playback stayed suspended and the loop looked stuck.
          resumePlayback();
          resumeListen();
          return;
        }
        setPendingReplace(answer?.kind === "replace");
        commitUtterance(text);
      });
      return;
    }
    commitUtterance(text, meta?.confirmedEarly);
  }
  sendTranscriptRef.current = sendTranscriptImpl;

  function overrideInterject(text: string, verdict: InterjectAction | null) {
    const repeated = queuedUtteranceRef.current.includes(text);
    // Re-asking the pending question is impatience: restarting threw the thinking away and lengthened the wait.
    const sameAsk = isSameAsk(lastUserPrompt(), text);
    // Something is already held for after this reply, so this is the rest of it
    // ("How many?" then "Too many cannolis"): it joins the hold, never dropped
    // on its own, and the two go as one when the reply ends.
    const continuing = queuedUtteranceRef.current.length > 0 && !repeated;
    // An "ignore" drops the words; past a backchannel's length a wrong one loses a whole correction, so it waits instead.
    const tooLongToIgnore = text.trim().split(/\s+/).length > IGNORE_MAX_WORDS;
    const awaitingApproval = voicedApprovalCount(useChatRuntimeStore.getState()) > 0;
    const action = interjectUnderApproval(
      sameAsk
        ? "ignore"
        : verdict !== "restart" && repeated
          ? "restart"
          : (continuing || tooLongToIgnore) && verdict === "ignore"
            ? "queue"
            : verdict,
      awaitingApproval,
    );
    return { action, sameAsk, continuing, awaitingApproval };
  }

  async function askEarly(fragment: string): Promise<EarlyAnswer | null> {
    if (voiceModeRef.current !== "active") return null;
    if (!auiRef.current.thread().getState().isRunning) return null;
    if (!latestAssistantIsReasoning() && !replyNotStarted()) return { verdict: "restart", kind: null };
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
    endTurns("barged", true);
    if (isSpeakingRef.current || isPlayingRef.current) stop();
    const f = fillerRef.current;
    if (f?.logId) {
      reportLayaOutcome(f.logId, "cut", { queued: f.queued });
      f.logId = null;
    }
    try {
      auiRef.current.thread().cancelRun();
    } catch {
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


  const confirmationCount = useChatRuntimeStore(voicedApprovalCount);
  const confirmationsSeenRef = useRef(0);
  useEffect(() => {
    const before = confirmationsSeenRef.current;
    confirmationsSeenRef.current = confirmationCount;
    if (confirmationCount <= before || voiceModeRef.current !== "active") return;
    // Held speech would sit behind the card: send it now; the cancel denies the parked call and the new run sees both.
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
      chatSwitching: chatModelStatus === "switching",
      sttDownload,
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
  }, [voiceMode, hasChatModel, chatModelStatus, sttDownload, voiceSlotLoading, voiceHearing, voiceTranscribing, isThreadRunning, isReasoning, isSpeaking, isPlaying, voicePaused, setVoiceOrbState]);

  const wasSlotLoadingRef = useRef(voiceSlotLoading);
  useEffect(() => {
    const was = wasSlotLoadingRef.current;
    wasSlotLoadingRef.current = voiceSlotLoading;
    if (!was || voiceSlotLoading) return;
    if (voiceModeRef.current !== "active" || !hasChatModelRef.current) return;
    // Only a voice that loaded: warming a failed one fails, reloads it and loops.
    if (useChatRuntimeStore.getState().voiceSlotLoaded) void warmAsidesIfTts(asidePhrases());
    // A failed load also ends loading: re-sending would re-park and reload in a loop.
    if (pendingForVoiceRef.current.length && useChatRuntimeStore.getState().voiceSlotLoaded) {
      // Cutting the session mid-transcription lost a turn; its own finish sends the lot.
      if (isUtteranceInFlight()) return;
      submitRef.current("");
      return;
    }
    resumeListen();
  }, [voiceSlotLoading, resumeListen, warmAsidesIfTts]);

  // The slot can already be serving when voice mode starts (loaded from the picker
  // or the API first), so the loading transition above never fires: warm here too.
  const voiceSlotLoaded = useChatRuntimeStore((s) => s.voiceSlotLoaded);
  useEffect(() => {
    if (voiceMode !== "active" || !voiceSlotLoaded || voiceSlotLoading) return;
    void warmAsidesIfTts(asidePhrases());
  }, [voiceMode, voiceSlotLoaded, voiceSlotLoading, warmAsidesIfTts]);

  useEffect(() => () => {
    if (synthGapTimerRef.current) clearTimeout(synthGapTimerRef.current);
    orbDwellRef.current?.dispose();
  }, []);

  const activateLoop = useCallback(() => {
    if (voiceModeRef.current !== "active") clearHeldTurn();
    setLoopVoiceMode("active");
    voiceModeRef.current = "active";
    setVoiceModeState("active");
    // A read-aloud already playing would talk over the mic, and barge-in cannot see it.
    const thread = auiRef.current.thread();
    if (thread.getState().speech) {
      try {
        thread.stopSpeaking();
      } catch {
      }
    }
    if (!hasChatModelRef.current) return;
    // A transcriber still to download cannot hear: listening starts again once it is loaded.
    if (useChatRuntimeStore.getState().sttSlotDownload !== null) return;
    if (useChatRuntimeStore.getState().voiceSlotLoading) return;
    if (!auiRef.current.thread().getState().isRunning && !isSpeakingRef.current) {
      auiRef.current.composer().startDictation();
    }
  }, []);

  const armedForModelRef = useRef(hasChatModel);
  // The chat model the loop was talking to when it went away (a swap, an eject, a load).
  const heldForModelRef = useRef<string | null>(null);
  useEffect(() => {
    if (armedForModelRef.current === hasChatModel) return;
    armedForModelRef.current = hasChatModel;
    const checkpoint = useChatRuntimeStore.getState().params.checkpoint ?? null;
    if (!hasChatModel) {
      heldForModelRef.current = readyCheckpointRef.current;
      voiceTrace("engine.chatModelAway", { checkpoint: heldForModelRef.current });
      // Words already heard become a draft (submit keeps them in the box), nothing new is heard,
      // and the reply is held (the hold effect) while the swap asks to stop it or loads.
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
      return;
    }
    const changed = heldForModelRef.current !== checkpoint;
    voiceTrace("engine.chatModelBack", { checkpoint, changed });
    // A different model: the old reply was stopped by the swap, so what is left of its audio goes too.
    if (changed) stop();
    if (voiceModeRef.current !== "active") return;
    resumeListen();
  }, [hasChatModel, resumeListen, stop]);
  // The picked transcriber went missing (or its download finished and it loaded): stop hearing what
  // cannot be transcribed, and listen again once it can be.
  const sttWaitingRef = useRef(sttDownload !== null);
  useEffect(() => {
    const waiting = sttDownload !== null;
    if (sttWaitingRef.current === waiting) return;
    sttWaitingRef.current = waiting;
    voiceTrace("engine.sttWaiting", { waiting, sttDownload });
    if (waiting) {
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
      return;
    }
    if (voiceModeRef.current === "active") resumeListen();
  }, [sttDownload, resumeListen]);
  // One rule for a held reply: the user paused, or there is no chat model to talk to (a swap asking
  // or loading). After the effect above, so a swapped-out reply is dropped before the hold lifts.
  // A pause outlives this engine (New Chat remounts it), so the new player starts held too.
  useEffect(() => {
    holdPlayback(voiceMode === "active" && (pausedForHold || !hasChatModel));
  }, [voiceMode, pausedForHold, hasChatModel, holdPlayback]);

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
        beginStreamRef.current({ turnId: liveVoiceTurn() ?? undefined });
        streamBegunRef.current = true;
        streamGenRef.current++;
        spokenRunRef.current =
          speaksWithChatModel() && lastUserTurnHasAudio(auiRef.current.thread().getState().messages);
        spokenClipsRef.current = 0;
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
        // isSpeaking may already be true (a typed follow-up over a reply): the mic re-arms per stream, not per flip.
        setStreamSeq((n) => n + 1);
        if (streamPollRef.current) clearInterval(streamPollRef.current);
        let quietFallbackAt: number | null = null;
        let writingDone = false;
        const FALLBACK_REPEAT_GAP_MS = 8000;
        streamPollRef.current = setInterval(() => {
          const f = fillerRef.current;
          const text = latestAssistantText();
          const live = liveVoiceTurn();
          // Sampled at this poll's 150 ms, the first text part (thinking is not text).
          if (live && text && _firstTokenTurn !== live) {
            _firstTokenTurn = live;
            voiceTrace("turn.firstToken", { turnId: live, approxMs: 150 });
          }
          if (f && !f.queued) {
          } else if (!spokenRunRef.current) {
            feedTextRef.current(text);
          }
          const now = Date.now();
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
      const added = now.startsWith(spoken.text) ? now.slice(spoken.text.length) : now;
      spokenReplyRef.current = { ...spoken, text: now };
      if (voiceModeRef.current === "active" && added.trim()) speakRef.current(added, { base: now.length - added.length });
      return;
    }
    setIsReasoning(false);
    const streamBegun = streamBegunRef.current;
    streamBegunRef.current = false;

    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    // The model was told about the filler, so a reply inside the delay still plays after it.
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
    const spokenRun = spokenRunRef.current;
    spokenRunRef.current = false;
    if (spokenRun && streamBegun && spokenClipsRef.current > 0) {
      // The reply played as it streamed: end the stream once the last chunk has.
      const endMsgs = auiRef.current.thread().getState().messages;
      const endTail = endMsgs.length ? endMsgs[endMsgs.length - 1] : null;
      spokenReplyRef.current = endTail?.role === "assistant" ? { id: endTail.id, text } : null;
      const gen = streamGenRef.current;
      setTimeout(() => {
        // A Regenerate inside the beat owns the stream now: finalizing it would drop the new reply's clips.
        if (gen === streamGenRef.current && voiceModeRef.current === "active") endClipsRef.current();
      }, 150);
      return;
    }
    if (!text) {
      if (!streamBegun) {
        endTurns("noTts", true);
        resumeListen();
        return;
      }
      // A short answer can land whole after isRunning flipped: read again after a beat. A run with
      // no answer text still closes its stream (playback end resumes listening).
      setTimeout(() => {
        // A run that began meanwhile opened its own stream; the one this run owned is already gone.
        if (streamBegunRef.current) return;
        const late = latestAssistantText();
        voiceTrace("engine.lateAnswer", { chars: late.length });
        if (late) {
          const lateMsgs = auiRef.current.thread().getState().messages;
          const lateTail = lateMsgs.length ? lateMsgs[lateMsgs.length - 1] : null;
          spokenReplyRef.current = lateTail?.role === "assistant" ? { id: lateTail.id, text: late } : null;
        }
        endStreamRef.current(late);
      }, 150);
      return;
    }
    const endMsgs = auiRef.current.thread().getState().messages;
    const endTail = endMsgs.length ? endMsgs[endMsgs.length - 1] : null;
    spokenReplyRef.current = endTail?.role === "assistant" ? { id: endTail.id, text } : null;
    // Read again after a beat: the last token can land after isRunning flipped.
    if (streamBegun) {
      const gen = streamGenRef.current;
      setTimeout(() => {
        // A Regenerate or Continue inside the beat owns the stream now: closing it would speak this reply again.
        if (gen !== streamGenRef.current || voiceModeRef.current !== "active") return;
        const final = latestAssistantText();
        const said = final.length >= text.length ? final : text;
        if (spokenReplyRef.current) spokenReplyRef.current = { ...spokenReplyRef.current, text: said };
        endStreamRef.current(said);
      }, 150);
    } else {
      speakRef.current(text, { turnId: liveVoiceTurn() ?? undefined });
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
  }, [isSpeaking, streamSeq]);

  // No session-end watcher: the adapter raises requestVoiceSubmit itself.

  const turnOff = useCallback(() => {
    // Dismissing the setup panel leaves a typed reply running: only an active voice owns the run.
    const wasActive = voiceModeRef.current === "active";
    _voicedThreadId = null;
    spokenRunRef.current = false;
    endTurns("cancelled");
    setLoopVoiceMode("off");
    voiceModeRef.current = "off";
    setVoiceModeState("off");
    useChatRuntimeStore.getState().setVoiceMode("off");
    heardSnapshotRef.current = null;
    if (voiceCoalesceTimerRef.current) {
      clearTimeout(voiceCoalesceTimerRef.current);
      voiceCoalesceTimerRef.current = null;
    }
    clearHeldTurn();
    forgetVoiceBox();
    clearPendingVoiceNotes();
    earlyInterjectRef.current?.reset();
    clearFiller();
    setPendingFiller(null);
    queuedUtteranceRef.current = [];
    pendingForVoiceRef.current = [];
    clearVoiceDecisions();
    holdPlayback(false);
    if (streamPollRef.current) {
      clearInterval(streamPollRef.current);
      streamPollRef.current = null;
    }
    // stop() drops the player's stream; a run ending after re-activation must speak, not close a dead stream.
    streamBegunRef.current = false;
    stop();
    pauseOutputSpectrum();
    const thread = auiRef.current.thread();
    if (wasActive && thread.getState().isRunning) {
      try {
        thread.cancelRun();
      } catch {
      }
    }
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
  }, [stop, clearFiller, holdPlayback]);

  const toggle = useCallback(() => {
    if (voiceModeRef.current !== "off") {
      turnOff();
      return;
    }
    setLoopVoiceMode("configuring");
    voiceModeRef.current = "configuring";
    setVoiceModeState("configuring");
    useChatRuntimeStore.getState().setVoiceMode("configuring");
    // The pick is not cleared: null means silence now, so voice would open mute every time.
    primeAudio();
  }, [turnOff, primeAudio]);


  const threadReset = useCallback((prevThreadId: string | null) => {
    if (voiceModeRef.current === "off") return;
    // The aui handle already points at the new thread, so stop the old run by id.
    const oldRun = voiceRunToStop(
      prevThreadId,
      _voicedThreadId,
      useChatRuntimeStore.getState().runningByThreadId,
    );
    voiceTrace("engine.threadReset", { prevThreadId, stopped: oldRun, paused: useChatRuntimeStore.getState().voicePaused });
    endTurns("cancelled");
    if (oldRun) stopChatThread(oldRun);
    _voicedThreadId = null;
    spokenRunRef.current = false;
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
    forgetVoiceBox();
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
      rearmSeqRef.current += 1;
      // A project landing swaps the composer without a thread switch: a held phrase must not submit there.
      clearHeldTurn();
      forgetVoiceBox();
      if (voiceCoalesceTimerRef.current) clearTimeout(voiceCoalesceTimerRef.current);
      if (streamPollRef.current) clearInterval(streamPollRef.current);
    };
  }, []);

  useEffect(() => {
    registerVoiceTypedSend(() => {
      if (voiceModeRef.current !== "active") return;
      clearHeldTurn();
      // A verdict still out for the last utterance belongs to the run this send replaces.
      clearPendingVoiceNotes();
    });
    return () => registerVoiceTypedSend(null);
  }, []);

  useEffect(() => {
    registerVoiceToggle(toggle);
    return () => {
      registerVoiceToggle(null);
    };
  }, [toggle]);

  useEffect(() => {
    registerVoiceStopSpeaking(() => {
      if (voiceModeRef.current !== "active") return;
      voiceTrace("engine.stopSpeaking", { speaking: isSpeakingRef.current, playing: isPlayingRef.current });
      endTurns("cancelled");
      // The hold effect releases the player.
      if (useChatRuntimeStore.getState().voicePaused) useChatRuntimeStore.getState().setVoicePaused(false);
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
        }
      }
      // The open barge-in session holds the epoch the clear just retired: its next turn would be dropped.
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
      resumeListen();
    });
    return () => registerVoiceStopSpeaking(null);
  }, [stop, resumeListen, clearFiller]);

  useEffect(() => {
    registerVoicePauseToggle(() => {
      if (voiceModeRef.current !== "active") return;
      const store = useChatRuntimeStore.getState();
      const next = !store.voicePaused;
      voiceTrace("engine.pause", { paused: next, speaking: isSpeakingRef.current, playing: isPlayingRef.current });
      store.setVoicePaused(next);
    });
    return () => registerVoicePauseToggle(null);
  }, []);

  const resumeOrFlush = useCallback(() => {
    // Only once the voice is up: after a failed load each silent session would re-park and reload in a loop.
    const rt = useChatRuntimeStore.getState();
    if (pendingForVoiceRef.current.length && !rt.voiceSlotLoading && rt.voiceSlotLoaded) {
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

  // Taken as a barge-in lands: what had played, and where the speaker was.
  const takeHeardSnapshot = useCallback((): HeardSnapshot | null => {
    if (!useVoiceSettingsStore.getState().heardCut) return null;
    const played = playedLogRef.current();
    const msgs = auiRef.current.thread().getState().messages;
    const tail = msgs.length ? msgs[msgs.length - 1] : null;
    if (!played || tail?.role !== "assistant") return null;
    return { messageId: tail.id, position: playbackPositionRef.current(), ...played };
  }, []);

  // Saved once no run is live (a thread import must not land mid-run); until then the cut applies from memory.
  // With another thread open it waits for its own, and a failed save is tried again, for up to half an hour.
  // One loop per reply: each pass saves the latest pending cut, so a second barge-in joins the first's loop.
  const saveHeardCut = useCallback((messageId: string) => {
    if (_heardSaving.has(messageId)) return;
    _heardSaving.add(messageId);
    let waited = 0;
    let failures = 0;
    const done = () => _heardSaving.delete(messageId);
    const later = (ms: number) => {
      waited += ms;
      if (waited <= HEARD_SAVE_PATIENCE_MS) setTimeout(tick, ms);
      else {
        // The cut still applies from memory until reload; the trace says it never reached the server.
        voiceTrace("voice.heardSaveGaveUp", { messageId });
        done();
      }
    };
    const tick = () => {
      const thread = auiRef.current.thread();
      const cut = pendingHeardCut(messageId);
      if (!cut) {
        done();
        return;
      }
      if (!thread.getState().messages.some((m) => m.id === messageId)) {
        later(5000);
        return;
      }
      if (thread.getState().isRunning) {
        later(500);
        return;
      }
      void setThreadMessageVoiceHeard({
        thread: { export: () => thread.export(), import: (data) => thread.import(data) },
        messageId,
        remoteId:
          auiRef.current.threadListItem().getState().remoteId ||
          useChatRuntimeStore.getState().activeThreadId ||
          undefined,
        isIncognito: useChatRuntimeStore.getState().incognito,
        heard: { offset: cut.offset, at: Date.now() },
      }).then(
        () => {
          // A later cut taken while this one saved goes next.
          if (pendingHeardCut(messageId) !== cut) return tick();
          clearPendingHeardCut(messageId);
          done();
        },
        (error: unknown) => {
          const refused = error instanceof ChatMessageProtectedError;
          voiceTrace(refused ? "voice.heardSaveRefused" : "voice.heardSaveFailed", {
            messageId,
            message: error instanceof Error ? error.message : String(error),
          });
          // A protected row stays protected (chat-api.ts): no retry, the cut keeps applying from memory.
          if (!refused) later(2000 * 2 ** Math.min(failures++, 6));
          else done();
        },
      );
    };
    tick();
  }, []);

  // A confirmed barge-in: the cut is in memory before the next turn's request is built.
  const recordHeardCut = useCallback(() => {
    const snap = heardSnapshotRef.current;
    heardSnapshotRef.current = null;
    if (!snap) return;
    const msgs = auiRef.current.thread().getState().messages;
    // Only the reply still being cut: a snapshot older than the tail heard some other reply.
    const msg = msgs.length ? msgs[msgs.length - 1] : null;
    if (!msg || msg.id !== snap.messageId) return;
    const text = withoutAudioPlayer(joinTextParts(msg.content as Array<{ type: string; text?: string }>).text);
    const cut = laterCut(
      laterCut(savedHeardCut(msg.metadata), pendingHeardCut(msg.id)),
      heardCut(splitSpoken(text), snap.log, snap.position, snap.base),
    );
    const settled = !("status" in msg) || msg.status?.type !== "running";
    voiceTrace("voice.heardCut", { messageId: msg.id, offset: cut?.offset ?? null, chars: text.trimEnd().length });
    // Heard to the end of a finished reply: nothing to cut.
    if (!cut || (settled && cut.offset >= text.trimEnd().length)) return;
    setPendingHeardCut(msg.id, cut);
    saveHeardCut(msg.id);
  }, [saveHeardCut]);
  heardCutRecorderRef.current = recordHeardCut;

  useEffect(() => {
    registerVoiceBargeIn(() => {
      voiceTrace("engine.bargeIn", { speaking: isSpeakingRef.current, playing: isPlayingRef.current });
      if (voiceModeRef.current !== "active") return;
      if (!isSpeakingRef.current && !isPlayingRef.current) return;
      // Before the pause, which stops the clock the position is read from.
      heardSnapshotRef.current = takeHeardSnapshot();
      pausePlayback();
    });
    registerVoiceBargeAbort(() => {
      voiceTrace("engine.bargeAbort", { speaking: isSpeakingRef.current, playing: isPlayingRef.current });
      // A false interruption cuts nothing.
      heardSnapshotRef.current = null;
      if (voiceModeRef.current !== "active") return;
      resumePlayback();
    });
    return () => {
      registerVoiceBargeIn(null);
      registerVoiceBargeAbort(null);
    };
  }, [pausePlayback, resumePlayback, takeHeardSnapshot]);

  submitRef.current = submitTranscript;
  useEffect(() => {
    // Not submitTranscript itself: the bridge's second argument is the turn, not `typed`.
    registerVoiceSubmit((text, meta) => {
      submitTranscript(text, false, meta);
    });
    return () => {
      registerVoiceSubmit(null);
    };
  }, [submitTranscript]);

  useEffect(() => {
    // The session is cancelled, not finished, so nothing it captured goes out a second time.
    registerVoiceTypedSubmit((box) => {
      if (voiceModeRef.current !== "active") return false;
      const shown = [pendingForVoiceRef.current.join(" "), getHeldTurnText()]
        .filter(Boolean)
        .join(" ");
      const paused = useChatRuntimeStore.getState().voicePaused;
      const revealing = voiceRevealText();
      if (isRevealPrefix(box, revealing)) {
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


  return null;
};

// Ctrl+M toggles mute: Cmd+M is the window's Minimize on macOS.
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
  // Screen coordinates are already logical (CSS) pixels, so no scale factor applies.
  const cursorRef = useRef<{ x: number; y: number } | null>(null);
  const holdRef = useRef<HoldToStop | null>(null);
  // Made on the first press, never during render (refs are read only from handlers and effects).
  const pressHold = () => {
    if (!isTauri) return;
    holdRef.current ??= createHoldToStop(
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
    holdRef.current.press();
  };
  useEffect(() => () => holdRef.current?.dispose(), []);
  const cancel = () => holdRef.current?.cancel();
  return {
    handlers: {
      onPointerDown: (e: React.PointerEvent) => {
        if (e.button !== 0) return;
        firedRef.current = false;
        cursorRef.current = { x: e.screenX, y: e.screenY };
        pressHold();
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
  // Not started yet: the button opens voice setup here, beside the mic, with Start in it.
  if (voiceMode === "configuring") return <VoiceSetupPanel anchor="composer" />;

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
