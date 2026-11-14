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
  dropVoiceUtteranceAudio,
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
  isSameAsk,
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
  registerVoiceSpokenAudio,
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
import { chatModelSpeaks, isChatModelVoice, voiceHearsAudio } from "@/features/chat/voice/voice-models";
import { voiceServing } from "@/features/chat/voice/voice-slot";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
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
import {
  coalescedTurnAudio,
  lastUserTurnHasAudio,
  voiceTurnAttachment,
  withoutAudioPlayer,
} from "./turn-audio";
import { isRevealPrefix, isTypedVoiceSubmit } from "./typed-send";

let _prevRunning = false;
// Module-level: New Chat remounts VoiceEngine, and the reset after it must still find the run to stop.
let _voicedThreadId: string | null = null;

// Filtered as the floating orb does: a Compare pane or background chat's card is not voice mode's to speak.
function voicedApprovalCount(s: ReturnType<typeof useChatRuntimeStore.getState>): number {
  const pending = pendingApproval(s.toolConfirmations, s.activeThreadId, (id) => describeToolCall(id) !== null);
  return pending ? pending.more + 1 : 0;
}

function speaksWithChatModel(): boolean {
  const s = useChatRuntimeStore.getState();
  return isChatModelVoice(s.selectedVoiceModelId) && chatModelSpeaks(s);
}

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
    // Listening never waits for a loading voice (a shut mic lost speech); only the send does.
    // Dictation can lag clearing after a session ends; clicking Dictate while it is set would toggle it OFF.
    const MAX_ATTEMPTS = 5;
    const RETRY_MS = 50;

    const clickDictate = () => {
      const fresh = auiRef.current.composer();
      // Already listening: startDictation() on a live session tears the mic down.
      if (fresh.getState().dictation) return;
      // assistant-ui appends transcripts onto the box text and the adapter emits the whole turn, so leftovers double up.
      if (voiceWroteBox(fresh.getState().text) && !isComposerHeldForEdit()) fresh.setText("");
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
  const feedClipRef = useRef(feedClip);
  feedClipRef.current = feedClip;
  const endClipsRef = useRef(endClips);
  endClipsRef.current = endClips;
  const spokenRunRef = useRef(false);
  const spokenClipsRef = useRef(0);
  const streamBegunRef = useRef(false);
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
      return withoutAudioPlayer(text);
    }
    return "";
  }, []);

  const commitUtterance = useCallback((said: string) => {
    const composer = auiRef.current.composer();
    // A barge-in cuts playback without its end firing, so held utterances go now or surface later as a stale prompt.
    const held = queuedUtteranceRef.current.filter((q) => q !== said);
    queuedUtteranceRef.current = [];
    const text = held.length ? [...held, said].join(" ") : said;
    if (held.length) voiceTrace("engine.foldHeld", { held, text });
    // Drained on every send, so speech kept while the model could not answer in it never rides along later.
    const taken = takeVoiceTurnAudio();
    // Non-GGUF backends refuse audio beside an image or video, so such a turn goes as its transcript.
    const visual = composer
      .getState()
      .attachments.some((a) => a.type === "image" || /^(image|video)\//.test(a.contentType ?? ""));
    const turnAudio =
      !visual && voiceHearsAudio(useChatRuntimeStore.getState(), useVoiceSettingsStore.getState().micTarget)
        ? taken
        : null;
    // Cover the tail clip (isPlaying) too, not just an active streaming session (isSpeaking).
    voiceTrace("engine.commitUtterance", { text, speaking: isSpeakingRef.current, playing: isPlayingRef.current });
    if (isSpeakingRef.current || isPlayingRef.current) stop();
    const thread = auiRef.current.thread();
    const cutRun = thread.getState().isRunning;
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
    const deferredClear = (n: number) => {
      if (voiceModeRef.current !== "active") return;
      const fresh = auiRef.current.composer();
      // Only the transcript the composer refills: a draft typed since, or another chat's, stays.
      if (!voiceWroteBox(fresh.getState().text)) return;
      fresh.setText("");
      if (n < 5) setTimeout(() => deferredClear(n + 1), DEFERRED_CLEAR_MS);
    };
    const sendNow = () => {
      composer.setText(text);
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
      void composer
        .addAttachment(new File([bytes], "voice.wav", { type: "audio/wav" }))
        .catch((error: unknown) => voiceTrace("engine.turnAudioAttachFailed", { message: String(error) }))
        .then(() => {
          if (voiceModeRef.current === "active") sendNow();
        });
    };

    // First message: the append + startRun path goes stale across the first-send remount.
    if (thread.getState().messages.length === 0) {
      // send() takes the box, not `text`: parked turns were never in it (the re-arm
      // clears it), so an empty box here sent nothing at all.
      sendThroughComposer();
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
    const fireCoalescedRun = (attempt: number) => {
      voiceCoalesceTimerRef.current = null;
      if (voiceModeRef.current !== "active") return;
      const rt = useChatRuntimeStore.getState();
      if ((rt.voiceHearing || rt.voiceTranscribing) && Date.now() - armedAt < VOICE_HEARING_HOLD_MAX_MS) {
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
        }
        return;
      }
      const msgs = t.getState().messages;
      if (msgs.length <= before && attempt < VOICE_COALESCE_MAX_RETRIES) {
        voiceCoalesceTimerRef.current = setTimeout(() => fireCoalescedRun(attempt + 1), VOICE_COALESCE_POLL_MS);
        return;
      }
      const last = msgs.length > 0 ? msgs[msgs.length - 1] : null;
      if (!last || last.role !== "user") return;
      // A new run while the old one is still "cancelling" is refused outright: ask the server, not the local state.
      const threadId = useChatRuntimeStore.getState().activeThreadId;
      const turn = voiceNotesEpoch();
      const start = () => {
        if (voiceModeRef.current !== "active") return;
        if (voiceNotesEpoch() !== turn) return;
        if (useChatRuntimeStore.getState().activeThreadId !== threadId) return;
        try {
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
    const holdOn = isHoldOn(said);
    if (holdOn) voiceTrace("engine.holdOn", { text: said });
    voiceCoalesceTimerRef.current = setTimeout(
      () => fireCoalescedRun(0),
      holdOn ? VOICE_HOLD_COALESCE_MS : cutRun ? VOICE_RESTART_COALESCE_MS : VOICE_COALESCE_MS,
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

  const sendTranscriptRef = useRef<(text: string, typed?: boolean) => void>(() => {});
  const sendTranscript = useCallback(
    (text: string, typed?: boolean) => sendTranscriptRef.current(text, typed),
    [],
  );
  const submitTranscript = useCallback((spoken?: string, typed?: boolean) => {
    if (voiceModeRef.current !== "active") return;
    const composer = auiRef.current.composer();
    // A typed send is the whole box already, draft included.
    const draft = spoken && !typed ? takeTypedDraft(composer.getState().text) : "";
    if (composer.getState().dictation) composer.stopDictation();
    let text = (spoken ?? composer.getState().text).trim();
    if (draft) text = `${draft} ${text}`;
    voiceTrace("engine.submit", { text });
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
    const sendNow = () => sendTranscript(text, typed);
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

  function sendTranscriptImpl(text: string, typed?: boolean) {
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
    // Speech during thinking is not always a barge-in: the interject model decides. Typed text is never held.
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
          else dropVoiceUtteranceAudio();
          composer.setText("");
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
    commitUtterance(text);
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

  async function askEarly(fragment: string): Promise<EarlyAnswer | null> {
    if (voiceModeRef.current !== "active") return null;
    if (!auiRef.current.thread().getState().isRunning) return null;
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
    void warmAsidesIfTts(asidePhrases());
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
        if (streamPollRef.current) clearInterval(streamPollRef.current);
        let quietFallbackAt: number | null = null;
        let writingDone = false;
        const FALLBACK_REPEAT_GAP_MS = 8000;
        streamPollRef.current = setInterval(() => {
          const f = fillerRef.current;
          const text = latestAssistantText();
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
      setTimeout(() => endClipsRef.current(), 150);
      return;
    }
    if (!text) {
      // A run with no answer text still closes its stream: playback end resumes listening.
      if (streamBegun) endStreamRef.current("");
      else resumeListen();
      return;
    }
    const endMsgs = auiRef.current.thread().getState().messages;
    const endTail = endMsgs.length ? endMsgs[endMsgs.length - 1] : null;
    spokenReplyRef.current = endTail?.role === "assistant" ? { id: endTail.id, text } : null;
    // Read again after a beat: the last token can land after isRunning flipped.
    if (streamBegun) {
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

  const turnOff = useCallback(() => {
    _voicedThreadId = null;
    spokenRunRef.current = false;
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

  useEffect(() => {
    registerVoiceStopSpeaking(() => {
      if (voiceModeRef.current !== "active") return;
      voiceTrace("engine.stopSpeaking", { speaking: isSpeakingRef.current, playing: isPlayingRef.current });
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
        }
      }
      // The open barge-in session holds the epoch the clear just retired: its next turn would be dropped.
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
      resumeListen();
    });
    return () => registerVoiceStopSpeaking(null);
  }, [stop, resumeListen, clearFiller, holdPlayback]);

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
