// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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
} from "@/features/chat/adapters/studio-voice-loop-adapter";
import { useTtsPlayer } from "@/features/chat/hooks/use-tts-player";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { deriveOrbState } from "@/features/chat/voice/orb-state";
import {
  getVoiceMode,
  registerVoiceBargeAbort,
  registerVoiceBargeIn,
  registerVoiceListen,
  registerVoicePlayback,
  registerVoiceResume,
  registerVoiceSubmit,
  registerVoiceThreadReset,
  registerVoiceToggle,
  setVoiceMode as setLoopVoiceMode,
} from "@/features/chat/voice/voice-loop-bridge";
import { cn } from "@/lib/utils";
import { useAui } from "@assistant-ui/react";
import { MessageSquareIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type FC } from "react";

let _prevRunning = false;

const VOICE_COALESCE_MS = 650;
const VOICE_COALESCE_MAX_RETRIES = 15;
const SYNTH_GAP_MS = 350;

export { VoiceOrb };
export { SttModelSelector } from "@/components/assistant-ui/stt-model-selector";
export { VoiceModelSelector } from "@/components/assistant-ui/voice-model-selector";
export { VadModelSelector } from "@/components/assistant-ui/vad-model-selector";
export { loadVoice, unloadVoice, voiceServing } from "@/features/chat/voice/voice-slot";

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
    if (useChatRuntimeStore.getState().voiceSlotLoading) return;
    const MAX_ATTEMPTS = 5;
    const RETRY_MS = 50;

    const clickDictate = () => {
      const fresh = auiRef.current.composer();
      // Already listening: startDictation() on a live session tears the mic down.
      if (fresh.getState().dictation) return;
      if (fresh.getState().text) {
        fresh.setText("");
      }
      // Via the runtime, not the Dictate button: voice mode hides it.
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

  const {
    isSpeaking,
    isPlaying,
    speak,
    beginStream,
    feedText,
    endStream,
    stop,
    primeAudio,
    pausePlayback,
    resumePlayback,
  } = useTtsPlayer(resumeListen);
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

  const submitTranscript = useCallback(() => {
    if (voiceModeRef.current !== "active") return;
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
    const text = composer.getState().text.trim();
    if (!text) {
      // No speech is not a barge-in: cutting the reply here truncated it after the first word.
      resumeListen();
      return;
    }
    if (isSpeakingRef.current || isPlayingRef.current) stop();
    const thread = auiRef.current.thread();
    if (thread.getState().isRunning) {
      try {
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
      composer.send();
      composer.setText("");
      setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
      return;
    }

    composer.setText("");
    setTimeout(() => deferredClear(1), DEFERRED_CLEAR_MS);
    resumeListen();
    const fireCoalescedRun = (attempt: number) => {
      voiceCoalesceTimerRef.current = null;
      if (voiceModeRef.current !== "active") return;
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
      try {
        t.startRun({ parentId: last.id });
      } catch {
      }
    };
    if (voiceCoalesceTimerRef.current) clearTimeout(voiceCoalesceTimerRef.current);
    voiceCoalesceTimerRef.current = setTimeout(
      () => fireCoalescedRun(0),
      VOICE_COALESCE_MS,
    );
  }, [resumeListen, stop]);

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
    if (synthGapTimerRef.current) return;
    synthGapTimerRef.current = setTimeout(() => {
      synthGapTimerRef.current = null;
      if (
        voiceModeRef.current === "active" &&
        isSpeakingRef.current &&
        !isPlayingRef.current
      ) {
        setVoiceOrbState("synthesizing");
      }
    }, SYNTH_GAP_MS);
  }, [voiceMode, hasChatModel, voiceSlotLoading, voiceHearing, voiceTranscribing, isThreadRunning, isReasoning, isSpeaking, isPlaying, setVoiceOrbState]);

  const wasSlotLoadingRef = useRef(voiceSlotLoading);
  useEffect(() => {
    const was = wasSlotLoadingRef.current;
    wasSlotLoadingRef.current = voiceSlotLoading;
    if (!was || voiceSlotLoading) return;
    if (voiceModeRef.current !== "active" || !hasChatModelRef.current) return;
    resumeListen();
  }, [voiceSlotLoading, resumeListen]);

  useEffect(() => () => {
    if (synthGapTimerRef.current) clearTimeout(synthGapTimerRef.current);
  }, []);

  const activateLoop = useCallback(() => {
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
      if (voiceModeRef.current === "active") {
        const composer = auiRef.current.composer();
        if (composer.getState().dictation) composer.stopDictation();
        beginStreamRef.current();
        streamBegunRef.current = true;
        if (streamPollRef.current) clearInterval(streamPollRef.current);
        streamPollRef.current = setInterval(() => {
          feedTextRef.current(latestAssistantText());
          setIsReasoning(latestAssistantIsReasoning());
        }, 150);
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
    if (voiceModeRef.current !== "active") return;
    // No isSpeaking guard: beginStream supersedes, and skipping would dead-end the loop.

    const text = latestAssistantText();
    if (!text) {
      resumeListen();
      return;
    }
    if (streamBegun) {
      endStreamRef.current(text);
    } else {
      speakRef.current(text);
    }
  }, [isThreadRunning, resumeListen, latestAssistantText]);

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

  const toggle = useCallback(() => {
    const next: "off" | "configuring" =
      voiceModeRef.current === "off" ? "configuring" : "off";
    setLoopVoiceMode(next);
    voiceModeRef.current = next;
    setVoiceModeState(next);
    useChatRuntimeStore.getState().setVoiceMode(next);

    if (next === "configuring") {
      // The pick is kept: null means silence now that the browser voice is gone.
      primeAudio();
    }

    if (next === "off") {
      if (voiceCoalesceTimerRef.current) {
        clearTimeout(voiceCoalesceTimerRef.current);
        voiceCoalesceTimerRef.current = null;
      }
      clearHeldTurn();
      stop();
      const composer = auiRef.current.composer();
      if (composer.getState().dictation) composer.stopDictation();
    }
  }, [stop, primeAudio]);

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
    stop();
    const composer = auiRef.current.composer();
    if (composer.getState().dictation) composer.stopDictation();
    if (voiceModeRef.current === "active") resumeListen();
  }, [stop, resumeListen]);

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
    registerVoiceResume(resumeListen);
    return () => {
      registerVoiceResume(null);
    };
  }, [resumeListen]);

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

  useEffect(() => {
    registerVoiceSubmit(submitTranscript);
    return () => {
      registerVoiceSubmit(null);
    };
  }, [submitTranscript]);


  return null;
};

export const VoiceControlButton: FC = () => {
  const voiceMode = useChatRuntimeStore((s) => s.voiceMode);
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceMode = useChatRuntimeStore((s) => s.setVoiceMode);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);

  if (voiceMode === "off") return null;

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
