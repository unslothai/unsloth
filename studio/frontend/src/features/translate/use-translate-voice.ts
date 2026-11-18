// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  StudioDictationAdapter,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
} from "@/features/chat/adapters/studio-dictation-adapter";
import { StudioSpeechSynthesisAdapter } from "@/features/chat/adapters/studio-speech-synthesis-adapter";
import type { StudioDictationSession } from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { useCallback, useEffect, useRef, useState } from "react";

/** A second stop while the last phrase transcribes cancels it. No `language`: a local or
 *  custom model detects what is spoken; the browser's engine cannot and hears the user's. */
export function useDictation(
  onTranscript: (transcript: string) => void,
  language?: string,
) {
  const [dictating, setDictating] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const sessionRef = useRef<StudioDictationSession | null>(null);
  const finalizingRef = useRef(false);
  const deliver = useRef(onTranscript);
  deliver.current = onTranscript;
  const languageRef = useRef(language);
  languageRef.current = language;

  const end = useCallback((session: StudioDictationSession) => {
    if (sessionRef.current === session) sessionRef.current = null;
    finalizingRef.current = false;
    setFinalizing(false);
    setDictating(false);
  }, []);

  const start = useCallback(() => {
    if (sessionRef.current) return;
    if (!isStudioDictationAvailable()) {
      notifyStudioDictationUnavailable();
      return;
    }
    let session: StudioDictationSession;
    try {
      session = new StudioDictationAdapter({
        chatId: null,
        language: languageRef.current,
        detectLanguage: languageRef.current === undefined,
      }).listen();
    } catch {
      notifyStudioDictationUnavailable();
      return;
    }
    sessionRef.current = session;
    setDictating(true);
    session.onSpeech((result) => {
      const transcript = result.isFinal
        ? (result.transcript?.trim() ?? "")
        : "";
      if (transcript) deliver.current(transcript);
    });
    session.onEnd?.(() => end(session));
  }, [end]);

  const stop = useCallback(() => {
    const session = sessionRef.current;
    if (!session) return;
    if (finalizingRef.current) {
      session.cancel();
      end(session);
      return;
    }
    finalizingRef.current = true;
    setFinalizing(true);
    // The session stays alive while the last phrase transcribes; onEnd clears it.
    void session.stop().catch(() => {
      session.cancel();
      end(session);
    });
  }, [end]);

  useEffect(
    () => () => {
      sessionRef.current?.cancel();
      sessionRef.current = null;
    },
    [],
  );

  return {
    dictating,
    finalizing,
    start,
    stop,
  };
}

type Utterance = ReturnType<StudioSpeechSynthesisAdapter["speak"]>;

export function useReadAloud() {
  const enabled = useVoiceSettingsStore((s) => s.ttsEnabled);
  const [speaking, setSpeaking] = useState(false);
  const utteranceRef = useRef<Utterance | null>(null);

  const stop = useCallback(() => {
    const utterance = utteranceRef.current;
    utteranceRef.current = null;
    utterance?.cancel();
    setSpeaking(false);
  }, []);

  const speak = useCallback(
    (text: string, lang?: string) => {
      stop();
      const utterance = new StudioSpeechSynthesisAdapter().speak(text, lang, {
        systemFallback: true,
      });
      utteranceRef.current = utterance;
      setSpeaking(true);
      utterance.subscribe(() => {
        if (utterance.status.type !== "ended") return;
        if (utteranceRef.current !== utterance) return;
        utteranceRef.current = null;
        setSpeaking(false);
      });
    },
    [stop],
  );

  // Chrome lists its voices only after a first request, too late for the first reading.
  useEffect(() => {
    if (StudioSpeechSynthesisAdapter.systemVoicesSupported())
      window.speechSynthesis.getVoices();
    return () => utteranceRef.current?.cancel();
  }, []);

  return {
    speaking,
    speak,
    stop,
    // Stop stays reachable when read aloud is switched off mid-reading.
    available:
      speaking || (enabled && StudioSpeechSynthesisAdapter.isSupported()),
  };
}
