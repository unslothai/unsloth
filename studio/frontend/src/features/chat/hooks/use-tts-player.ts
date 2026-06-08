// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useCallback, useEffect, useRef, useState } from "react";

export const TTS_AUDIO_TYPES = new Set(["snac", "csm", "bicodec", "dac"]);

export function useTtsPlayer(
  audioType: string | null | undefined,
  onPlaybackEnd?: () => void,
): {
  isSpeaking: boolean;
  speak(text: string): void;
  stop(): void;
} {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const onPlaybackEndRef = useRef(onPlaybackEnd);
  onPlaybackEndRef.current = onPlaybackEnd;

  const isTtsModel = TTS_AUDIO_TYPES.has(audioType ?? "");

  const revokeUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  const stopTts = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      audioRef.current = null;
    }
    revokeUrl();
  }, [revokeUrl]);

  const stopSynth = useCallback(() => {
    if (utteranceRef.current) {
      window.speechSynthesis.cancel();
      utteranceRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    stopTts();
    stopSynth();
    setIsSpeaking(false);
  }, [stopTts, stopSynth]);

  const speak = useCallback(
    async (text: string) => {
      stop();
      if (!text) return;

      if (isTtsModel) {
        setIsSpeaking(true);
        try {
          const response = await authFetch("/api/audio/speech", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input: text, voice: "default" }),
          });
          if (!response.ok) throw new Error("TTS request failed");
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          objectUrlRef.current = url;
          const audio = new Audio(url);
          audioRef.current = audio;
          const cleanup = () => {
            if (objectUrlRef.current === url) {
              URL.revokeObjectURL(url);
              objectUrlRef.current = null;
            }
            if (audioRef.current === audio) audioRef.current = null;
            setIsSpeaking(false);
            onPlaybackEndRef.current?.();
          };
          audio.onended = cleanup;
          audio.onerror = cleanup;
          audio.play();
        } catch {
          setIsSpeaking(false);
          onPlaybackEndRef.current?.();
        }
      } else {
        if (typeof window === "undefined" || !("speechSynthesis" in window)) {
          onPlaybackEndRef.current?.();
          return;
        }
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => {
          utteranceRef.current = null;
          setIsSpeaking(false);
          onPlaybackEndRef.current?.();
        };
        utterance.onerror = () => {
          utteranceRef.current = null;
          setIsSpeaking(false);
          onPlaybackEndRef.current?.();
        };
        utteranceRef.current = utterance;
        window.speechSynthesis.speak(utterance);
        setIsSpeaking(true);
      }
    },
    [isTtsModel, stop],
  );

  useEffect(() => {
    return () => {
      stopTts();
      stopSynth();
    };
  }, [stopTts, stopSynth]);

  return { isSpeaking, speak, stop };
}
