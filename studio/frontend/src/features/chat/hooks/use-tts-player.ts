// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useCallback, useEffect, useRef, useState } from "react";

export const TTS_AUDIO_TYPES = new Set(["snac", "csm", "bicodec", "dac"]);

// Split assistant text into sentence-sized chunks so the first one can start
// speaking while the rest are still synthesizing, instead of waiting for the
// whole response. Collapses whitespace, breaks on sentence punctuation and
// newlines, and falls back to the whole text when there's no boundary.
export function splitIntoSentences(text: string): string[] {
  const chunks = text
    .replace(/\s+/g, " ")
    .trim()
    .match(/[^.!?\n]*[.!?]+|\S[^.!?\n]*$/g);
  const out = (chunks ?? [text]).map((s) => s.trim()).filter(Boolean);
  return out;
}

export function useTtsPlayer(
  audioType: string | null | undefined,
  onPlaybackEnd?: () => void,
  voiceSlotLoaded = false,
): {
  isSpeaking: boolean;
  speak(text: string): void;
  stop(): void;
  primeAudio(): void;
} {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  // Bumped on every stop()/new speak()/unmount so an in-flight /api/audio/speech
  // response (or a queued sentence) can detect it was superseded and skip late
  // playback + state updates.
  const requestIdRef = useRef(0);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  // Resolver for the sentence currently awaiting playback, so stop() can unwind
  // the speak loop immediately (pause() doesn't fire "ended").
  const playResolveRef = useRef<(() => void) | null>(null);
  const onPlaybackEndRef = useRef(onPlaybackEnd);
  onPlaybackEndRef.current = onPlaybackEnd;

  const isTtsModel = TTS_AUDIO_TYPES.has(audioType ?? "") || voiceSlotLoaded;

  // Unlock Safari's audio autoplay policy by calling play()+pause() during
  // a synchronous user gesture. The element is reused for all subsequent plays
  // so the unlock survives async fetch callbacks.
  const primeAudio = useCallback(() => {
    if (typeof window === "undefined") return;
    if (!audioRef.current) {
      audioRef.current = new Audio();
    }
    audioRef.current.play().then(() => audioRef.current?.pause()).catch(() => {});
  }, []);

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
      audio.src = "";  // release current source but keep element alive for reuse
    }
    revokeUrl();
    // Release a sentence mid-playback so the speak loop's await resolves and the
    // chunk pipeline unwinds instead of hanging.
    playResolveRef.current?.();
    playResolveRef.current = null;
  }, [revokeUrl]);

  const stopSynth = useCallback(() => {
    if (utteranceRef.current) {
      window.speechSynthesis.cancel();
      utteranceRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    requestIdRef.current += 1;
    stopTts();
    stopSynth();
    setIsSpeaking(false);
  }, [stopTts, stopSynth]);

  // Play a single audio blob; resolves when it ends, errors, or is superseded by
  // a stop()/new speak().
  const playBlob = useCallback((blob: Blob, reqId: number) => {
    return new Promise<void>((resolve) => {
      if (requestIdRef.current !== reqId) {
        resolve();
        return;
      }
      const url = URL.createObjectURL(blob);
      objectUrlRef.current = url;
      const audio = audioRef.current ?? new Audio();
      audioRef.current = audio;
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        if (playResolveRef.current === done) playResolveRef.current = null;
        if (objectUrlRef.current === url) {
          URL.revokeObjectURL(url);
          objectUrlRef.current = null;
        }
        audio.onended = null;
        audio.onerror = null;
        resolve();
      };
      playResolveRef.current = done;
      audio.onended = done;
      audio.onerror = done;
      audio.src = url;
      // play() returns a promise that can reject (autoplay policy, decode error)
      // without ever firing onerror; settle so the loop can't stall.
      void audio.play().catch(done);
    });
  }, []);

  const speak = useCallback(
    async (text: string) => {
      stop();
      if (!text) return;
      // stop() above bumped the counter; this is now our request's id.
      const reqId = requestIdRef.current;
      const sentences = splitIntoSentences(text);
      if (sentences.length === 0) {
        onPlaybackEndRef.current?.();
        return;
      }

      if (isTtsModel) {
        setIsSpeaking(true);

        const synth = (sentence: string): Promise<Blob | null> =>
          authFetch("/api/audio/speech", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input: sentence, voice: "default" }),
          })
            .then((response) => (response.ok ? response.blob() : null))
            .catch(() => null);

        // Pipeline one sentence ahead: synthesize the next clip while the current
        // one plays, but keep synthesis strictly serial. The voice slot runs
        // --parallel 1 and /api/audio/speech has no lock, so overlapping calls
        // would clash; chaining the next synth off the current one guarantees a
        // single request in flight at a time.
        let nextSynth: Promise<Blob | null> = synth(sentences[0]);
        for (let i = 0; i < sentences.length; i++) {
          if (requestIdRef.current !== reqId) return;  // superseded
          const current = nextSynth;
          nextSynth =
            i + 1 < sentences.length
              ? current.then(() => synth(sentences[i + 1]))
              : Promise.resolve(null);
          const blob = await current;
          if (requestIdRef.current !== reqId) return;
          if (!blob) continue;  // skip a sentence that failed to synthesize
          await playBlob(blob, reqId);
        }

        if (requestIdRef.current !== reqId) return;
        setIsSpeaking(false);
        onPlaybackEndRef.current?.();
      } else {
        if (typeof window === "undefined" || !("speechSynthesis" in window)) {
          onPlaybackEndRef.current?.();
          return;
        }
        // Queue one utterance per sentence: gives the same chunked cadence and
        // sidesteps Chrome's long-utterance cutoff bug. Completion fires on the
        // last sentence; any error ends the loop.
        setIsSpeaking(true);
        let remaining = sentences.length;
        const finish = () => {
          if (utteranceRef.current === null) return;
          utteranceRef.current = null;
          setIsSpeaking(false);
          onPlaybackEndRef.current?.();
        };
        const utterances = sentences.map((sentence) => {
          const utterance = new SpeechSynthesisUtterance(sentence);
          utterance.onend = () => {
            remaining -= 1;
            if (remaining <= 0) finish();
          };
          utterance.onerror = () => {
            window.speechSynthesis.cancel();
            finish();
          };
          return utterance;
        });
        // Sentinel so stop()/stopSynth() knows synth playback is active.
        utteranceRef.current = utterances[utterances.length - 1] ?? null;
        for (const utterance of utterances) window.speechSynthesis.speak(utterance);
      }
    },
    [isTtsModel, stop, playBlob],
  );

  useEffect(() => {
    return () => {
      requestIdRef.current += 1;
      stopTts();
      stopSynth();
    };
  }, [stopTts, stopSynth]);

  return { isSpeaking, speak, stop, primeAudio };
}
