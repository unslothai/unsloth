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
  // Token-by-token streaming TTS playback resources (Web Audio).
  const streamAbortRef = useRef<AbortController | null>(null);
  const streamCtxRef = useRef<AudioContext | null>(null);
  const streamSourcesRef = useRef<AudioBufferSourceNode[]>([]);
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

  const stopStreaming = useCallback(() => {
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    for (const src of streamSourcesRef.current) {
      try {
        src.onended = null;
        src.stop();
        src.disconnect();
      } catch {
        /* already stopped */
      }
    }
    streamSourcesRef.current = [];
    const ctx = streamCtxRef.current;
    streamCtxRef.current = null;
    if (ctx) void ctx.close().catch(() => {});
  }, []);

  const stop = useCallback(() => {
    requestIdRef.current += 1;
    stopTts();
    stopSynth();
    stopStreaming();
    setIsSpeaking(false);
  }, [stopTts, stopSynth, stopStreaming]);

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

  // Token-by-token streaming: POST the full text and play PCM16 chunks as they
  // arrive, scheduled back-to-back via Web Audio for gapless output. Returns:
  //   "played"     – streamed and finished playing (or nothing to play)
  //   "fallback"   – streaming unavailable (e.g. non-SNAC 400); use chunking
  //   "superseded" – a stop()/new speak() took over mid-stream
  const speakStreaming = useCallback(
    async (
      text: string,
      reqId: number,
    ): Promise<"played" | "fallback" | "superseded"> => {
      const AudioCtx =
        typeof window !== "undefined"
          ? window.AudioContext ??
            (window as unknown as { webkitAudioContext?: typeof AudioContext })
              .webkitAudioContext
          : undefined;
      if (!AudioCtx) return "fallback";

      const controller = new AbortController();
      streamAbortRef.current = controller;
      let ctx: AudioContext | null = null;
      let scheduledAny = false;

      try {
        const response = await authFetch("/api/audio/speech/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input: text, voice: "default" }),
          signal: controller.signal,
        });
        if (requestIdRef.current !== reqId) return "superseded";
        // 400 means the loaded voice isn't SNAC; use the chunked endpoint.
        if (!response.ok || !response.body) return "fallback";

        const sampleRate = Number(response.headers.get("X-Sample-Rate")) || 24000;
        ctx = new AudioCtx();
        streamCtxRef.current = ctx;
        // A fresh context may start suspended under autoplay policy; resume so
        // scheduled buffers actually play (voice mode was armed by a gesture).
        if (ctx.state === "suspended") void ctx.resume().catch(() => {});
        let playHead = ctx.currentTime;
        let leftover = new Uint8Array(0);

        const scheduleBytes = (bytes: Uint8Array) => {
          let buf = bytes;
          if (leftover.length) {
            const merged = new Uint8Array(leftover.length + bytes.length);
            merged.set(leftover, 0);
            merged.set(bytes, leftover.length);
            buf = merged;
            leftover = new Uint8Array(0);
          }
          const usable = buf.length - (buf.length % 2); // int16 alignment
          if (usable < buf.length) leftover = buf.slice(usable);
          if (usable === 0 || !ctx) return;
          const aligned = buf.slice(0, usable); // own buffer, offset 0
          const int16 = new Int16Array(aligned.buffer);
          const float = new Float32Array(int16.length);
          for (let i = 0; i < int16.length; i++) float[i] = int16[i] / 32768;
          const audioBuf = ctx.createBuffer(1, float.length, sampleRate);
          audioBuf.getChannelData(0).set(float);
          const src = ctx.createBufferSource();
          src.buffer = audioBuf;
          src.connect(ctx.destination);
          const startAt = Math.max(playHead, ctx.currentTime);
          src.start(startAt);
          playHead = startAt + audioBuf.duration;
          src.onended = () => src.disconnect();
          streamSourcesRef.current.push(src);
          scheduledAny = true;
        };

        const reader = response.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (requestIdRef.current !== reqId) return "superseded";
          if (done) break;
          if (value && value.length) scheduleBytes(value);
        }

        if (ctx) {
          const remainingMs = Math.max(0, (playHead - ctx.currentTime) * 1000);
          await new Promise<void>((r) => setTimeout(r, remainingMs + 80));
        }
        return "played";
      } catch {
        if (requestIdRef.current !== reqId) return "superseded";
        // Only fall back if nothing has played, else chunking would replay audio.
        return scheduledAny ? "played" : "fallback";
      }
    },
    [],
  );

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

        // Prefer token-by-token streaming (SNAC/Orpheus): audio starts before
        // the reply is fully synthesized and prosody flows across sentences.
        // Falls back to chunked synthesis for non-SNAC voices or if unavailable.
        const streamResult = await speakStreaming(text, reqId);
        if (requestIdRef.current !== reqId) return;
        if (streamResult === "superseded") return;
        if (streamResult === "played") {
          setIsSpeaking(false);
          onPlaybackEndRef.current?.();
          return;
        }
        // streamResult === "fallback": chunked synthesis below.

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
    [isTtsModel, stop, playBlob, speakStreaming],
  );

  useEffect(() => {
    return () => {
      requestIdRef.current += 1;
      stopTts();
      stopSynth();
      stopStreaming();
    };
  }, [stopTts, stopSynth, stopStreaming]);

  return { isSpeaking, speak, stop, primeAudio };
}
