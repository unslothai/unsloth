// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useCallback, useEffect, useRef, useState } from "react";

export const TTS_AUDIO_TYPES = new Set(["snac", "csm", "bicodec", "dac"]);

// Subset of TTS_AUDIO_TYPES that are standalone TTS voices (Spark/bicodec,
// Dia/dac) rather than speech-LLMs (Orpheus/snac, Sesame CSM/csm) that speak
// with their own voice and don't need a separate TTS picker.
export const STANDALONE_TTS_AUDIO_TYPES = new Set(["bicodec", "dac"]);

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

// For streaming: while the LLM is still writing, only fully-terminated sentences
// are safe to synthesize; the trailing chunk is the sentence in progress.
// Emoji / pictographs / symbol chars a TTS model can't pronounce -- it otherwise
// voices their raw codepoints as gibberish. Stripped for speech only; the on-
// screen chat text keeps them. Covers emoji, regional-indicator flags, keycaps,
// variation selectors and the zero-width joiner.
const SPEECH_STRIP_RE =
  /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{20E3}\u{FE00}-\u{FE0F}\u{200D}]/gu;

export function stripForSpeech(text: string): string {
  return text.replace(SPEECH_STRIP_RE, "").replace(/\s+/g, " ").trim();
}

export function splitStreaming(text: string): {
  complete: string[];
  partial: string;
} {
  const parts = splitIntoSentences(text);
  if (parts.length === 0) return { complete: [], partial: "" };
  // If the text already ends with terminal punctuation, everything is complete.
  if (/[.!?]["')\]]?\s*$/.test(text)) return { complete: parts, partial: "" };
  return { complete: parts.slice(0, -1), partial: parts[parts.length - 1] ?? "" };
}

// Live output loudness (RMS-ish, ~0..0.3) of the TTS clip currently playing,
// refreshed each animation frame from a decoded copy of the audio -- the live
// <audio> element is NEVER routed through Web Audio, so playback can't be
// silenced. The speaking orb reads this to move its bars with the voice. Plain
// mutable ref so it never triggers React re-renders; reset to 0 when playback
// ends or is cut off.
export const voiceOutputLevel = { current: 0 };

const ENVELOPE_BUCKET_S = 1 / 60;
let _analysisCtx: AudioContext | null = null;

// Decode a clip and return its RMS envelope (one value per ~1/60s bucket) so the
// speaking bars can be indexed by playback time. Analysis-only: the context is
// never connected to an output, so decoding here cannot affect what the user hears.
async function envelopeFromBlob(blob: Blob): Promise<Float32Array | null> {
  try {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctor) return null;
    if (!_analysisCtx) _analysisCtx = new Ctor();
    const audioBuffer = await _analysisCtx.decodeAudioData(await blob.arrayBuffer());
    const ch = audioBuffer.getChannelData(0);
    const per = Math.max(1, Math.floor(audioBuffer.sampleRate * ENVELOPE_BUCKET_S));
    const n = Math.max(1, Math.ceil(ch.length / per));
    const env = new Float32Array(n);
    for (let b = 0; b < n; b++) {
      const start = b * per;
      const end = Math.min(ch.length, start + per);
      let sum = 0;
      for (let i = start; i < end; i++) sum += ch[i] * ch[i];
      env[b] = Math.sqrt(sum / Math.max(1, end - start));
    }
    return env;
  } catch {
    return null;
  }
}

export function useTtsPlayer(
  audioType: string | null | undefined,
  onPlaybackEnd?: () => void,
  voiceSlotLoaded = false,
): {
  isSpeaking: boolean;
  /** True only while an audio clip is actually playing (not during synthesis). */
  isPlaying: boolean;
  speak(text: string): void;
  /** Streaming: start a session, feed growing text, then end. Synthesizes each
   *  complete sentence as it arrives so the first one plays fast. */
  beginStream(): void;
  feedText(text: string): void;
  endStream(finalText: string): void;
  stop(): void;
  primeAudio(): void;
} {
  const [isSpeaking, setIsSpeaking] = useState(false);
  // Distinct from isSpeaking: true only while a clip is audibly playing, so the
  // orb can show a separate "synthesizing" state during the (slow) synth gaps.
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  // Bumped on every stop()/new speak()/unmount so an in-flight /api/audio/speech
  // response (or a queued sentence) can detect it was superseded and skip late
  // playback + state updates.
  const requestIdRef = useRef(0);
  // Aborts in-flight /api/audio/speech synth fetches on stop()/barge-in, so the
  // backend voice slot stops grinding through stale sentences and is free to
  // synthesize the new turn immediately (llama-server cancels a slot when its
  // request connection closes).
  const synthAbortRef = useRef<AbortController | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  // Resolver for the sentence currently awaiting playback, so stop() can unwind
  // the speak loop immediately (pause() doesn't fire "ended").
  const playResolveRef = useRef<(() => void) | null>(null);
  const onPlaybackEndRef = useRef(onPlaybackEnd);
  onPlaybackEndRef.current = onPlaybackEnd;

  // Streaming session state. `sentences` holds every known sentence text (grows
  // as the reply streams); `jobs`/`launched` track synth jobs actually fired --
  // only a small lookahead window ahead of `playIndex` is launched, so the backend
  // never backs up with stale sentences and barge-in leaves almost nothing queued.
  const streamRef = useRef<{
    reqId: number;
    sentences: string[];
    jobs: Array<Promise<Blob | null>>;
    launched: number;
    playIndex: number;
    final: boolean;
  } | null>(null);

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
    // Cancel any in-flight synth so the voice slot stops on stale sentences and
    // is free for the next turn; then arm a fresh controller for that turn.
    synthAbortRef.current?.abort();
    synthAbortRef.current = new AbortController();
    stopTts();
    stopSynth();
    setIsSpeaking(false);
    setIsPlaying(false);
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

      // Drive the speaking-orb bars from this clip's loudness. Decode a copy for
      // its envelope (async; the live element is untouched) and, while it plays,
      // publish the level at the current playback time each animation frame.
      let env: Float32Array | null = null;
      void envelopeFromBlob(blob).then((e) => {
        env = e;
      });
      let levelRaf = 0;
      const runLevel = () => {
        if (env) {
          voiceOutputLevel.current =
            env[Math.floor(audio.currentTime / ENVELOPE_BUCKET_S)] ?? 0;
        }
        levelRaf = requestAnimationFrame(runLevel);
      };
      const stopLevel = () => {
        if (levelRaf) cancelAnimationFrame(levelRaf);
        levelRaf = 0;
        voiceOutputLevel.current = 0;
      };

      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        stopLevel();
        if (playResolveRef.current === done) playResolveRef.current = null;
        if (objectUrlRef.current === url) {
          URL.revokeObjectURL(url);
          objectUrlRef.current = null;
        }
        audio.onended = null;
        audio.onerror = null;
        audio.onplaying = null;
        // Between chunks we're synthesizing again, not playing.
        if (requestIdRef.current === reqId) setIsPlaying(false);
        resolve();
      };
      playResolveRef.current = done;
      audio.onended = done;
      audio.onerror = done;
      // Flip to "playing" only once audio actually starts, so the gap before it
      // (synthesis) stays in the synthesizing state.
      audio.onplaying = () => {
        if (requestIdRef.current === reqId) setIsPlaying(true);
        if (!levelRaf) levelRaf = requestAnimationFrame(runLevel);
      };
      audio.src = url;
      // play() returns a promise that can reject (autoplay policy, decode error)
      // without ever firing onerror; settle so the loop can't stall.
      void audio.play().catch(done);
    });
  }, []);

  // POST one sentence to /api/audio/speech and return the audio blob. If the
  // backend voice slot is gone (400 -- unloaded by a ChatPage remount, an auth
  // bounce, or a studio relaunch), reload it once via the store hook and retry,
  // so TTS heals itself instead of silently 400ing for the rest of the session.
  const requestSpeechBlob = useCallback(
    async (input: string): Promise<Blob | null> => {
      const voice =
        useChatRuntimeStore.getState().selectedVoiceName || "tara";
      const doFetch = () =>
        authFetch("/api/audio/speech", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input, voice }),
          signal: synthAbortRef.current?.signal,
        });
      try {
        let r = await doFetch();
        if (r.ok) return await r.blob();
        if (r.status === 400) {
          const reload = useChatRuntimeStore.getState().ensureVoiceSlotLoaded;
          if (reload && (await reload())) {
            r = await doFetch();
            if (r.ok) return await r.blob();
          }
        }
        return null;
      } catch {
        return null;
      }
    },
    [],
  );

  const speak = useCallback(
    async (text: string) => {
      stop();
      text = stripForSpeech(text);
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
          requestSpeechBlob(sentence);

        // Bounded-concurrency pipeline: keep up to N sentences synthesizing at
        // once (voiceParallelN), play them back strictly in order. N=1 is the old
        // one-ahead behavior. The GGUF voice slot must be loaded with matching
        // --parallel N; the backend serializes only the shared codec decode.
        const N = Math.min(
          sentences.length,
          Math.max(1, useChatRuntimeStore.getState().voiceParallelN),
        );
        const jobs: Array<Promise<Blob | null>> = [];
        let launched = 0;
        const launchUpTo = (limit: number) => {
          while (launched < sentences.length && launched < limit) {
            jobs[launched] = synth(sentences[launched]);
            launched++;
          }
        };
        launchUpTo(N); // prime N in flight
        for (let i = 0; i < sentences.length; i++) {
          if (requestIdRef.current !== reqId) return;  // superseded
          const blob = await jobs[i];
          if (requestIdRef.current !== reqId) return;
          // Refill the window so N stay in flight ahead of playback.
          launchUpTo(i + 1 + N);
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
          setIsPlaying(false);
          onPlaybackEndRef.current?.();
        };
        const utterances = sentences.map((sentence) => {
          const utterance = new SpeechSynthesisUtterance(sentence);
          utterance.onstart = () => setIsPlaying(true);
          utterance.onend = () => {
            setIsPlaying(false);
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
    [isTtsModel, stop, playBlob, requestSpeechBlob],
  );

  // ── Streaming TTS ───────────────────────────────────────────────
  // POST one sentence to /api/audio/speech; null on failure.
  const synthOne = useCallback(
    (sentence: string): Promise<Blob | null> => {
      // Strip emoji here -- the single synth chokepoint for streaming, hit by both
      // feedText and the endStream flush -- so no path can send unpronounceable
      // glyphs. An emoji-only chunk has nothing to say, so skip it.
      const clean = stripForSpeech(sentence);
      if (!clean) return Promise.resolve(null);
      return requestSpeechBlob(clean);
    },
    [requestSpeechBlob],
  );

  // Launch synth jobs only up to a small lookahead window ahead of the sentence
  // currently playing -- "always one (or voiceParallelN) ahead", never the whole
  // reply. This keeps audio close to real time and means a barge-in leaves at most
  // a couple of stale sentences on the backend instead of a long tail.
  const pumpSynth = useCallback(() => {
    const st = streamRef.current;
    if (!st || requestIdRef.current !== st.reqId || !isTtsModel) return;
    // Keep at most ONE sentence synthesizing ahead of the one playing. The voice
    // slot is compute-bound on a single GPU even at --parallel N, so firing
    // further ahead just backs up the queue and makes a barge-in throw away more
    // in-flight audio; one ahead is enough to hide the gap between sentences.
    const lookahead = 1;
    const limit = Math.min(st.sentences.length, st.playIndex + 1 + lookahead);
    while (st.launched < limit) {
      st.jobs[st.launched] = synthOne(st.sentences[st.launched] ?? "");
      st.launched++;
    }
  }, [isTtsModel, synthOne]);

  // Start a streaming session. Sentences fed via feedText are synthesized within a
  // bounded lookahead window and played strictly in order, so the first sentence
  // plays without waiting for the whole reply. Browser voice has no server synth,
  // so it just waits for endStream and speaks the whole thing.
  const beginStream = useCallback(() => {
    stop();
    const reqId = requestIdRef.current;
    streamRef.current = {
      reqId,
      sentences: [],
      jobs: [],
      launched: 0,
      playIndex: 0,
      final: false,
    };
    if (!isTtsModel) return;
    setIsSpeaking(true);
    void (async () => {
      while (true) {
        if (requestIdRef.current !== reqId) return;
        const st = streamRef.current;
        if (!st || st.reqId !== reqId) return;
        if (st.playIndex < st.sentences.length) {
          pumpSynth(); // ensure the current sentence (and window) is launched
          const blob = await st.jobs[st.playIndex];
          if (requestIdRef.current !== reqId) return;
          st.playIndex++;
          pumpSynth(); // playback advanced -> refill the lookahead window
          if (blob) await playBlob(blob, reqId);
        } else if (st.final) {
          break;
        } else {
          await new Promise<void>((r) => setTimeout(r, 40));
        }
      }
      if (requestIdRef.current !== reqId) return;
      setIsSpeaking(false);
      streamRef.current = null;
      onPlaybackEndRef.current?.();
    })();
  }, [stop, isTtsModel, playBlob, pumpSynth]);

  // Feed the growing assistant text; records newly-complete sentences and lets the
  // pump launch synth for them within the lookahead window (not all at once).
  const feedText = useCallback(
    (text: string) => {
      const s = streamRef.current;
      if (!s || requestIdRef.current !== s.reqId || !isTtsModel) return;
      const { complete } = splitStreaming(text);
      while (s.sentences.length < complete.length) {
        s.sentences.push(complete[s.sentences.length] ?? "");
      }
      pumpSynth();
    },
    [isTtsModel, pumpSynth],
  );

  // Finish the session: record the final (incl. trailing) sentences, mark done.
  const endStream = useCallback(
    (finalText: string) => {
      const s = streamRef.current;
      if (!s || requestIdRef.current !== s.reqId) return;
      if (!isTtsModel) {
        // Browser voice: nothing streamed; speak the whole reply now.
        streamRef.current = null;
        speak(finalText);
        return;
      }
      const all = splitIntoSentences(finalText);
      while (s.sentences.length < all.length) {
        s.sentences.push(all[s.sentences.length] ?? "");
      }
      s.final = true;
      pumpSynth();
      if (s.sentences.length === 0) {
        streamRef.current = null;
        setIsSpeaking(false);
        onPlaybackEndRef.current?.();
      }
    },
    [isTtsModel, pumpSynth, speak],
  );

  useEffect(() => {
    return () => {
      requestIdRef.current += 1;
      synthAbortRef.current?.abort();
      stopTts();
      stopSynth();
    };
  }, [stopTts, stopSynth]);

  return { isSpeaking, isPlaying, speak, beginStream, feedText, endStream, stop, primeAudio };
}
