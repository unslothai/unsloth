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

// Normalize text for TTS: some characters derail Orpheus (like the colon it reads
// as a speaker tag) or get voiced literally by any TTS model. Em/en dashes and the
// single-char ellipsis are the main offenders (an em dash breaks the voice), and
// markdown/markup symbols get read out ("asterisk asterisk"). All of these are just
// dropped (replaced with a space, not a comma, so no phantom pauses are inserted);
// smart quotes are normalized. Speech ONLY -- the on-screen chat text keeps everything.
export function stripForSpeech(text: string): string {
  return text
    .replace(SPEECH_STRIP_RE, "")
    .replace(/\s*[—–―‒−]\s*/g, " ") // — – ― ‒ −  -> drop
    .replace(/\s*…\s*/g, " ") //                           …  -> drop
    .replace(/\.{2,}/g, " ") //                                 ... -> drop
    .replace(/[‐‑]/g, "-") //                          unicode hyphens -> ASCII
    .replace(/[*_`~^|#<>\\{}[\]]/g, " ") //                      markdown / markup -> space
    .replace(/[‘’‚‛]/g, "'") //             smart single quotes
    .replace(/[“”„‟]/g, '"') //             smart double quotes
    .replace(/\s+/g, " ")
    .trim();
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

// Shared AudioContext for STREAMING PCM playback. Unlike the analysis-only context
// above, this one IS connected to the speakers: incoming 24 kHz int16 PCM chunks are
// scheduled back-to-back on it so audio starts on the first chunk (~1s) instead of
// waiting for the whole clip. Created lazily; resumed on a user gesture (primeAudio).
let _playCtx: AudioContext | null = null;
function getPlayCtx(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const Ctor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;
  if (!_playCtx) _playCtx = new Ctor();
  return _playCtx;
}

function streamPlaybackSupported(): boolean {
  return getPlayCtx() !== null && typeof ReadableStream !== "undefined";
}

// Jitter buffer: Orpheus generates at ~real time (RTF ~1), so streamed chunks
// barely keep pace with playback. Starting the first buffer this far in the future
// keeps the scheduler ~this many seconds ahead of generation, so a late chunk
// (GPU jitter) doesn't starve playback into a gap/click. Costs ~this much extra
// first-audio latency, still far below waiting for the whole clip.
const STREAM_PREROLL_S = 0.35;

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
  // Streaming-PCM playback state: the Web Audio sources currently scheduled (so
  // stop()/barge-in can cut them), the orb-level rAF handle, and a resolver so
  // stop() can immediately unwind a sentence that's mid-stream.
  const streamSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const streamLevelRafRef = useRef(0);
  const streamResolveRef = useRef<(() => void) | null>(null);
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
  // Stream PCM straight from /api/audio/speech/stream (SNAC/Orpheus voice slot) and
  // play it as it arrives, so first audio lands ~1s in instead of after the whole
  // clip. Only for the loaded voice slot (the streaming endpoint is SNAC-only); if
  // the stream 400s (non-SNAC), playSentenceStream falls back to the blocking blob.
  const streamMode = voiceSlotLoaded && streamPlaybackSupported();

  // Unlock Safari's audio autoplay policy by calling play()+pause() during
  // a synchronous user gesture. The element is reused for all subsequent plays
  // so the unlock survives async fetch callbacks. Also resume the streaming
  // AudioContext in the same gesture so scheduled PCM isn't blocked by autoplay.
  const primeAudio = useCallback(() => {
    if (typeof window === "undefined") return;
    if (!audioRef.current) {
      audioRef.current = new Audio();
    }
    audioRef.current.play().then(() => audioRef.current?.pause()).catch(() => {});
    void getPlayCtx()?.resume().catch(() => {});
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

  // Cut streaming-PCM playback: stop every scheduled buffer source, drop the orb
  // level loop, and resolve the sentence that's mid-stream so its awaiter unwinds.
  const stopStream = useCallback(() => {
    for (const src of streamSourcesRef.current) {
      try {
        src.onended = null;
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    streamSourcesRef.current.clear();
    if (streamLevelRafRef.current) cancelAnimationFrame(streamLevelRafRef.current);
    streamLevelRafRef.current = 0;
    voiceOutputLevel.current = 0;
    const resolve = streamResolveRef.current;
    streamResolveRef.current = null;
    resolve?.();
  }, []);

  const stop = useCallback(() => {
    requestIdRef.current += 1;
    // Cancel any in-flight synth so the voice slot stops on stale sentences and
    // is free for the next turn; then arm a fresh controller for that turn.
    synthAbortRef.current?.abort();
    synthAbortRef.current = new AbortController();
    stopTts();
    stopStream();
    stopSynth();
    setIsSpeaking(false);
    setIsPlaying(false);
  }, [stopTts, stopStream, stopSynth]);

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

  // Stream ONE sentence from /api/audio/speech/stream and play its 24 kHz int16 PCM
  // as it arrives (Web Audio, buffer sources scheduled back-to-back), so audio
  // starts on the first chunk (~1s) instead of after the whole clip. Resolves true
  // once playback finished or was cut; resolves FALSE without playing if the stream
  // isn't usable (non-SNAC voice / error) so the caller can fall back to the blob.
  const playSentenceStream = useCallback(
    (sentence: string, reqId: number): Promise<boolean> => {
      return new Promise<boolean>((resolve) => {
        if (requestIdRef.current !== reqId) {
          resolve(true);
          return;
        }
        const ctx = getPlayCtx();
        if (!ctx) {
          resolve(false);
          return;
        }
        void ctx.resume().catch(() => {});
        const voice = useChatRuntimeStore.getState().selectedVoiceName || "tara";

        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.connect(ctx.destination);
        const levelBuf = new Float32Array(analyser.fftSize);
        const runLevel = () => {
          analyser.getFloatTimeDomainData(levelBuf);
          let s = 0;
          for (let i = 0; i < levelBuf.length; i++) s += levelBuf[i] * levelBuf[i];
          voiceOutputLevel.current = Math.sqrt(s / levelBuf.length);
          streamLevelRafRef.current = requestAnimationFrame(runLevel);
        };

        let playHead = 0;
        let started = false;
        let streamEnded = false;
        let pending = 0;
        let settled = false;
        let leftover: Uint8Array | null = null;

        const finish = (played: boolean) => {
          if (settled) return;
          settled = true;
          streamResolveRef.current = null;
          try {
            analyser.disconnect();
          } catch {
            /* already gone */
          }
          if (streamLevelRafRef.current && streamSourcesRef.current.size === 0) {
            cancelAnimationFrame(streamLevelRafRef.current);
            streamLevelRafRef.current = 0;
            voiceOutputLevel.current = 0;
          }
          if (requestIdRef.current === reqId) setIsPlaying(false);
          resolve(played);
        };
        // stop()/barge-in calls this (via stopStream) to unwind immediately.
        streamResolveRef.current = () => finish(true);

        const schedule = (bytes: Uint8Array) => {
          let data = bytes;
          if (leftover && leftover.length) {
            const merged = new Uint8Array(leftover.length + bytes.length);
            merged.set(leftover);
            merged.set(bytes, leftover.length);
            data = merged;
            leftover = null;
          }
          const nSamples = data.length >> 1;
          if (nSamples === 0) {
            leftover = data;
            return;
          }
          const usable = nSamples * 2;
          if (usable < data.length) leftover = data.slice(usable);
          const view = new DataView(data.buffer, data.byteOffset, usable);
          const f32 = new Float32Array(nSamples);
          for (let i = 0; i < nSamples; i++) f32[i] = view.getInt16(i * 2, true) / 32768;
          const audioBuf = ctx.createBuffer(1, nSamples, 24000);
          audioBuf.copyToChannel(f32, 0);
          const src = ctx.createBufferSource();
          src.buffer = audioBuf;
          src.connect(analyser);
          // First buffer starts a jitter-buffer ahead (STREAM_PREROLL_S) so the
          // scheduler stays ahead of the ~real-time generation; the rest chain
          // gaplessly off the running playhead. The max() guard means a chunk that
          // arrived late still schedules just ahead of now instead of in the past.
          const startAt = !started
            ? ctx.currentTime + STREAM_PREROLL_S
            : Math.max(ctx.currentTime + 0.005, playHead);
          src.start(startAt);
          playHead = startAt + audioBuf.duration;
          streamSourcesRef.current.add(src);
          pending++;
          if (!started) {
            started = true;
            if (requestIdRef.current === reqId) setIsPlaying(true);
            if (!streamLevelRafRef.current)
              streamLevelRafRef.current = requestAnimationFrame(runLevel);
          }
          src.onended = () => {
            streamSourcesRef.current.delete(src);
            pending--;
            if (streamEnded && pending <= 0) finish(true);
          };
        };

        void (async () => {
          try {
            const resp = await authFetch("/api/audio/speech/stream", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ input: sentence, voice }),
              signal: synthAbortRef.current?.signal,
            });
            if (!resp.ok || !resp.body) {
              finish(false); // let the caller fall back to the blocking blob path
              return;
            }
            const reader = resp.body.getReader();
            for (;;) {
              const { done: rdone, value } = await reader.read();
              if (rdone) break;
              if (requestIdRef.current !== reqId) {
                try {
                  await reader.cancel();
                } catch {
                  /* ignore */
                }
                break;
              }
              if (value && value.length) schedule(value);
            }
          } catch {
            // aborted (stop / barge-in) or a network error
          } finally {
            streamEnded = true;
            if (pending <= 0) finish(true);
          }
        })();
      });
    },
    [],
  );

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

        if (streamMode) {
          // Stream each sentence's PCM and play it as it generates (first audio in
          // ~1s). Sequential per sentence: the single voice slot generates one at a
          // time, so there's nothing to pre-synthesize ahead.
          for (let i = 0; i < sentences.length; i++) {
            if (requestIdRef.current !== reqId) return;
            const played = await playSentenceStream(sentences[i], reqId);
            if (requestIdRef.current !== reqId) return;
            if (!played) {
              // Stream not usable (non-SNAC voice) -> blocking blob fallback.
              const blob = await requestSpeechBlob(sentences[i]);
              if (requestIdRef.current !== reqId) return;
              if (blob) await playBlob(blob, reqId);
            }
          }
        } else {
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
    [isTtsModel, streamMode, stop, playBlob, playSentenceStream, requestSpeechBlob],
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
    // In stream mode the loop pulls each sentence straight off st.sentences and
    // streams it, so there are no blob jobs to pre-launch.
    if (!st || requestIdRef.current !== st.reqId || !isTtsModel || streamMode) return;
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
  }, [isTtsModel, synthOne, streamMode]);

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
          if (streamMode) {
            // Stream this sentence's PCM and play as it generates (fast first audio),
            // keeping the per-sentence order. Fall back to a blob if the stream 400s.
            const sentence = stripForSpeech(st.sentences[st.playIndex] ?? "");
            st.playIndex++;
            if (sentence) {
              const played = await playSentenceStream(sentence, reqId);
              if (requestIdRef.current !== reqId) return;
              if (!played) {
                const blob = await synthOne(sentence);
                if (requestIdRef.current !== reqId) return;
                if (blob) await playBlob(blob, reqId);
              }
            }
          } else {
            pumpSynth(); // ensure the current sentence (and window) is launched
            const blob = await st.jobs[st.playIndex];
            if (requestIdRef.current !== reqId) return;
            st.playIndex++;
            pumpSynth(); // playback advanced -> refill the lookahead window
            if (blob) await playBlob(blob, reqId);
          }
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
  }, [stop, isTtsModel, streamMode, playBlob, pumpSynth, playSentenceStream, synthOne]);

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
      stopStream();
      stopSynth();
    };
  }, [stopTts, stopStream, stopSynth]);

  return { isSpeaking, isPlaying, speak, beginStream, feedText, endStream, stop, primeAudio };
}
