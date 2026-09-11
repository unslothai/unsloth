// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useCallback, useEffect, useRef, useState } from "react";
import { TTS_AUDIO_TYPES, VOICE_SLOT_AUDIO_TYPES } from "../voice/tts-audio-types";
import {
  splitIntoSentences,
  stripForSpeech,
} from "../voice/speech-text";

// Re-exported for importers that predate the split into a dependency-free module.
export { splitIntoSentences, stripForSpeech };

// The codec vocabulary lives in its own dependency-free module so the pure voice
// helpers and their node:test suites can read it without pulling in React and
// auth. Re-exported here for the importers that predate that split.
export { TTS_AUDIO_TYPES, VOICE_SLOT_AUDIO_TYPES };

// Subset of TTS_AUDIO_TYPES that are standalone TTS voices (Spark/bicodec,
// Dia/dac) rather than speech-LLMs (Orpheus/snac, Sesame CSM/csm) that speak
// with their own voice and don't need a separate TTS picker.
export const STANDALONE_TTS_AUDIO_TYPES = new Set(["bicodec", "dac"]);

// For streaming: while the LLM is still writing, only fully-terminated sentences
// are safe to synthesize; the trailing chunk is the sentence in progress.
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
// 0.35 was tuned on a box where the voice slot had the GPU to itself. Voice mode
// now shares it with the chat model and the STT sidecar, and that contention shows
// up as generation jitter rather than a slower average, so the old cushion drains
// mid-sentence and playback skips. Buying ~0.65s of first-audio latency here is
// worth it: a late first word is far less noticeable than a stuttering one.
const STREAM_PREROLL_S = 1.0;

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
  // One unwind callback per sentence currently streaming. A set rather than a
  // single slot because the lookahead keeps more than one in flight at a time.
  const streamUnwindRef = useRef<Set<() => void>>(new Set());
  // Absolute AudioContext time where the next PCM buffer should start. Shared
  // across sentences so they chain onto ONE continuous timeline: the jitter
  // buffer is paid once at the start of a reply, not again at every boundary.
  const streamPlayHeadRef = useRef(0);
  // Shared analyser for the orb level -- per-sentence ones would fight over
  // voiceOutputLevel as soon as two sentences overlap.
  const streamAnalyserRef = useRef<AnalyserNode | null>(null);
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
    streamPlayHeadRef.current = 0;
    const unwinds = [...streamUnwindRef.current];
    streamUnwindRef.current.clear();
    for (const unwind of unwinds) unwind();
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

  // Stream ONE sentence from /api/audio/speech/stream as 24 kHz int16 PCM, played
  // through Web Audio as it arrives.
  //
  // The fetch starts immediately, but nothing is scheduled until `gate` resolves
  // (the previous sentence finished scheduling), so the voice slot can be
  // generating sentence N+1 while N is still playing and the audio still comes out
  // in order. That overlap is the whole point: synthesize strictly one at a time
  // and every sentence boundary costs a full time-to-first-audio plus another
  // STREAM_PREROLL_S of jitter buffer, which is audible as a break.
  //
  // Resolves true once this sentence is fully SCHEDULED -- not when it finishes
  // playing; the shared playhead is what keeps the order. Resolves false without
  // scheduling anything if the stream isn't usable (non-SNAC voice / error), so the
  // caller can fall back to the blocking blob.
  const playSentenceStream = useCallback(
    (sentence: string, reqId: number, gate: Promise<unknown>): Promise<boolean> => {
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

        if (!streamAnalyserRef.current) {
          const node = ctx.createAnalyser();
          node.fftSize = 256;
          node.connect(ctx.destination);
          streamAnalyserRef.current = node;
        }
        const analyser = streamAnalyserRef.current;
        const levelBuf = new Float32Array(analyser.fftSize);
        const runLevel = () => {
          analyser.getFloatTimeDomainData(levelBuf);
          let s = 0;
          for (let i = 0; i < levelBuf.length; i++) s += levelBuf[i] * levelBuf[i];
          voiceOutputLevel.current = Math.sqrt(s / levelBuf.length);
          streamLevelRafRef.current = requestAnimationFrame(runLevel);
        };

        let settled = false;
        let scheduledAny = false;
        let leftover: Uint8Array | null = null;
        // Before the gate opens, chunks pile up here instead of being scheduled --
        // this sentence is generating ahead of its turn.
        let open = false;
        const queued: Uint8Array[] = [];

        const finish = (played: boolean) => {
          if (settled) return;
          settled = true;
          streamUnwindRef.current.delete(unwind);
          resolve(played);
        };
        // stop()/barge-in runs this (via stopStream) to unwind immediately.
        const unwind = () => finish(true);
        streamUnwindRef.current.add(unwind);

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
          // Chain off the SHARED playhead, so sentence N+1 lands flush against the
          // tail of N. When the playhead is in the past -- the start of a reply, or
          // after generation underran -- buy the jitter buffer back instead.
          const head = streamPlayHeadRef.current;
          const startAt =
            head > ctx.currentTime
              ? head
              : ctx.currentTime + (scheduledAny ? 0.005 : STREAM_PREROLL_S);
          src.start(startAt);
          streamPlayHeadRef.current = startAt + audioBuf.duration;
          scheduledAny = true;
          const wasIdle = streamSourcesRef.current.size === 0;
          streamSourcesRef.current.add(src);
          if (wasIdle && requestIdRef.current === reqId) setIsPlaying(true);
          if (!streamLevelRafRef.current)
            streamLevelRafRef.current = requestAnimationFrame(runLevel);
          src.onended = () => {
            streamSourcesRef.current.delete(src);
            // The timeline is only actually idle when no sentence has anything
            // left scheduled -- not merely when this one runs out.
            if (streamSourcesRef.current.size === 0) {
              if (streamLevelRafRef.current) {
                cancelAnimationFrame(streamLevelRafRef.current);
                streamLevelRafRef.current = 0;
              }
              voiceOutputLevel.current = 0;
              if (requestIdRef.current === reqId) setIsPlaying(false);
            }
          };
        };

        const openGate = () => {
          if (open || settled) return;
          if (requestIdRef.current !== reqId) return;
          open = true;
          for (const chunk of queued) schedule(chunk);
          queued.length = 0;
        };
        void gate.then(openGate, openGate);

        void (async () => {
          let resp: Response;
          try {
            resp = await authFetch("/api/inference/audio/speech/stream", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ input: sentence, voice }),
              signal: synthAbortRef.current?.signal,
            });
          } catch {
            // Aborted (stop / barge-in) or a network error. Still wait our turn
            // before resolving, for the same reason as the 400 below.
            await gate.catch(() => {});
            finish(scheduledAny);
            return;
          }
          if (!resp.ok || !resp.body) {
            // Wait our turn even though we have nothing to schedule. The caller
            // answers a false here by synthesizing and playing a blob on the
            // shared <audio> element, and a 400 comes back almost instantly -- so
            // resolving early would let every sentence in the window start its
            // fallback at once and stamp over each other's playback. This is the
            // path a Q2 Orpheus quant takes for EVERY sentence.
            await gate.catch(() => {});
            finish(false);
            return;
          }
          try {
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
              if (!value || !value.length) continue;
              if (open) schedule(value);
              else queued.push(value);
            }
          } catch {
            // aborted (stop / barge-in) or a network error mid-stream
          }
          // Generation finished ahead of our turn: wait for it, flush, and only
          // then resolve -- the next sentence gates on this, so it starts
          // scheduling the moment we are done.
          await gate.catch(() => {});
          openGate();
          finish(true);
        })();
      });
    },
    [],
  );

  // Wait for the scheduled PCM timeline to actually run out. The sentence
  // promises resolve once everything is SCHEDULED, so the end of a reply is a
  // second or more ahead of them; without this the loop would re-arm the mic over
  // the tail of its own last sentence.
  const drainStream = useCallback(async (reqId: number): Promise<void> => {
    const ctx = getPlayCtx();
    for (;;) {
      if (requestIdRef.current !== reqId) return;
      if (streamSourcesRef.current.size === 0) return;
      // Bound the wait by the timeline itself: a suspended context or a source
      // that never fires onended must not strand the loop with the mic shut.
      if (ctx && ctx.currentTime > streamPlayHeadRef.current + 0.5) return;
      await new Promise<void>((r) => setTimeout(r, 50));
    }
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
        authFetch("/api/inference/audio/speech", {
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
          // ~1s). Up to voiceParallelN generate at once, gated so they still
          // schedule in order -- the next sentence is already buffered when the
          // current one runs out, so the boundary has no synth gap in it.
          const N = Math.min(
            sentences.length,
            Math.max(1, useChatRuntimeStore.getState().voiceParallelN),
          );
          let gate: Promise<unknown> = Promise.resolve();
          const inflight: Array<Promise<void>> = [];
          for (let i = 0; i < sentences.length; i++) {
            if (requestIdRef.current !== reqId) return;
            const sentence = sentences[i] ?? "";
            const job = playSentenceStream(sentence, reqId, gate).then(
              async (played) => {
                if (played || requestIdRef.current !== reqId) return;
                // Stream not usable (non-SNAC voice) -> blocking blob fallback.
                const blob = await requestSpeechBlob(sentence);
                if (requestIdRef.current !== reqId) return;
                if (blob) await playBlob(blob, reqId);
              },
            );
            gate = job;
            inflight.push(job);
            while (inflight.length >= N) {
              const head = inflight.shift();
              if (head) await head;
            }
          }
          for (const job of inflight) await job;
          if (requestIdRef.current !== reqId) return;
          await drainStream(reqId);
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
    [
      isTtsModel,
      streamMode,
      stop,
      playBlob,
      playSentenceStream,
      requestSpeechBlob,
      drainStream,
    ],
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
      // Stream mode keeps up to voiceParallelN sentences generating at once. They
      // are chained on `gate` so they schedule in order however they finish, and
      // `inflight` bounds how far ahead the voice slot may run -- too far and a
      // barge-in throws away more audio than it saves.
      let gate: Promise<unknown> = Promise.resolve();
      const inflight: Array<Promise<void>> = [];
      const parallelN = () =>
        Math.max(1, useChatRuntimeStore.getState().voiceParallelN);
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
              const job = playSentenceStream(sentence, reqId, gate).then(
                async (played) => {
                  if (played || requestIdRef.current !== reqId) return;
                  const blob = await synthOne(sentence);
                  if (requestIdRef.current !== reqId) return;
                  if (blob) await playBlob(blob, reqId);
                },
              );
              gate = job;
              inflight.push(job);
              const limit = parallelN();
              while (inflight.length >= limit) {
                const head = inflight.shift();
                if (head) await head;
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
      for (const job of inflight) await job;
      if (requestIdRef.current !== reqId) return;
      // Everything is scheduled, but the tail is still seconds from playing out.
      if (streamMode) await drainStream(reqId);
      if (requestIdRef.current !== reqId) return;
      setIsSpeaking(false);
      streamRef.current = null;
      onPlaybackEndRef.current?.();
    })();
  }, [
    stop,
    isTtsModel,
    streamMode,
    playBlob,
    pumpSynth,
    playSentenceStream,
    synthOne,
    drainStream,
  ]);

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
