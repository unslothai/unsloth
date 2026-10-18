// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { generateStudioTtsAudio } from "@/features/chat/adapters/studio-speech-synthesis-adapter";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import { splitIntoSentences, stripForSpeech } from "../voice/speech-text";
import { planTtsChunk } from "../voice/tts-chunk-plan";
import { getVoicePlayContext } from "../voice/voice-loop-bridge";
import { outputSpectrumNode } from "../voice/voice-spectrum";
import { voiceTrace } from "../voice/voice-trace";
import { voiceTtsCanClone } from "../voice/voice-models";

// While the LLM is still writing, only terminated sentences are safe to speak;
// the trailing chunk is the sentence in progress.
export function splitStreaming(text: string): {
  complete: string[];
  partial: string;
} {
  const parts = splitIntoSentences(text);
  if (parts.length === 0) return { complete: [], partial: "" };
  if (/[.!?]["')\]]?\s*$/.test(text)) return { complete: parts, partial: "" };
  return { complete: parts.slice(0, -1), partial: parts[parts.length - 1] ?? "" };
}

// Loudness of what is playing (RMS, ~0..0.3), read by the speaking orb each frame.
export const voiceOutputLevel = { current: 0 };

// audio.cpp serializes requests per model, so synthesizing further ahead only
// queues work a barge-in throws away. One clip queued behind the playing one hides
// the gap between sentences.
const MAX_QUEUED_CLIPS = 2;
const ENVELOPE_BUCKET_S = 1 / 60;

const getPlayCtx = getVoicePlayContext;

function wavBytesFromDataUrl(url: string): ArrayBuffer {
  const binary = atob(url.slice(url.indexOf(",") + 1));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function envelopeOf(buffer: AudioBuffer): Float32Array {
  const ch = buffer.getChannelData(0);
  const per = Math.max(1, Math.floor(buffer.sampleRate * ENVELOPE_BUCKET_S));
  const env = new Float32Array(Math.max(1, Math.ceil(ch.length / per)));
  for (let b = 0; b < env.length; b++) {
    const start = b * per;
    const end = Math.min(ch.length, start + per);
    let sum = 0;
    for (let i = start; i < end; i++) sum += ch[i] * ch[i];
    env[b] = Math.sqrt(sum / Math.max(1, end - start));
  }
  return env;
}

interface Clip {
  source: AudioBufferSourceNode;
  start: number;
  env: Float32Array;
}

interface Stream {
  reqId: number;
  sentences: string[];
  next: number;
  final: boolean;
  /** Sentences spoken that are not part of the reply text (thinking fillers). */
  extra: number;
}

/**
 * Speaks a reply through voice mode's audio.cpp voice while the LLM is still
 * writing it. The first sentence is one /audio/generate call; later sentences are
 * merged per planTtsChunk. Clips play in order on one AudioContext playhead.
 */
export function useTtsPlayer(onPlaybackEnd?: () => void): {
  isSpeaking: boolean;
  /** True only while a clip is audible (not while the first one synthesizes). */
  isPlaying: boolean;
  speak(text: string): void;
  beginStream(): void;
  feedText(text: string): void;
  endStream(finalText: string): void;
  /** Queue a sentence that is not in the reply text, e.g. a thinking filler. */
  sayAside(text: string): void;
  warmAsides(texts: readonly string[]): Promise<void>;
  stop(): void;
  /** Hold the reply while a possible barge-in is checked; resumePlayback picks it back up. */
  pausePlayback(): void;
  resumePlayback(): void;
  primeAudio(): void;
} {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const reqRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const clipsRef = useRef(new Set<Clip>());
  const playHeadRef = useRef(0);
  const streamRef = useRef<Stream | null>(null);
  const asideCacheRef = useRef(new Map<string, AudioBuffer>());
  // A reply sentence is at the synthesizer: an aside now would land right before it.
  const realSynthRef = useRef(false);
  const failedReqRef = useRef(-1);
  const rafRef = useRef<number | null>(null);
  const pausedRef = useRef(false);
  const onEndRef = useRef(onPlaybackEnd);
  onEndRef.current = onPlaybackEnd;

  const primeAudio = useCallback(() => {
    void getPlayCtx()?.resume().catch(() => {});
  }, []);

  const meter = useCallback(() => {
    const ctx = getPlayCtx();
    rafRef.current = null;
    if (!ctx || clipsRef.current.size === 0) {
      voiceOutputLevel.current = 0;
      return;
    }
    let level = 0;
    for (const clip of clipsRef.current) {
      const i = Math.floor((ctx.currentTime - clip.start) / ENVELOPE_BUCKET_S);
      if (i >= 0 && i < clip.env.length) level = clip.env[i] ?? 0;
    }
    voiceOutputLevel.current = level;
    rafRef.current = requestAnimationFrame(meter);
  }, []);

  const stop = useCallback(() => {
    voiceTrace("tts.stop", { clips: clipsRef.current.size, stream: Boolean(streamRef.current) });
    reqRef.current += 1;
    if (pausedRef.current) {
      pausedRef.current = false;
      void getPlayCtx()?.resume().catch(() => {});
    }
    abortRef.current?.abort();
    abortRef.current = null;
    for (const clip of clipsRef.current) {
      clip.source.onended = null;
      try {
        clip.source.stop();
      } catch {
        // never started
      }
    }
    clipsRef.current.clear();
    playHeadRef.current = 0;
    streamRef.current = null;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    voiceOutputLevel.current = 0;
    setIsSpeaking(false);
    setIsPlaying(false);
  }, []);

  const synthesize = useCallback(
    async (sentence: string, reqId: number): Promise<AudioBuffer | null> => {
      const ctx = getPlayCtx();
      if (!ctx) return null;
      const request = () => {
        const { selectedVoiceModelId, selectedVoiceId, selectedVoiceSpeaker } =
          useChatRuntimeStore.getState();
        return generateStudioTtsAudio(sentence, abortRef.current?.signal, {
          model: selectedVoiceModelId,
          voiceId: voiceTtsCanClone(selectedVoiceModelId) ? selectedVoiceId : "",
          speaker: selectedVoiceSpeaker,
        });
      };
      let url: string;
      try {
        url = await request();
      } catch (error) {
        if (reqRef.current !== reqId) return null;
        // The voice may have been evicted (a bigger load, the backend restarting):
        // load it again once rather than going quiet for the rest of the session.
        const ensure = useChatRuntimeStore.getState().ensureVoiceSlotLoaded;
        if (!ensure || !(await ensure())) throw error;
        if (reqRef.current !== reqId) return null;
        url = await request();
      }
      if (reqRef.current !== reqId) return null;
      const bytes = wavBytesFromDataUrl(url);
      try {
        return await ctx.decodeAudioData(bytes.slice(0));
      } catch (error) {
        // A clip the decoder refused once has decoded on the second try; one retry
        // before the sentence is given up as unspeakable.
        voiceTrace("tts.decodeRetry", { bytes: bytes.byteLength });
        return await ctx.decodeAudioData(bytes.slice(0));
      }
    },
    [],
  );

  const schedule = useCallback(
    (buffer: AudioBuffer) => {
      const ctx = getPlayCtx();
      if (!ctx) return;
      // WebKit parks the context ("interrupted"/"suspended") when another surface takes
      // the audio session, e.g. the Audio tab mounting its players; a start() on a parked
      // context is silent, so wake it first.
      if (ctx.state !== "running" && !pausedRef.current) void ctx.resume().catch(() => {});
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(outputSpectrumNode(ctx));
      const start = Math.max(ctx.currentTime + 0.02, playHeadRef.current);
      const clip: Clip = { source, start, env: envelopeOf(buffer) };
      source.onended = () => {
        clipsRef.current.delete(clip);
        if (clipsRef.current.size === 0) {
          voiceTrace("tts.lastClipEnded", { stream: Boolean(streamRef.current) });
          setIsPlaying(false);
        }
      };
      clipsRef.current.add(clip);
      source.start(start);
      playHeadRef.current = start + buffer.duration;
      setIsPlaying(true);
      if (rafRef.current === null) rafRef.current = requestAnimationFrame(meter);
    },
    [meter],
  );

  const run = useCallback(
    async (reqId: number) => {
      const live = () => reqRef.current === reqId;
      const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
      let first = true;
      let waitSince: number | null = null;
      for (;;) {
        const st = streamRef.current;
        if (!live() || !st) return;
        if (st.next < st.sentences.length) {
          if (clipsRef.current.size >= MAX_QUEUED_CLIPS) {
            await sleep(40);
            continue;
          }
          const pending = st.sentences.slice(st.next).map(stripForSpeech);
          const take = planTtsChunk({
            pending,
            first,
            final: st.final,
            playing: clipsRef.current.size > 0,
            waitedMs: waitSince === null ? 0 : performance.now() - waitSince,
          });
          if (take === 0) {
            waitSince ??= performance.now();
            await sleep(40);
            continue;
          }
          waitSince = null;
          const i = st.next;
          st.next += take;
          const text = pending.slice(0, take).filter(Boolean).join(" ");
          if (!text) continue;
          first = false;
          try {
            voiceTrace("tts.sentence", { i, sentences: take, chars: text.length, text });
            realSynthRef.current = true;
            let buffer: AudioBuffer | null;
            try {
              buffer = await synthesize(text, reqId);
            } finally {
              realSynthRef.current = false;
            }
            if (!live()) return;
            if (buffer) {
              voiceTrace("tts.clip", { i, sentences: take, seconds: Math.round(buffer.duration * 100) / 100 });
              schedule(buffer);
            }
          } catch (error) {
            if (!live()) return;
            voiceTrace("tts.error", {
              i,
              message: error instanceof Error ? error.message : String(error),
            });
            // One toast per reply: a voice that is down fails every sentence.
            if (failedReqRef.current !== reqId) {
              failedReqRef.current = reqId;
              toast.error("Could not synthesize speech", {
                description: error instanceof Error ? error.message : undefined,
              });
            }
          }
        } else if (st.final) {
          break;
        } else {
          await sleep(40);
        }
      }
      // Bounded by the timeline itself, so a source that never fires onended
      // cannot keep the mic shut.
      const ctx = getPlayCtx();
      while (live() && clipsRef.current.size > 0) {
        if (ctx && ctx.currentTime > playHeadRef.current + 0.5) break;
        await sleep(50);
      }
      if (!live()) return;
      // onended fires when the graph is done, not the speaker: the last word is
      // still in the output buffer (a Bluetooth headset holds a few hundred ms),
      // and re-arming the mic that instant reconfigures the audio route under it.
      const latencyMs =
        ((ctx?.outputLatency ?? 0) + (ctx?.baseLatency ?? 0)) * 1000;
      voiceTrace("tts.streamDone", { latencyMs: Math.round(latencyMs) });
      await sleep(Math.max(250, latencyMs + 150));
      if (!live()) return;
      streamRef.current = null;
      setIsSpeaking(false);
      setIsPlaying(false);
      onEndRef.current?.();
    },
    [synthesize, schedule],
  );

  const beginStream = useCallback(() => {
    stop();
    const reqId = reqRef.current;
    abortRef.current = new AbortController();
    streamRef.current = { reqId, sentences: [], next: 0, final: false, extra: 0 };
    setIsSpeaking(true);
    void run(reqId);
  }, [stop, run]);

  const feedText = useCallback((text: string) => {
    const st = streamRef.current;
    if (!st || st.reqId !== reqRef.current) return;
    const { complete } = splitStreaming(text);
    while (st.sentences.length - st.extra < complete.length) {
      st.sentences.push(complete[st.sentences.length - st.extra] ?? "");
    }
  }, []);

  // Asides ("one sec", "almost there") used to join the sentence queue, so they
  // were synthesized in turn and came out seconds after the moment they were for,
  // often right before the answer itself. Now one plays only when nothing is on
  // the timeline and no reply sentence is being synthesized, straight from a cache
  // when it has been said before; a miss is dropped if the reply has caught up by
  // the time it is ready.
  const sayAside = useCallback(
    (text: string) => {
      const st = streamRef.current;
      if (!st || st.reqId !== reqRef.current || st.final) return;
      const reqId = reqRef.current;
      const blocked = () => clipsRef.current.size > 0 || realSynthRef.current;
      const play = (buffer: AudioBuffer, cached: boolean) => {
        if (reqRef.current !== reqId) return;
        if (blocked()) {
          voiceTrace("tts.asideDropped", { text, cached, clips: clipsRef.current.size, synth: realSynthRef.current });
          return;
        }
        voiceTrace("tts.aside", { text, cached });
        schedule(buffer);
      };
      const hit = asideCacheRef.current.get(text);
      if (hit) {
        play(hit, true);
        return;
      }
      if (blocked()) {
        voiceTrace("tts.asideDropped", { text, cached: false, clips: clipsRef.current.size, synth: realSynthRef.current });
        return;
      }
      void synthesize(text, reqId)
        .then((buffer) => {
          if (!buffer) return;
          asideCacheRef.current.set(text, buffer);
          play(buffer, false);
        })
        .catch(() => {});
    },
    [synthesize, schedule],
  );

  // Fill the aside cache while the loop is idle, one phrase at a time, so the
  // first "one sec" of a session is not itself three seconds late.
  const warmAsides = useCallback(
    async (texts: readonly string[]) => {
      for (const text of texts) {
        if (asideCacheRef.current.has(text)) continue;
        // A live reply owns the synthesizer; stop here and let the next idle call carry on.
        if (streamRef.current || realSynthRef.current) return;
        try {
          const buffer = await synthesize(text, reqRef.current);
          if (buffer) asideCacheRef.current.set(text, buffer);
        } catch {
          return;
        }
      }
    },
    [synthesize],
  );

  // A stream stopped by a barge-in or a thread switch is gone, and ending it is a
  // no-op: the mic that cut it in is already the next turn's.
  const endStream = useCallback((finalText: string) => {
    const st = streamRef.current;
    if (!st || st.reqId !== reqRef.current) return;
    const all = splitIntoSentences(finalText);
    while (st.sentences.length - st.extra < all.length) {
      st.sentences.push(all[st.sentences.length - st.extra] ?? "");
    }
    st.final = true;
    voiceTrace("tts.streamEnd", { sentences: st.sentences.length, tail: finalText.slice(-60) });
  }, []);

  const speak = useCallback(
    (text: string) => {
      beginStream();
      endStream(text);
    },
    [beginStream, endStream],
  );

  // Suspending the context freezes every scheduled clip in place; the run loop's tail wait
  // reads the frozen clock, so nothing ends early while held.
  // Held even with no clip scheduled yet: the next sentence is often still
  // synthesizing when a burst lands, and it must not start under the check.
  const pausePlayback = useCallback(() => {
    voiceTrace("tts.pause", { paused: pausedRef.current, stream: Boolean(streamRef.current), clips: clipsRef.current.size });
    if (pausedRef.current || !streamRef.current) return;
    pausedRef.current = true;
    void getPlayCtx()?.suspend().catch(() => {});
  }, []);

  const resumePlayback = useCallback(() => {
    voiceTrace("tts.resume", { paused: pausedRef.current, stream: Boolean(streamRef.current), clips: clipsRef.current.size, ctx: getPlayCtx()?.state });
    if (!pausedRef.current) return;
    pausedRef.current = false;
    void getPlayCtx()?.resume().catch(() => {});
  }, []);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    const ctx = getPlayCtx();
    if (!ctx) return;
    const wake = () => {
      if (pausedRef.current) return;
      if (ctx.state !== "running" && clipsRef.current.size > 0) {
        void ctx.resume().catch(() => {});
      }
    };
    ctx.addEventListener("statechange", wake);
    return () => ctx.removeEventListener("statechange", wake);
  }, []);

  return {
    isSpeaking,
    isPlaying,
    speak,
    beginStream,
    feedText,
    endStream,
    sayAside,
    warmAsides,
    stop,
    pausePlayback,
    resumePlayback,
    primeAudio,
  };
}
