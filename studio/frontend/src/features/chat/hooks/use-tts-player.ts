// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { generateStudioTtsAudio } from "@/features/chat/adapters/studio-speech-synthesis-adapter";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import { splitIntoSentences, stripForSpeech } from "../voice/speech-text";
import { getVoicePlayContext } from "../voice/voice-loop-bridge";
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
}

/**
 * Speaks a reply sentence by sentence through voice mode's audio.cpp voice while the
 * LLM is still writing it. Each sentence is one /audio/generate call; clips play in
 * order on one AudioContext playhead.
 */
export function useTtsPlayer(onPlaybackEnd?: () => void): {
  isSpeaking: boolean;
  /** True only while a clip is audible (not while the first one synthesizes). */
  isPlaying: boolean;
  speak(text: string): void;
  beginStream(): void;
  feedText(text: string): void;
  endStream(finalText: string): void;
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
        const { selectedVoiceModelId, selectedVoiceId } = useChatRuntimeStore.getState();
        return generateStudioTtsAudio(sentence, abortRef.current?.signal, {
          model: selectedVoiceModelId,
          voiceId: voiceTtsCanClone(selectedVoiceModelId) ? selectedVoiceId : "",
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
      return await ctx.decodeAudioData(wavBytesFromDataUrl(url));
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
      if (ctx.state !== "running") void ctx.resume().catch(() => {});
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const start = Math.max(ctx.currentTime + 0.02, playHeadRef.current);
      const clip: Clip = { source, start, env: envelopeOf(buffer) };
      source.onended = () => {
        clipsRef.current.delete(clip);
        if (clipsRef.current.size === 0) setIsPlaying(false);
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
      for (;;) {
        const st = streamRef.current;
        if (!live() || !st) return;
        if (st.next < st.sentences.length) {
          if (clipsRef.current.size >= MAX_QUEUED_CLIPS) {
            await sleep(40);
            continue;
          }
          const sentence = stripForSpeech(st.sentences[st.next] ?? "");
          st.next++;
          if (!sentence) continue;
          try {
            const buffer = await synthesize(sentence, reqId);
            if (!live()) return;
            if (buffer) schedule(buffer);
          } catch (error) {
            if (!live()) return;
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
    streamRef.current = { reqId, sentences: [], next: 0, final: false };
    setIsSpeaking(true);
    void run(reqId);
  }, [stop, run]);

  const feedText = useCallback((text: string) => {
    const st = streamRef.current;
    if (!st || st.reqId !== reqRef.current) return;
    const { complete } = splitStreaming(text);
    while (st.sentences.length < complete.length) {
      st.sentences.push(complete[st.sentences.length] ?? "");
    }
  }, []);

  // A stream stopped by a barge-in or a thread switch is gone, and ending it is a
  // no-op: the mic that cut it in is already the next turn's.
  const endStream = useCallback((finalText: string) => {
    const st = streamRef.current;
    if (!st || st.reqId !== reqRef.current) return;
    const all = splitIntoSentences(finalText);
    while (st.sentences.length < all.length) {
      st.sentences.push(all[st.sentences.length] ?? "");
    }
    st.final = true;
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
  const pausePlayback = useCallback(() => {
    if (pausedRef.current || clipsRef.current.size === 0) return;
    pausedRef.current = true;
    void getPlayCtx()?.suspend().catch(() => {});
  }, []);

  const resumePlayback = useCallback(() => {
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
    stop,
    pausePlayback,
    resumePlayback,
    primeAudio,
  };
}
