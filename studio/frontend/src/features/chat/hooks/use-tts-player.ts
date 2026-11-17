// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  generateStudioTtsClip,
  type TtsTiming,
  uploadVoiceReference,
} from "@/features/chat/adapters/studio-speech-synthesis-adapter";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { getLocale, translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import { splitSpoken, splitStreaming, splitStreamingSpans, stripForSpeech, type SpokenSentence } from "../voice/speech-text";
import type { PlayedChunk, PlayedEntry } from "../voice/heard-cut";
import { cleanTtsClip } from "../voice/tts-clip-clean";
import { firstClauseCut, pauseAfterChunk, planTtsChunk, spokenChunk } from "../voice/tts-chunk-plan";
import { useVoiceSettingsStore } from "../../settings/stores/voice-settings-store";
import { normalizeForSpeech } from "../voice/speech-normalize";
import { getVoicePlayContext, settleVoiceTurn } from "../voice/voice-loop-bridge";
import { outputSpectrumNode } from "../voice/voice-spectrum";
import { voiceTrace } from "../voice/voice-trace";
import {
  VOICE_REFERENCE_SECONDS,
  VOICE_REFERENCE_TEXT,
  type VoiceReference,
  voicePinGeneration,
  wavSeconds,
  withVoicePin,
} from "../voice/voice-pin";
import {
  cutAbortsSpeechRequest,
  isChatModelVoice,
  voiceTtsCanClone,
  voiceVariesPerSentence,
} from "../voice/voice-models";
import { asideBlocked, createClipLedger, type ClipLedger } from "../voice/tts-clip-ledger";

export { splitStreaming };

export const voiceOutputLevel = { current: 0 };

// audio.cpp serializes per model, so synthesizing further ahead only queues work a barge-in discards.
const MAX_QUEUED_CLIPS = 2;
// Paused, no barge-in can land, so synthesis runs ahead; bounded because a decoded sentence is a few MB.
const MAX_QUEUED_CLIPS_HELD = 8;
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

/** When a sample scheduled at context time `start` reaches the speaker, on the performance.now() clock. */
export function audibleAt(
  start: number,
  ctx: Pick<AudioContext, "currentTime"> &
    Partial<Pick<AudioContext, "outputLatency" | "baseLatency" | "getOutputTimestamp">>,
  now: number,
): number {
  const stamp = ctx.getOutputTimestamp?.();
  // A suspended or unsupported context reports zeros: count from now plus the output latency instead.
  if (stamp?.contextTime && stamp.performanceTime) {
    return stamp.performanceTime + (start - stamp.contextTime) * 1000;
  }
  return now + (start - ctx.currentTime + (ctx.outputLatency ?? 0) + (ctx.baseLatency ?? 0)) * 1000;
}

/** One requested piece of text and where it sits in splitter sentence `source`. */
export interface SentenceEntry {
  text: string;
  source: number;
  from: number;
  to: number;
}

export interface SentenceQueue {
  sentences: SentenceEntry[];
  /** Splitter sentences queued so far. */
  sources: number;
  /** The clause sent ahead of sentence 0, until that sentence settles. */
  head: string | null;
}

/**
 * Queues newly settled splitter sentences, counted by source so a clause sent early never
 * skips one. With `clause` and nothing queued, the head of the unsettled first sentence goes now.
 */
export function queueSentences(
  q: SentenceQueue,
  settled: readonly string[],
  partial: string,
  clause: boolean,
): { mismatch: boolean; head: string | null } {
  let mismatch = false;
  while (q.sources < settled.length) {
    const source = q.sources++;
    const text = settled[source] ?? "";
    if (source === 0 && q.head !== null) {
      const head = q.head;
      q.head = null;
      if (text.startsWith(head)) {
        const from = text.length - text.slice(head.length).trimStart().length;
        if (from < text.length) q.sentences.push({ text: text.slice(from), source, from, to: text.length });
        continue;
      }
      // The sentence settled differently: say it whole, repeating a few words.
      mismatch = true;
    }
    q.sentences.push({ text, source, from: 0, to: text.length });
  }
  if (!clause || q.sources > 0 || q.sentences.length > 0) return { mismatch, head: null };
  const cut = firstClauseCut(partial);
  if (cut === null) return { mismatch, head: null };
  const head = partial.slice(0, cut).trimEnd();
  q.head = head;
  q.sentences.push({ text: head, source: 0, from: 0, to: head.length });
  return { mismatch, head };
}

interface Stream extends SentenceQueue {
  reqId: number;
  /** Where this stream's text starts in the reply: 0, or the old length on the run-flicker path. */
  base: number;
  /** Settled sentences by source, with their offsets into the stream's text. */
  spans: SpokenSentence[];
  /** Every clip scheduled for this stream, on the AudioContext clock; never pruned. */
  log: PlayedChunk[];
  /** The chat model spoke the reply (feedClip): its clips carry no text timing. */
  untimed: boolean;
  /** The voice turn this reply answers, for the turn.* trace; null outside one. */
  turnId: string | null;
  /** A reply clip has been scheduled, so the turn has its first audio. */
  audible: boolean;
  /** Read once per stream, so a clause already sent is always followed by its remainder. */
  clause: boolean;
  next: number;
  final: boolean;
  /** Spoken-reply chunks (feedClip) still decoding; the stream ends only once they are scheduled. */
  decoding: number;
  clips: Promise<void>;
}

function currentVoiceKey(): string {
  const { selectedVoiceModelId, selectedVoiceId, selectedVoiceSpeaker } = useChatRuntimeStore.getState();
  // The session voice of a model with no preset speaker is redrawn each session.
  return `${selectedVoiceModelId ?? ""}\u0000${selectedVoiceId ?? ""}\u0000${selectedVoiceSpeaker ?? ""}\u0000${voicePinGeneration()}`;
}

const ASIDE_MAX_LATE_MS = 4000;

export function useTtsPlayer(onPlaybackEnd?: () => void): {
  isSpeaking: boolean;
  isPlaying: boolean;
  speak(text: string, meta?: StreamMeta): void;
  beginStream(meta?: StreamMeta): void;
  feedText(text: string): void;
  endStream(finalText: string): void;
  feedClip(wavBase64: string): void;
  endClips(): void;
  sayAside(text: string, opts?: AsideOptions): void;
  warmAsides(texts: readonly string[]): Promise<void>;
  stop(): void;
  pausePlayback(): void;
  resumePlayback(): void;
  /** Survives stop(), so a reply that starts while held stays silent until holdPlayback(false). */
  holdPlayback(held: boolean): void;
  primeAudio(): void;
  playbackPosition(): number;
  playedLog(): { log: PlayedChunk[]; base: number } | null;
} {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const reqRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const playHeadRef = useRef(0);
  const streamRef = useRef<Stream | null>(null);
  const asideCacheRef = useRef(new Map<string, AudioBuffer>());
  // The voice the cached asides were made in: a new pick (model, saved voice or speaker)
  // starts the cache over, so a filler never speaks in the voice the user moved away from.
  const asideVoiceRef = useRef("");
  const asides = useCallback((): Map<string, AudioBuffer> => {
    const voice = currentVoiceKey();
    if (voice !== asideVoiceRef.current) {
      asideVoiceRef.current = voice;
      asideCacheRef.current.clear();
    }
    return asideCacheRef.current;
  }, []);
  const keepAside = useCallback(
    (text: string, buffer: AudioBuffer, voice: string) => {
      if (voice === currentVoiceKey()) asides().set(text, buffer);
    },
    [asides],
  );
  // A reply sentence is at the synthesizer: an aside now would land right before it.
  const realSynthRef = useRef(false);
  const failedReqRef = useRef(-1);
  const rafRef = useRef<number | null>(null);
  const pausedRef = useRef(false);
  const heldRef = useRef(false);
  const onEndRef = useRef(onPlaybackEnd);
  onEndRef.current = onPlaybackEnd;
  // isPlaying follows the ledger alone, so a clip that never sounds cannot hold it on.
  const [ledger] = useState<ClipLedger>(() => createClipLedger({
    ctx: getPlayCtx,
    frozen: () => pausedRef.current || heldRef.current,
    onBusy: () => setIsPlaying(true),
    onIdle: () => {
      voiceTrace("tts.lastClipEnded", { stream: Boolean(streamRef.current) });
      playHeadRef.current = 0;
      setIsPlaying(false);
    },
    trace: voiceTrace,
    now: () => performance.now(),
    setTimer: (fn, ms) => setTimeout(fn, ms),
    clearTimer: (t) => clearTimeout(t as ReturnType<typeof setTimeout>),
  }));
  const clipsRef = useRef(ledger);

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
    for (const clip of clipsRef.current.clips()) {
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
      if (!heldRef.current) void getPlayCtx()?.resume().catch(() => {});
    }
    if (cutAbortsSpeechRequest(useChatRuntimeStore.getState().selectedVoiceModelId)) {
      abortRef.current?.abort();
    }
    abortRef.current = null;
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
    async (
      sentence: string,
      reqId: number,
      onResponse?: (timing?: TtsTiming) => void,
    ): Promise<AudioBuffer | null> => {
      const ctx = getPlayCtx();
      if (!ctx) return null;
      const request = async () => {
        const { selectedVoiceModelId, selectedVoiceId, selectedVoiceSpeaker, params } =
          useChatRuntimeStore.getState();
        // The request text only: what the reply shows, and what Phase 4 records, stay as written.
        const { speechNormalize, dictationLanguage } = useVoiceSettingsStore.getState();
        // "auto" names no language: the app's own stands in, so the default setting is not a no-op.
        const language = dictationLanguage === "auto" ? getLocale() : dictationLanguage;
        const spoken = speechNormalize ? normalizeForSpeech(sentence, language) : sentence;
        const signal = abortRef.current?.signal;
        // The chat model's voice is the selected chat model, which may sit in a kept slot.
        const model = isChatModelVoice(selectedVoiceModelId) ? params.checkpoint : selectedVoiceModelId;
        const voiceId = voiceTtsCanClone(selectedVoiceModelId) ? selectedVoiceId : "";
        const speak = (reference: VoiceReference | null) =>
          generateStudioTtsClip(spoken, signal, {
            model,
            voiceId,
            speaker: selectedVoiceSpeaker,
            reference: reference ?? undefined,
          });
        // A voice with no preset speaker draws a new one per sentence: keep the first for the session.
        const clip =
          !voiceId && !selectedVoiceSpeaker && voiceVariesPerSentence(selectedVoiceModelId)
            ? await withVoicePin(model, speak, async () => {
                // Not tied to this reply's signal: a barge-in must not cost the session its voice.
                const drawn = await generateStudioTtsClip(VOICE_REFERENCE_TEXT, undefined, {
                  model,
                  voiceId: "",
                });
                const wav = wavBytesFromDataUrl(drawn.url);
                const seconds = wavSeconds(wav);
                if (seconds === null || seconds < VOICE_REFERENCE_SECONDS.min || seconds > VOICE_REFERENCE_SECONDS.max) {
                  voiceTrace("tts.voicePinSkipped", { model, seconds });
                  return null;
                }
                const inputId = await uploadVoiceReference(wav);
                voiceTrace("tts.voicePinned", { model, inputId, seconds: Math.round(seconds * 10) / 10 });
                return { inputId, text: VOICE_REFERENCE_TEXT };
              })
            : await speak(null);
        onResponse?.(clip.timing);
        return clip.url;
      };
      // A named generation lets "switch model by request" swap the chat model out for the voice: own the slot first.
      const { voiceSlotLoaded, ensureVoiceSlotLoaded } = useChatRuntimeStore.getState();
      if (!voiceSlotLoaded && ensureVoiceSlotLoaded) {
        if (!(await ensureVoiceSlotLoaded())) return null;
        if (reqRef.current !== reqId) return null;
      }
      let url: string;
      try {
        url = await request();
      } catch (error) {
        if (reqRef.current !== reqId) return null;
        if (/saved voice no longer exists/i.test(error instanceof Error ? error.message : "")) {
          // Deleted on the Audio page: the pick is dropped so this and every later sentence use the model's voice.
          useChatRuntimeStore.getState().setSelectedVoiceId("");
          url = await request();
        } else {
          const ensure = useChatRuntimeStore.getState().ensureVoiceSlotLoaded;
          if (!ensure || !(await ensure())) throw error;
          if (reqRef.current !== reqId) return null;
          url = await request();
        }
      }
      if (reqRef.current !== reqId) return null;
      const bytes = wavBytesFromDataUrl(url);
      let decoded: AudioBuffer;
      try {
        decoded = await ctx.decodeAudioData(bytes.slice(0));
      } catch (error) {
        // A clip the decoder refused once has decoded on the second try, so retry once before giving up.
        voiceTrace("tts.decodeRetry", { bytes: bytes.byteLength });
        decoded = await ctx.decodeAudioData(bytes.slice(0));
      }
      const chans = Array.from({ length: decoded.numberOfChannels }, (_, c) => decoded.getChannelData(c));
      const cleaned = cleanTtsClip(chans, decoded.sampleRate);
      if (!cleaned) {
        voiceTrace("tts.clipSilent", { text: sentence, seconds: Math.round(decoded.duration * 100) / 100 });
        return null;
      }
      const out = ctx.createBuffer(cleaned.length, cleaned[0]!.length, decoded.sampleRate);
      cleaned.forEach((data, c) => out.getChannelData(c).set(data));
      return out;
    },
    [],
  );

  const schedule = useCallback(
    (buffer: AudioBuffer, text: string, entries: readonly PlayedEntry[] = []): number | null => {
      const ctx = getPlayCtx();
      if (!ctx) return null;
      // WebKit parks the context when another surface takes the audio session; start() on it is silent.
      if (ctx.state !== "running" && !pausedRef.current && !heldRef.current)
        void ctx.resume().catch(() => {});
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(outputSpectrumNode(ctx));
      const start = Math.max(ctx.currentTime + 0.02, playHeadRef.current);
      voiceTrace("tts.scheduled", {
        inMs: Math.round((start - ctx.currentTime) * 1000),
        seconds: Math.round(buffer.duration * 1000) / 1000,
      });
      clipsRef.current.add({ source, start, duration: buffer.duration, text, env: envelopeOf(buffer) });
      streamRef.current?.log.push({ entries, start, duration: buffer.duration });
      playHeadRef.current = start + buffer.duration;
      if (rafRef.current === null) rafRef.current = requestAnimationFrame(meter);
      return start;
    },
    [meter],
  );

  // The first reply clip on the timeline ends the voice turn; asides do not count.
  const markAudible = useCallback((st: Stream, start: number | null) => {
    const ctx = getPlayCtx();
    if (st.audible || start === null || !ctx) return;
    st.audible = true;
    if (!st.turnId) return;
    voiceTrace("turn.firstAudible", {
      turnId: st.turnId,
      at: Math.round(audibleAt(start, ctx, performance.now())),
      held: pausedRef.current || heldRef.current,
    });
    settleVoiceTurn(st.turnId);
  }, []);

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
          if (clipsRef.current.size >= (heldRef.current ? MAX_QUEUED_CLIPS_HELD : MAX_QUEUED_CLIPS)) {
            await sleep(40);
            continue;
          }
          const pending = st.sentences.slice(st.next).map((s) => stripForSpeech(s.text));
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
          const text = spokenChunk(pending.slice(0, take));
          if (!text) continue;
          const entries = st.sentences.slice(i, i + take).map((piece, k) => playedEntry(st, piece, pending[k].length));
          first = false;
          try {
            voiceTrace("tts.sentence", { i, sentences: take, chars: text.length, text });
            if (st.turnId) voiceTrace("turn.ttsSent", { turnId: st.turnId, i });
            realSynthRef.current = true;
            let buffer: AudioBuffer | null;
            try {
              buffer = await synthesize(text, reqId, (timing) => {
                if (st.turnId) voiceTrace("turn.ttsReceived", { turnId: st.turnId, i, timing });
              });
            } finally {
              realSynthRef.current = false;
            }
            if (!live()) return;
            if (buffer) {
              voiceTrace("tts.clip", { i, sentences: take, seconds: Math.round(buffer.duration * 100) / 100 });
              const start = schedule(buffer, text, entries);
              // A clause head is followed by the comma pause it lost by being sent alone.
              if (start !== null) playHeadRef.current += pauseAfterChunk(text) / 1000;
              markAudible(st, start);
            }
          } catch (error) {
            if (!live()) return;
            voiceTrace("tts.error", {
              i,
              message: error instanceof Error ? error.message : String(error),
            });
            if (failedReqRef.current !== reqId) {
              failedReqRef.current = reqId;
              toast.error(translate("voiceControls.synthFailed"), {
                description: error instanceof Error ? error.message : undefined,
              });
            }
          }
        } else if (st.final && st.decoding === 0) {
          break;
        } else {
          await sleep(40);
        }
      }
      const ctx = getPlayCtx();
      while (live() && clipsRef.current.size > 0) {
        if (ctx && ctx.currentTime > playHeadRef.current + 0.5) break;
        await sleep(50);
      }
      if (!live()) return;
      // onended fires before the speaker drains (Bluetooth holds ~100s of ms); re-arming the mic then moves the route.
      const latencyMs =
        ((ctx?.outputLatency ?? 0) + (ctx?.baseLatency ?? 0)) * 1000;
      voiceTrace("tts.streamDone", { latencyMs: Math.round(latencyMs) });
      await sleep(Math.max(250, latencyMs + 150));
      if (!live()) return;
      const done = streamRef.current;
      if (done?.turnId && !done.audible) {
        voiceTrace("turn.end", { turnId: done.turnId, reason: "noTts" });
        settleVoiceTurn(done.turnId);
      }
      streamRef.current = null;
      setIsSpeaking(false);
      setIsPlaying(false);
      onEndRef.current?.();
    },
    [synthesize, schedule, markAudible],
  );

  const beginStream = useCallback((meta?: StreamMeta) => {
    stop();
    const reqId = reqRef.current;
    abortRef.current = new AbortController();
    streamRef.current = {
      reqId,
      turnId: meta?.turnId ?? null,
      base: meta?.base ?? 0,
      spans: [],
      log: [],
      untimed: false,
      audible: false,
      clause: useVoiceSettingsStore.getState().clauseFirstChunk,
      sentences: [],
      sources: 0,
      head: null,
      next: 0,
      final: false,
      decoding: 0,
      clips: Promise.resolve(),
    };
    setIsSpeaking(true);
    void run(reqId);
  }, [stop, run]);

  const feedText = useCallback((text: string) => {
    const st = streamRef.current;
    if (!st || st.reqId !== reqRef.current) return;
    const { complete, partial } = splitStreamingSpans(text);
    st.spans = complete;
    queueAndTrace(
      st,
      complete.map((s) => s.text),
      partial,
    );
  }, []);

  // Asides used to queue behind sentences and came out late: one plays only on an empty timeline, from cache.
  const sayAside = useCallback(
    (text: string, opts?: AsideOptions) => {
      const st = streamRef.current;
      if (!st || st.reqId !== reqRef.current || st.final) return;
      const patient = Boolean(opts?.patient);
      // Held or paused, the reply synthesizes ahead and plays on resume, so a filler would only delay it.
      if (heldRef.current || (pausedRef.current && !patient)) {
        voiceTrace("tts.asideDropped", { text, paused: true });
        return;
      }
      const reqId = reqRef.current;
      const blocked = () => asideBlocked(clipsRef.current, { patient, replySynth: realSynthRef.current });
      const play = (buffer: AudioBuffer, cached: boolean, said = text) => {
        // The reply may have ended while this synthesized: a late filler would talk over the next turn.
        if (reqRef.current !== reqId || streamRef.current !== st || st.final) return;
        if (blocked()) {
          voiceTrace("tts.asideDropped", { text: said, cached, clips: clipsRef.current.size, synth: realSynthRef.current });
          return;
        }
        voiceTrace("tts.aside", { text: said, cached });
        schedule(buffer, said);
      };
      const voice = currentVoiceKey();
      const hit = asides().get(text);
      if (hit) {
        play(hit, true);
        return;
      }
      if (blocked()) {
        voiceTrace("tts.asideDropped", { text, cached: false, clips: clipsRef.current.size, synth: realSynthRef.current });
        return;
      }
      // Live synthesis competes with the model for the GPU: say a ready stand-in now, cache this one.
      const standIn = opts?.standIns?.find((t) => asides().has(t));
      if (standIn) {
        voiceTrace("tts.asideStandIn", { text, standIn });
        play(asides().get(standIn)!, true, standIn);
        void synthesize(text, reqId)
          .then((buffer) => {
            if (buffer) keepAside(text, buffer, voice);
          })
          .catch(() => {});
        return;
      }
      const askedAt = performance.now();
      void synthesize(text, reqId)
        .then((buffer) => {
          if (!buffer) return;
          keepAside(text, buffer, voice);
          const late = performance.now() - askedAt;
          if (late > ASIDE_MAX_LATE_MS && !patient) {
            voiceTrace("tts.asideDropped", { text, cached: false, late: Math.round(late) });
            return;
          }
          play(buffer, false);
        })
        .catch(() => {});
    },
    [synthesize, schedule, asides, keepAside],
  );

  // Phrases a warm-up is synthesizing now: overlapping calls share them instead of each sending its own.
  const warmingRef = useRef(new Set<string>());
  const warmAsides = useCallback(
    async (texts: readonly string[]) => {
      let added = 0;
      const voice = currentVoiceKey();
      for (const text of texts) {
        if (asides().has(text) || warmingRef.current.has(text)) continue;
        if (voice !== currentVoiceKey()) break;
        // A live reply owns the synthesizer; the next idle call carries on.
        if (streamRef.current || realSynthRef.current) break;
        warmingRef.current.add(text);
        try {
          const buffer = await synthesize(text, reqRef.current);
          if (buffer && voice === currentVoiceKey()) {
            keepAside(text, buffer, voice);
            added++;
          }
        } catch {
          break;
        } finally {
          warmingRef.current.delete(text);
        }
      }
      if (added) voiceTrace("tts.warm", { added, cached: asides().size, of: texts.length });
    },
    [synthesize, asides, keepAside],
  );

  // A stream stopped by a barge-in is gone; ending it is a no-op, the mic that cut it is the next turn's.
  const endStream = useCallback((finalText: string) => {
    const st = streamRef.current;
    if (!st || st.reqId !== reqRef.current) return;
    st.spans = splitSpoken(finalText);
    queueAndTrace(
      st,
      st.spans.map((s) => s.text),
      "",
    );
    st.final = true;
    voiceTrace("tts.streamEnd", { sentences: st.sentences.length, tail: finalText.slice(-60) });
  }, []);

  const feedClip = useCallback(
    (wavBase64: string) => {
      const st = streamRef.current;
      const ctx = getPlayCtx();
      if (!st || st.reqId !== reqRef.current || st.final || !ctx) return;
      const reqId = st.reqId;
      st.untimed = true;
      st.decoding += 1;
      // Decoded off the chain, scheduled on it, so a chunk never jumps the one before.
      const decoded = ctx
        .decodeAudioData(wavBytesFromDataUrl(`data:audio/wav;base64,${wavBase64}`))
        .catch((error: unknown) => {
          voiceTrace("tts.spokenDecodeFailed", { message: error instanceof Error ? error.message : String(error) });
          return null;
        });
      st.clips = st.clips.then(async () => {
        try {
          const buffer = await decoded;
          if (buffer && reqRef.current === reqId) {
            voiceTrace("tts.spokenClip", { seconds: Math.round(buffer.duration * 100) / 100 });
            markAudible(st, schedule(buffer, ""));
          }
        } finally {
          st.decoding -= 1;
        }
      });
    },
    [schedule, markAudible],
  );

  const endClips = useCallback(() => {
    const st = streamRef.current;
    if (!st || st.reqId !== reqRef.current) return;
    st.final = true;
    voiceTrace("tts.spokenEnd", { decoding: st.decoding });
  }, []);

  const speak = useCallback(
    (text: string, meta?: StreamMeta) => {
      beginStream(meta);
      endStream(text);
    },
    [beginStream, endStream],
  );

  // Held even with no clip scheduled: the next sentence is often still synthesizing and must not start under the check.
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
    if (!heldRef.current) void getPlayCtx()?.resume().catch(() => {});
  }, []);

  const holdPlayback = useCallback((held: boolean) => {
    voiceTrace("tts.hold", { held, stream: Boolean(streamRef.current), clips: clipsRef.current.size });
    if (heldRef.current === held) return;
    heldRef.current = held;
    const ctx = getPlayCtx();
    if (!ctx) return;
    if (held) void ctx.suspend().catch(() => {});
    else if (!pausedRef.current) void ctx.resume().catch(() => {});
  }, []);

  // The audible position, not the scheduled one: the same latency correction the stream's end uses.
  const playbackPosition = useCallback((): number => {
    const ctx = getPlayCtx();
    return ctx ? ctx.currentTime - ((ctx.outputLatency ?? 0) + (ctx.baseLatency ?? 0)) : 0;
  }, []);

  /** This stream's playback log; null with no stream, or when the chat model spoke the reply. */
  const playedLog = useCallback((): { log: PlayedChunk[]; base: number } | null => {
    const st = streamRef.current;
    return st && !st.untimed ? { log: [...st.log], base: st.base } : null;
  }, []);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    const ctx = getPlayCtx();
    if (!ctx) return;
    const wake = () => {
      if (pausedRef.current || heldRef.current) return;
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
    feedClip,
    endClips,
    sayAside,
    warmAsides,
    stop,
    pausePlayback,
    resumePlayback,
    holdPlayback,
    primeAudio,
    playbackPosition,
    playedLog,
  };
}

// Only a piece that ends its sentence carries the sentence's end, the one offset a cut lands on.
function playedEntry(st: Stream, piece: SentenceEntry, chars: number): PlayedEntry {
  const span = st.spans[piece.source];
  const start = span?.start ?? 0;
  const whole = span !== undefined && piece.to === span.text.length;
  return { source: piece.source, from: start + piece.from, to: whole ? span.end : start + piece.to, chars };
}function queueAndTrace(st: Stream, settled: readonly string[], partial: string): void {
  const had = st.sentences.length;
  const { mismatch, head } = queueSentences(st, settled, partial, st.clause && !st.final);
  if (head !== null) voiceTrace("tts.clauseHead", { chars: head.length });
  if (mismatch) voiceTrace("tts.clauseMismatch", { sentence: settled[0]?.slice(0, 80) });
  if (had === 0 && st.sentences.length > 0 && st.turnId) voiceTrace("turn.firstChunk", { turnId: st.turnId });
}

export interface StreamMeta {
  turnId?: string;
  /** Offset in the reply where the streamed text starts (the run-flicker path speaks only what was added). */
  base?: number;
}

export interface AsideOptions {
  standIns?: readonly string[];
  patient?: boolean;
}


