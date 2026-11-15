// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dependency-free so the node test runner loads it directly.

import { joinWavBase64, pcmWavBase64 } from "../utils/wav-chunks.ts";

// Audio-input models read 16 kHz; the mic runs at 48 kHz, so sending it raw is 3x the bytes.
export const TURN_AUDIO_RATE = 16000;

/** Box-filtered decimation: each output sample averages the input window it covers. */
export function downsample(samples: Float32Array, rate: number, target = TURN_AUDIO_RATE): Float32Array {
  if (rate <= target) return samples;
  const step = rate / target;
  const out = new Float32Array(Math.floor(samples.length / step));
  for (let i = 0; i < out.length; i++) {
    const from = Math.floor(i * step);
    const to = Math.min(samples.length, Math.floor((i + 1) * step));
    let sum = 0;
    for (let j = from; j < to; j++) sum += samples[j] ?? 0;
    out[i] = to > from ? sum / (to - from) : 0;
  }
  return out;
}

export function createTurnAudio() {
  let chunks: Float32Array[] = [];
  let rate = 0;
  // Speech of utterances queued behind the reply: kept when a later one is dropped.
  let parked: string[] = [];
  const current = (): string | null => {
    if (!chunks.length || !rate) return null;
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let at = 0;
    for (const c of chunks) {
      merged.set(c, at);
      at += c.length;
    }
    chunks = [];
    return pcmWavBase64(merged, rate);
  };
  return {
    add(samples: Float32Array, sampleRate: number): void {
      const next = Math.min(sampleRate, TURN_AUDIO_RATE);
      if (rate && next !== rate) chunks = [];
      rate = next;
      chunks.push(downsample(samples, sampleRate));
    },
    take(): string | null {
      const all = [...parked, current()].filter((w): w is string => Boolean(w));
      parked = [];
      return all.length ? joinWavBase64(all) : null;
    },
    park(): void {
      const wav = current();
      if (wav) parked.push(wav);
    },
    dropCurrent(): void {
      chunks = [];
    },
    clear(): void {
      chunks = [];
      parked = [];
    },
    get size(): number {
      return chunks.length + parked.length;
    },
  };
}

export function voiceTurnAttachment(wavBase64: string, id: string) {
  return {
    id,
    type: "file" as const,
    name: "voice.wav",
    contentType: "audio/wav",
    content: [{ type: "audio" as const, audio: { data: wavBase64, format: "wav" as const } }],
    status: { type: "complete" as const },
  };
}

type Part = { type: string };
type Message = { role: string; content?: readonly Part[]; attachments?: readonly { content?: readonly Part[] }[] };

export function lastUserTurnHasAudio(messages: readonly Message[]): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "user") continue;
    const parts = [...(m.content ?? []), ...(m.attachments ?? []).flatMap((a) => a.content ?? [])];
    return parts.some((p) => p.type === "audio");
  }
  return false;
}

type AudioPart = { type: string; audio?: { data?: string } | string };

function audioOf(m: { content?: readonly Part[]; attachments?: readonly { content?: readonly Part[] }[] }): string[] {
  const parts = [...(m.content ?? []), ...(m.attachments ?? []).flatMap((a) => a.content ?? [])] as AudioPart[];
  return parts
    .filter((p) => p.type === "audio")
    .map((p) => (typeof p.audio === "string" ? p.audio : (p.audio?.data ?? "")))
    .filter(Boolean);
}

/** Coalesced bubbles are one prompt but only the newest bubble's audio is sent; each already holds the run's speech so far. */
export function coalescedTurnAudio(messages: readonly Message[], turn: string): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "user") break;
    const earlier = audioOf(m);
    if (earlier.length) return joinWavBase64([...earlier, turn]);
  }
  return turn;
}

const AUDIO_PLAYER = /<audio-player\b[^>]*\/>/g;

export function withoutAudioPlayer(text: string): string {
  return text.includes("<audio-player") ? text.replace(AUDIO_PLAYER, "").trim() : text;
}
