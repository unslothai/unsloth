// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dependency-free so the node test runner loads it directly.

import { encodeWav } from "../adapters/pcm-recorder.ts";

function bytesOf(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function base64Of(bytes: Uint8Array): string {
  let binary = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    binary += String.fromCharCode(...bytes.subarray(i, i + step));
  }
  return btoa(binary);
}

function parseWav(bytes: Uint8Array): { fmt: Uint8Array; data: Uint8Array } | null {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (at: number) => String.fromCharCode(...bytes.subarray(at, at + 4));
  if (bytes.length < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  let fmt: Uint8Array | null = null;
  for (let at = 12; at + 8 <= bytes.length; ) {
    const size = view.getUint32(at + 4, true);
    const body = bytes.subarray(at + 8, Math.min(bytes.length, at + 8 + size));
    if (tag(at) === "fmt ") fmt = body;
    else if (tag(at) === "data") return fmt ? { fmt, data: body } : null;
    at += 8 + size + (size % 2);
  }
  return null;
}

function wavBytes(fmt: Uint8Array, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + 8 + fmt.length + 8 + data.length);
  const view = new DataView(out.buffer);
  const ascii = (at: number, s: string) => {
    for (let i = 0; i < 4; i++) out[at + i] = s.charCodeAt(i);
  };
  ascii(0, "RIFF");
  view.setUint32(4, out.length - 8, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, fmt.length, true);
  out.set(fmt, 20);
  const dataAt = 20 + fmt.length;
  ascii(dataAt, "data");
  view.setUint32(dataAt + 4, data.length, true);
  out.set(data, dataAt + 8);
  return out;
}

/** One WAV of every chunk, in order. A chunk that is not a WAV, or differs in format, is left out. */
export function joinWavBase64(chunks: readonly string[]): string {
  if (chunks.length === 1) return chunks[0] ?? "";
  let fmt: Uint8Array | null = null;
  const parts: Uint8Array[] = [];
  for (const chunk of chunks) {
    const wav = parseWav(bytesOf(chunk));
    if (!wav) continue;
    if (fmt && fmt.join(",") !== wav.fmt.join(",")) continue;
    fmt ??= wav.fmt;
    parts.push(wav.data);
  }
  if (!fmt) return chunks[0] ?? "";
  const data = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    data.set(p, at);
    at += p.length;
  }
  return base64Of(wavBytes(fmt, data));
}

export function pcmWavBase64(samples: Float32Array, sampleRate: number): string {
  return base64Of(encodeWav(samples, sampleRate));
}
