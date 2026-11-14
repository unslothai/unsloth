// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


const BASELINE_MS = 50;
const FRAME_MS = 20;
// Neighbours down to EXTEND_DB stay with the onset frame, so soft onsets and releases are kept.
const ONSET_DB = 24;
const EXTEND_DB = 30;
const SILENT_DBFS = -45;
const PRE_MS = 30;
const POST_MS = 60;
const FADE_IN_MS = 5;
const FADE_OUT_MS = 15;

function removeBaseline(x: Float32Array, win: number): Float32Array {
  const half = win >> 1;
  const n = x.length;
  const at = (i: number) => x[Math.min(n - 1, Math.max(0, i))]!;
  const out = new Float32Array(n);
  let sum = 0;
  for (let i = -half; i <= half; i++) sum += at(i);
  for (let i = 0; i < n; i++) {
    out[i] = x[i]! - sum / (2 * half + 1);
    sum += at(i + half + 1) - at(i - half);
  }
  return out;
}

export function cleanTtsClip(
  channels: readonly Float32Array[],
  sampleRate: number,
): Float32Array[] | null {
  const first = channels[0];
  if (!first || first.length === 0) return null;
  const win = Math.max(1, Math.round((sampleRate * BASELINE_MS) / 1000));
  const flat = channels.map((c) => removeBaseline(c, win));
  const ref = flat[0]!;
  const hop = Math.max(1, Math.round((sampleRate * FRAME_MS) / 1000));
  const frames = Math.floor(ref.length / hop);
  const db = new Float32Array(frames);
  let peak = -Infinity;
  for (let f = 0; f < frames; f++) {
    let e = 0;
    for (let i = f * hop; i < (f + 1) * hop; i++) e += ref[i]! * ref[i]!;
    db[f] = 10 * Math.log10(e / hop + 1e-18);
    if (db[f]! > peak) peak = db[f]!;
  }
  if (!(peak > SILENT_DBFS)) return null;
  let on = 0;
  while (db[on]! <= peak - ONSET_DB) on++;
  while (on > 0 && db[on - 1]! > peak - EXTEND_DB) on--;
  let off = frames - 1;
  while (db[off]! <= peak - ONSET_DB) off--;
  while (off < frames - 1 && db[off + 1]! > peak - EXTEND_DB) off++;
  const start = Math.max(0, on * hop - Math.round((sampleRate * PRE_MS) / 1000));
  const end = Math.min(ref.length, (off + 1) * hop + Math.round((sampleRate * POST_MS) / 1000));
  const fin = Math.round((sampleRate * FADE_IN_MS) / 1000);
  const fout = Math.round((sampleRate * FADE_OUT_MS) / 1000);
  return flat.map((c) => {
    const y = c.slice(start, end);
    for (let i = 0; i < Math.min(fin, y.length); i++) y[i] = y[i]! * (i / fin);
    for (let i = 0; i < Math.min(fout, y.length); i++) y[y.length - 1 - i] = y[y.length - 1 - i]! * (i / fout);
    return y;
  });
}
