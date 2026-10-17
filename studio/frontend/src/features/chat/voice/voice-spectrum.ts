// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Live spectrum of the two voices in the loop, for the orb. One analyser on the mic
// graph and one on the playback graph; the orb reads log-spaced bands per frame.

export const SPECTRUM_BANDS = 32;
const FFT_SIZE = 1024;
const F_LO = 80;
const F_HI = 8000;
const DB_FLOOR = -85;
const DB_CEIL = -25;
const ATTACK = 0.5;
const RELEASE = 0.14;

type Side = "input" | "output";

interface Tap {
  analyser: AnalyserNode;
  raw: Float32Array<ArrayBuffer>;
  bands: Float32Array;
  edges: Int32Array;
}

const taps: Record<Side, Tap | null> = { input: null, output: null };

function makeTap(ctx: AudioContext): Tap {
  const analyser = ctx.createAnalyser();
  analyser.fftSize = FFT_SIZE;
  analyser.smoothingTimeConstant = 0;
  const binHz = ctx.sampleRate / FFT_SIZE;
  const edges = new Int32Array(SPECTRUM_BANDS + 1);
  for (let i = 0; i <= SPECTRUM_BANDS; i++) {
    const f = F_LO * Math.pow(F_HI / F_LO, i / SPECTRUM_BANDS);
    edges[i] = Math.min(analyser.frequencyBinCount - 1, Math.round(f / binHz));
  }
  return {
    analyser,
    raw: new Float32Array(new ArrayBuffer(analyser.frequencyBinCount * 4)),
    bands: new Float32Array(SPECTRUM_BANDS),
    edges,
  };
}

/** Tap the mic graph. Call once per capture session; the adapter detaches on teardown. */
export function attachInputSpectrum(ctx: AudioContext, source: AudioNode): void {
  const tap = makeTap(ctx);
  source.connect(tap.analyser);
  taps.input = tap;
}

export function detachInputSpectrum(): void {
  try {
    taps.input?.analyser.disconnect();
  } catch {
    // already gone with its context
  }
  taps.input = null;
}

/** The node TTS clips play into (analyser → destination), made once per play context. */
export function outputSpectrumNode(ctx: AudioContext): AudioNode {
  if (!taps.output || taps.output.analyser.context !== ctx) {
    const tap = makeTap(ctx);
    tap.analyser.connect(ctx.destination);
    taps.output = tap;
  }
  return taps.output.analyser;
}

/**
 * Current bands for one side, 0..1 each, smoothed (fast attack, slow release).
 * Returns the mean level, 0 when that side has no tap.
 */
export function readSpectrum(side: Side, out: Float32Array): number {
  const tap = taps[side];
  if (!tap) {
    out.fill(0);
    return 0;
  }
  tap.analyser.getFloatFrequencyData(tap.raw);
  let sum = 0;
  for (let b = 0; b < SPECTRUM_BANDS; b++) {
    const lo = tap.edges[b];
    const hi = Math.max(lo + 1, tap.edges[b + 1]);
    let peak = -Infinity;
    for (let i = lo; i < hi; i++) if (tap.raw[i] > peak) peak = tap.raw[i];
    const v = Math.min(1, Math.max(0, (peak - DB_FLOOR) / (DB_CEIL - DB_FLOOR)));
    const prev = tap.bands[b];
    tap.bands[b] = prev + (v - prev) * (v > prev ? ATTACK : RELEASE);
    out[b] = tap.bands[b];
    sum += tap.bands[b];
  }
  return sum / SPECTRUM_BANDS;
}
