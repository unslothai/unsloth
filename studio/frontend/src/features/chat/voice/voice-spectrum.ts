// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export const SPECTRUM_BANDS = 32;
const FFT_SIZE = 2048;
const F_LO = 90;
const F_HI = 4000;
const DB_FLOOR = -85;
const DB_CEIL = -25;
const TILT_DB_PER_OCTAVE = 6;
const TILT_FROM_HZ = 300;
const ATTACK = 0.5;
const RELEASE = 0.14;
// Each band's running peak sinks back over a few seconds, and never below this, so room
// noise in a band that is always quiet is not blown up to a full band.
const NORM_TAU_MS = 4000;
const NORM_FLOOR = 0.15;
const NORM_STRENGTH = 1;
const FLOOR_CUT = 0.9;

type Side = "input" | "output";

export interface BandState {
  raw: Float32Array;
  tilt: Float32Array;
  bands: Float32Array;
  peaks: Float32Array;
  edges: Int32Array;
  lastMs: number;
}

interface Tap extends BandState {
  analyser: AnalyserNode;
  clock: ScriptProcessorNode | null;
  raw: Float32Array<ArrayBuffer>;
  out: Float32Array;
  level: number;
  readAt: number;
}

const READ_REUSE_MS = 8;

export function bandEdges(
  sampleRate: number,
  fftSize: number = FFT_SIZE,): Int32Array {
  const binHz =sampleRate / fftSize;
  const bins = fftSize / 2;
  const edges = new Int32Array(SPECTRUM_BANDS + 1);
  for (let i = 0; i <= SPECTRUM_BANDS; i++) {
    const f = F_LO * Math.pow(F_HI / F_LO, i / SPECTRUM_BANDS);
    edges[i] = Math.min(bins - 1, Math.round(f / binHz));
  }
  return edges;
}

export function createBandState(
    raw: Float32Array,
  edges: Int32Array,
): BandState {
  const tilt = new Float32Array(SPECTRUM_BANDS);
  for (let b = 0; b < SPECTRUM_BANDS; b++) {
    const f = F_LO * Math.pow(F_HI / F_LO, (b + 0.5) / SPECTRUM_BANDS);
    tilt[b] = Math.max(0, TILT_DB_PER_OCTAVE * Math.log2(f / TILT_FROM_HZ));
  }
  return {
    raw,
    tilt,
    bands: new Float32Array(SPECTRUM_BANDS),
    peaks: new Float32Array(SPECTRUM_BANDS),
    edges,
    lastMs: -1,
  };
}

export function shapeBands(
  state: BandState,
  out: Float32Array,
  nowMs: number,
): number {
  const dt =
    state.lastMs < 0 ? 16 : Math.min(250, Math.max(0, nowMs - state.lastMs));
  state.lastMs = nowMs;
  const sink = Math.exp(-dt / NORM_TAU_MS);
  let sum = 0;
  let loudest = 0;
  for (let b = 0; b < SPECTRUM_BANDS; b++) {
    const lo = state.edges[b];
    const hi = Math.max(lo + 1, state.edges[b + 1]);
    let peak = Number.NEGATIVE_INFINITY;
    for (let i = lo; i < hi; i++) if (state.raw[i] > peak) peak = state.raw[i];
    const v = Math.min(
      1,
      Math.max(0, (peak + state.tilt[b] - DB_FLOOR) / (DB_CEIL - DB_FLOOR)),
    );
    const prev = state.bands[b];
    const band = prev + (v - prev) * (v > prev ? ATTACK : RELEASE);
    state.bands[b] = band;
    state.peaks[b] = Math.max(band, state.peaks[b] * sink);
    sum += band;
    if (band > loudest) loudest = band;
  }
  let top = 0;
  let bottom = Number.POSITIVE_INFINITY;
  for (let b = 0; b < SPECTRUM_BANDS; b++) {
    out[b] =
      state.bands[b] / Math.max(NORM_FLOOR, state.peaks[b]) ** NORM_STRENGTH;
    if (out[b] > top) top = out[b];
    if (out[b] < bottom) bottom = out[b];
  }
  const cut = FLOOR_CUT * bottom;
  for (let b = 0; b < SPECTRUM_BANDS; b++) out[b] -= cut;
  top -= cut;
  const g = top > 1e-6 ? loudest / top : 0;
  for (let b = 0; b < SPECTRUM_BANDS; b++) out[b] *= g;
  return sum / SPECTRUM_BANDS;
}

const taps: Record<Side, Tap | null> = { input: null, output: null };
const frameListeners = new Set<(side: Side) => void>();

export function onSpectrumFrame(cb: (side: Side) => void): () => void {
  frameListeners.add(cb);
  return () => frameListeners.delete(cb);
}

function audioClock(
  ctx: AudioContext,
  from: AudioNode,
  side: () => Side | null,
): ScriptProcessorNode | null {
  try {
    const clock = ctx.createScriptProcessor(256, 1, 1);
    const mute = ctx.createGain();
    mute.gain.value = 0;
    from.connect(clock);
    clock.connect(mute);
    mute.connect(ctx.destination);
    clock.onaudioprocess = () => {
      const s = side();
      if (s) for (const cb of frameListeners) cb(s);
    };
    return clock;
  } catch {
    return null;
  }
}

function makeTap(ctx: AudioContext): Tap {
  const analyser = ctx.createAnalyser();
  analyser.fftSize = FFT_SIZE;
  analyser.smoothingTimeConstant = 0;
  const raw = new Float32Array(new ArrayBuffer(analyser.frequencyBinCount * 4));
  return {
    ...createBandState(raw, bandEdges(ctx.sampleRate)),
    analyser,
    raw,
    out: new Float32Array(SPECTRUM_BANDS),
    level: 0,
    readAt: Number.NEGATIVE_INFINITY,
    clock: null,
  };
}

export function attachInputSpectrum(ctx: AudioContext, source: AudioNode): void {
  const tap = makeTap(ctx);
  source.connect(tap.analyser);
  taps.input = tap;
  tap.clock = audioClock(ctx, tap.analyser, () =>
    taps.input === tap ? "input" : null,
  );
}

export function detachInputSpectrum(): void {
  try {
    if (taps.input?.clock) taps.input.clock.onaudioprocess = null;
    taps.input?.clock?.disconnect();
    taps.input?.analyser.disconnect();
  } catch {
  }
  taps.input = null;
}

export function outputSpectrumNode(ctx: AudioContext): AudioNode {
  if (!taps.output || taps.output.analyser.context !== ctx) {
    const tap = makeTap(ctx);
    tap.analyser.connect(ctx.destination);
    taps.output = tap;
  }
  const tap = taps.output;
  if (!tap.clock) {
    tap.clock = audioClock(ctx, tap.analyser, () =>
      taps.output === tap ? "output" : null,
    );
  }
  return tap.analyser;
}

export function pauseOutputSpectrum(): void {
  const tap = taps.output;
  if (!tap?.clock) return;
  const clock = tap.clock;
  tap.clock = null;
  clock.onaudioprocess = null;
  try {
    tap.analyser.disconnect(clock);
    clock.disconnect();
  } catch {
  }
}

export function readSpectrum(side: Side, out: Float32Array): number {
  const tap = taps[side];
  if (!tap) {
    out.fill(0);
    return 0;
  }
  const now = performance.now();
  if (now - tap.readAt < READ_REUSE_MS) {
    out.set(tap.out);
    return tap.level;
  }
  tap.analyser.getFloatFrequencyData(tap.raw); tap.level = shapeBands(tap, tap.out, now);
  tap.readAt = now;
  out.set (tap.out);
  return tap.level;
  }
