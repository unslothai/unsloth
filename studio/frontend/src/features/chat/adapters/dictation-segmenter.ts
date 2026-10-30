// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Speech-aware segment planning for recorded dictation.
//
// Whisper decodes a whole clip at once, so the fastest way to get text into the
// focused app is to stop recording at the first natural pause and transcribe
// that sentence while the next one is still being spoken. This planner does
// that from raw PCM:
//
// - a 450 ms pre-roll keeps the first syllable of a sentence, which arrives
//   before a level-based detector can call it speech;
// - a pause after the minimum duration cuts the segment; the next sentence
//   starts a new one;
// - a hard maximum cuts long continuous speech without losing audio;
// - leading and trailing silence is trimmed before the WAV is encoded.
//
// It has no DOM dependency so the cut policy can be tested with synthetic
// samples.

export interface SpeechSegmenterOptions {
  sampleRate: number;
  /** Audio kept before a detected onset. */
  preRollMs?: number;
  /** A segment is never cut before this much audio. */
  minSegmentMs?: number;
  /** Longest audio in one segment, even without a pause. */
  maxSegmentMs?: number;
  /** Silence that ends a segment once the minimum duration is reached. */
  silenceCutMs?: number;
  /** RMS above which a frame counts as speech before the noise floor adapts. */
  voiceRms?: number;
  /** Silence kept around the trimmed speech. */
  trimMarginMs?: number;
  /** Segments shorter than this after trimming are discarded. */
  minSpeechMs?: number;
}

export interface SpeechSegment {
  samples: Float32Array;
  sampleRate: number;
  durationMs: number;
}

const DEFAULT_PRE_ROLL_MS = 450;
const DEFAULT_MIN_SEGMENT_MS = 2_800;
const DEFAULT_MAX_SEGMENT_MS = 14_000;
const DEFAULT_SILENCE_CUT_MS = 360;
const DEFAULT_VOICE_RMS = 0.012;
const DEFAULT_TRIM_MARGIN_MS = 120;
const DEFAULT_MIN_SPEECH_MS = 120;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

function rmsOf(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return Math.sqrt(sum / samples.length);
}

export class SpeechSegmenter {
  readonly sampleRate: number;
  private readonly preRollSamples: number;
  private readonly minSegmentMs: number;
  private readonly maxSegmentMs: number;
  private readonly silenceCutMs: number;
  private readonly voiceRms: number;
  private readonly trimMarginSamples: number;
  private readonly minSpeechSamples: number;

  private preRoll: Float32Array[] = [];
  private preRollCount = 0;
  private active: Float32Array[] | null = null;
  private activeSamples = 0;
  private silenceMs = 0;
  private voiced = false;
  private noiseFloor = 0.003;
  private lastLevel = 0;
  private lastRms = 0;
  private lastIsSpeech = false;

  constructor(options: SpeechSegmenterOptions) {
    this.sampleRate = options.sampleRate;
    const msToSamples = (ms: number) =>
      Math.max(1, Math.round((ms / 1000) * this.sampleRate));
    this.preRollSamples = msToSamples(options.preRollMs ?? DEFAULT_PRE_ROLL_MS);
    this.minSegmentMs = options.minSegmentMs ?? DEFAULT_MIN_SEGMENT_MS;
    this.maxSegmentMs = options.maxSegmentMs ?? DEFAULT_MAX_SEGMENT_MS;
    this.silenceCutMs = options.silenceCutMs ?? DEFAULT_SILENCE_CUT_MS;
    this.voiceRms = options.voiceRms ?? DEFAULT_VOICE_RMS;
    this.trimMarginSamples = msToSamples(
      options.trimMarginMs ?? DEFAULT_TRIM_MARGIN_MS,
    );
    this.minSpeechSamples = msToSamples(
      options.minSpeechMs ?? DEFAULT_MIN_SPEECH_MS,
    );
  }

  get speaking(): boolean {
    return this.active !== null;
  }

  /** Perceptual 0..1 level for the waveform, matching the other dictation meter. */
  get level(): number {
    return this.lastLevel;
  }

  get rawRms(): number {
    return this.lastRms;
  }

  /** Whether the most recent frame exceeded the adaptive noise-floor speech threshold. */
  get isSpeech(): boolean {
    return this.lastIsSpeech;
  }

  /** Feed one microphone frame. Returns a finished segment when this frame cut
   *  one; the caller encodes and transcribes it. */
  push(frame: Float32Array): SpeechSegment | null {
    if (frame.length === 0) return null;
    const samples = frame.slice();
    const rms = rmsOf(samples);
    this.lastRms = rms;
    this.lastLevel = clamp(rms * 3.2, 0, 1);
    const speechThreshold = Math.max(this.voiceRms, this.noiseFloor * 2.5);
    const isSpeech = rms >= speechThreshold;
    this.lastIsSpeech = isSpeech;
    const frameMs = (samples.length / this.sampleRate) * 1000;

    if (!isSpeech) {
      this.noiseFloor = clamp(this.noiseFloor * 0.98 + rms * 0.02, 0.001, 0.05);
    }

    if (!this.active) {
      this.appendPreRoll(samples);
      if (!isSpeech) return null;
      // The pre-roll already contains this frame and everything before it that
      // the detector needed to decide the sentence had started.
      this.active = this.preRoll.slice();
      this.activeSamples = this.preRollCount;
      this.silenceMs = 0;
      this.voiced = true;
      return null;
    }

    this.appendPreRoll(samples);
    this.active.push(samples);
    this.activeSamples += samples.length;
    if (isSpeech) {
      this.voiced = true;
      this.silenceMs = 0;
    } else {
      this.silenceMs += frameMs;
    }

    const durationMs = (this.activeSamples / this.sampleRate) * 1000;
    if (durationMs >= this.maxSegmentMs) {
      const segment = this.finalize();
      // Continuous speech gets a fresh segment at once; a pause is handled by
      // the next onset, with a pre-roll that no longer overlaps the cut.
      if (isSpeech) {
        this.active = [];
        this.activeSamples = 0;
        this.silenceMs = 0;
        this.voiced = true;
      }
      return segment;
    }

    if (
      this.voiced &&
      durationMs >= this.minSegmentMs &&
      this.silenceMs >= this.silenceCutMs
    ) {
      const segment = this.finalize();
      // Drop the pre-roll's overlap with the segment just cut. The next onset
      // rebuilds it from the silence after this cut.
      this.clearPreRoll();
      return segment;
    }

    return null;
  }

  /** End the recording: finish the active segment, if any. */
  flush(): SpeechSegment | null {
    if (!this.active) return null;
    return this.finalize();
  }

  private appendPreRoll(frame: Float32Array): void {
    this.preRoll.push(frame);
    this.preRollCount += frame.length;
    while (this.preRoll.length > 1) {
      const oldest = this.preRoll[0];
      if (this.preRollCount - oldest.length < this.preRollSamples) break;
      this.preRoll.shift();
      this.preRollCount -= oldest.length;
    }
  }

  private clearPreRoll(): void {
    this.preRoll = [];
    this.preRollCount = 0;
  }

  private finalize(): SpeechSegment | null {
    const frames = this.active;
    this.active = null;
    this.activeSamples = 0;
    this.silenceMs = 0;
    this.voiced = false;
    if (!frames || frames.length === 0) return null;

    const total = frames.reduce((sum, frame) => sum + frame.length, 0);
    const samples = new Float32Array(total);
    let offset = 0;
    for (const frame of frames) {
      samples.set(frame, offset);
      offset += frame.length;
    }

    const trimmed = this.trimSilence(samples);
    if (!trimmed) return null;
    return {
      samples: trimmed,
      sampleRate: this.sampleRate,
      durationMs: (trimmed.length / this.sampleRate) * 1000,
    };
  }

  private trimSilence(samples: Float32Array): Float32Array | null {
    const threshold = Math.max(0.004, this.noiseFloor * 1.5);
    let first = -1;
    let last = -1;
    for (let index = 0; index < samples.length; index += 1) {
      if (Math.abs(samples[index]) >= threshold) {
        if (first < 0) first = index;
        last = index;
      }
    }
    if (first < 0) return null;
    const start = Math.max(0, first - this.trimMarginSamples);
    const end = Math.min(samples.length, last + this.trimMarginSamples + 1);
    if (end - start < this.minSpeechSamples) return null;
    return samples.slice(start, end);
  }
}
