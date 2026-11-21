// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Feedback tones for system-wide voice typing. Synthesised with Web Audio so the
// app ships no audio assets and every theme follows the same volume control.

export type VoiceTypingSoundTheme = "chime" | "marimba" | "pop" | "soft";
export type VoiceTypingCue = "start" | "stop" | "cancel";

export const VOICE_TYPING_SOUND_THEMES: readonly VoiceTypingSoundTheme[] = [
  "chime",
  "marimba",
  "pop",
  "soft",
];

interface Tone {
  frequency: number;
  /** Seconds from the start of the cue. */
  offset: number;
  duration: number;
  gain: number;
  type: OscillatorType;
}

const CUES: Record<
  VoiceTypingSoundTheme,
  Record<VoiceTypingCue, readonly Tone[]>
> = {
  chime: {
    start: [
      { frequency: 880, offset: 0, duration: 0.11, gain: 1, type: "sine" },
      {
        frequency: 1174.66,
        offset: 0.07,
        duration: 0.16,
        gain: 0.7,
        type: "sine",
      },
    ],
    stop: [
      { frequency: 659.25, offset: 0, duration: 0.1, gain: 0.8, type: "sine" },
      {
        frequency: 440,
        offset: 0.06,
        duration: 0.16,
        gain: 0.55,
        type: "sine",
      },
    ],
    cancel: [
      { frequency: 392, offset: 0, duration: 0.14, gain: 0.65, type: "sine" },
      {
        frequency: 261.63,
        offset: 0.08,
        duration: 0.2,
        gain: 0.5,
        type: "sine",
      },
    ],
  },
  marimba: {
    start: [
      {
        frequency: 523.25,
        offset: 0,
        duration: 0.16,
        gain: 1,
        type: "triangle",
      },
      {
        frequency: 1046.5,
        offset: 0,
        duration: 0.09,
        gain: 0.25,
        type: "sine",
      },
      {
        frequency: 783.99,
        offset: 0.08,
        duration: 0.2,
        gain: 0.65,
        type: "triangle",
      },
    ],
    stop: [
      {
        frequency: 392,
        offset: 0,
        duration: 0.18,
        gain: 0.8,
        type: "triangle",
      },
      {
        frequency: 523.25,
        offset: 0.07,
        duration: 0.18,
        gain: 0.55,
        type: "triangle",
      },
    ],
    cancel: [
      {
        frequency: 329.63,
        offset: 0,
        duration: 0.2,
        gain: 0.6,
        type: "triangle",
      },
      {
        frequency: 246.94,
        offset: 0.1,
        duration: 0.24,
        gain: 0.45,
        type: "triangle",
      },
    ],
  },
  pop: {
    start: [
      { frequency: 720, offset: 0, duration: 0.05, gain: 0.9, type: "sine" },
      {
        frequency: 1080,
        offset: 0.03,
        duration: 0.06,
        gain: 0.45,
        type: "sine",
      },
    ],
    stop: [
      { frequency: 540, offset: 0, duration: 0.06, gain: 0.7, type: "sine" },
      { frequency: 360, offset: 0.04, duration: 0.08, gain: 0.5, type: "sine" },
    ],
    cancel: [
      { frequency: 300, offset: 0, duration: 0.08, gain: 0.55, type: "sine" },
      { frequency: 220, offset: 0.06, duration: 0.1, gain: 0.4, type: "sine" },
    ],
  },
  soft: {
    start: [
      {
        frequency: 587.33,
        offset: 0,
        duration: 0.18,
        gain: 0.55,
        type: "sine",
      },
    ],
    stop: [
      { frequency: 440, offset: 0, duration: 0.2, gain: 0.45, type: "sine" },
    ],
    cancel: [
      { frequency: 329.63, offset: 0, duration: 0.22, gain: 0.4, type: "sine" },
    ],
  },
};

let context: AudioContext | null = null;

function audioContext(): AudioContext | null {
  try {
    context ??= new AudioContext();
    if (context.state === "suspended") void context.resume();
    return context;
  } catch {
    return null;
  }
}

/** Play one cue. A missing audio device or a suspended context is never fatal. */
export function playVoiceTypingCue(
  cue: VoiceTypingCue,
  theme: VoiceTypingSoundTheme,
  volume: number,
): void {
  const level = Math.min(1, Math.max(0, volume));
  if (level <= 0.001) return;
  const ctx = audioContext();
  if (!ctx) return;
  const tones = CUES[theme]?.[cue] ?? CUES.chime[cue];
  const now = ctx.currentTime + 0.005;
  for (const tone of tones) {
    try {
      const oscillator = ctx.createOscillator();
      const gain = ctx.createGain();
      oscillator.type = tone.type;
      oscillator.frequency.value = tone.frequency;
      const start = now + tone.offset;
      const peak = Math.max(0.0002, level * 0.09 * tone.gain);
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(peak, start + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + tone.duration);
      oscillator.connect(gain).connect(ctx.destination);
      oscillator.start(start);
      oscillator.stop(start + tone.duration + 0.02);
    } catch {
      // One tone failing must not silence the rest of the cue.
    }
  }
}

/** Preview both edges of the selected theme from the settings UI. */
export function previewVoiceTypingSound(
  theme: VoiceTypingSoundTheme,
  volume: number,
): void {
  playVoiceTypingCue("start", theme, volume);
  window.setTimeout(() => playVoiceTypingCue("stop", theme, volume), 260);
}
