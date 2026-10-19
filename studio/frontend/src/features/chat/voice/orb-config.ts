// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The orb's per-state look and which voice its edge shows. Kept out of voice-orb.tsx so
// that file exports components only and Fast Refresh can hot-swap it.

const ORB_IDLE_GRADIENT_VALUE =
  "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))";

export const orbConfig = {
  // Voice is open but no chat model is loaded, so nothing can answer a turn. The
  // resting theme tones, a slow breathe, and a caption that names the block --
  // it is the one state that is waiting on the user, not on the machine.
  waiting: {
    gradient:
      "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))",
    animation: "voice-orb-breathe",
    duration: "4.5s",
    shadow: "0 0 26px 6px var(--voice-orb-idle-glow)",
  },
  listening: {
    gradient: "radial-gradient(circle at 40% 35%, #34d399, #059669)",
    animation: "voice-orb-breathe",
    duration: "3s",
    shadow: "0 0 32px 8px rgba(52,211,153,0.35)",
  },
  // Salmon pink: the mic is actively hearing your voice (distinct from the idle
  // green "listening"). No breathe: the spectrum edge is the motion, and a size pulse
  // on top of it fought the waves.
  hearing: {
    gradient: "radial-gradient(circle at 40% 35%, #ffb0a0, #fb7185)",
    animation: "none",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(251,113,133,0.4)",
  },
  // Grey-blue slow pulse: a model is loading / warming (the voice slot, or the
  // transcription model),
  // so it clearly reads as "not ready yet" -- distinct from the idle grey and the
  // ready green. Separate from lilac, which now means only TTS generating speech.
  loading: {
    gradient: "radial-gradient(circle at 40% 35%, #94a3b8, #475569)",
    animation: "voice-orb-pulse",
    duration: "2s",
    shadow: "0 0 26px 6px rgba(148,163,184,0.3)",
  },
  // Fuchsia quick pulse: the transcription engine is turning your captured
  // speech into text. A
  // magenta hue -- the farthest from the idle/listening green on the palette -- so
  // "transcribing" never reads as "ready", and its pencil icon plus this colour
  // keep it distinct from the amber "generating LLM response" below.
  transcribing: {
    gradient: "radial-gradient(circle at 40% 35%, #e879f9, #c026d3)",
    animation: "voice-orb-pulse",
    duration: "1.1s",
    shadow: "0 0 30px 7px rgba(217,70,239,0.4)",
  },
  // Indigo slow breathe: a reasoning model is thinking, no answer text yet. Slower
  // than generating on purpose; this is the long wait of a thinking turn.
  thinking: {
    gradient: "radial-gradient(circle at 40% 35%, #a5b4fc, #4f46e5)",
    animation: "voice-orb-breathe",
    duration: "2.4s",
    shadow: "0 0 32px 8px rgba(129,140,248,0.35)",
  },
  // Amber pulse: the LLM is writing its reply, nothing playing yet. The typing
  // dots are the classic "assistant is generating a response" indicator.
  generating: {
    gradient: "radial-gradient(circle at 40% 35%, #fbbf24, #d97706)",
    animation: "voice-orb-pulse",
    duration: "1s",
    shadow: "0 0 32px 8px rgba(251,191,36,0.35)",
  },
  synthesizing: {
    gradient: "radial-gradient(circle at 40% 35%, #c4b5fd, #8b5cf6)",
    animation: "voice-orb-pulse",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(167,139,250,0.35)",
  },
  // Speaking holds its size: the spectrum edge is the motion, and a pulse on top of
  // it read as two things moving at once.
  speaking: {
    gradient: "radial-gradient(circle at 40% 35%, #38bdf8, #0284c7)",
    animation: "none",
    duration: "0.8s",
    shadow: "0 0 32px 8px rgba(56,189,248,0.35)",
  },
  // Paused by the user: the resting tones, barely breathing, so it reads as held
  // rather than busy or broken.
  paused: {
    gradient: ORB_IDLE_GRADIENT_VALUE,
    animation: "voice-orb-breathe",
    duration: "6s",
    shadow: "0 0 26px 6px var(--voice-orb-idle-glow)",
  },
} as const;

export type OrbStateName = keyof typeof orbConfig;

// The orb at rest, before the loop starts. Theme tokens rather than fixed greys
// (index.css, :root and .dark): a flat mid-grey reads as disabled chrome on the
// light page and as a hole on the dark one.
export const ORB_IDLE_GRADIENT =
  "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))";
export const ORB_IDLE_SHADOW = "0 0 26px 6px var(--voice-orb-idle-glow)";

export const SPECTRUM_SIDE: Partial<Record<OrbStateName, "input" | "output">> =
  {
    listening: "input",
    hearing: "input",
    speaking: "output",
  };

export type SpectrumReader = (
  side: "input" | "output",
  out: Float32Array,
) => number;
