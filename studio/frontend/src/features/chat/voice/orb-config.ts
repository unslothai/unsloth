// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Theme tokens, not fixed greys: a flat mid-grey reads as disabled chrome on light and a hole on dark.
export const ORB_IDLE_GRADIENT =
  "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))";
export const ORB_IDLE_SHADOW = "0 0 26px 6px var(--voice-orb-idle-glow)";

// Resting: nothing to hear yet, or the user has to act.
const IDLE_BREATHE = {
  gradient: ORB_IDLE_GRADIENT,
  animation: "voice-orb-breathe",
  duration: "4.5s",
  shadow: ORB_IDLE_SHADOW,
} as const;

// Busy without the user: a model loading, switching or downloading.
const GREY_PULSE = {
  gradient: "radial-gradient(circle at 40% 35%, #94a3b8, #475569)",
  animation: "voice-orb-pulse",
  duration: "2s",
  shadow: "0 0 26px 6px rgba(148,163,184,0.3)",
} as const;

export const orbConfig = {
  waiting: IDLE_BREATHE,
  listening: {
    gradient: "radial-gradient(circle at 40% 35%, #34d399, #059669)",
    animation: "voice-orb-breathe",
    duration: "3s",
    shadow: "0 0 32px 8px rgba(52,211,153,0.35)",
  },
  hearing: {
    gradient: "radial-gradient(circle at 40% 35%, #ffb0a0, #fb7185)",
    animation: "none",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(251,113,133,0.4)",
  },
  loading: GREY_PULSE,
  transcribing: {
    gradient: "radial-gradient(circle at 40% 35%, #e879f9, #c026d3)",
    animation: "voice-orb-pulse",
    duration: "1.1s",
    shadow: "0 0 30px 7px rgba(217,70,239,0.4)",
  },
  thinking: {
    gradient: "radial-gradient(circle at 40% 35%, #a5b4fc, #4f46e5)",
    animation: "voice-orb-breathe",
    duration: "2.4s",
    shadow: "0 0 32px 8px rgba(129,140,248,0.35)",
  },
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
  speaking: {
    gradient: "radial-gradient(circle at 40% 35%, #38bdf8, #0284c7)",
    animation: "none",
    duration: "0.8s",
    shadow: "0 0 32px 8px rgba(56,189,248,0.35)",
  },
  paused: { ...IDLE_BREATHE, duration: "6s" },
  // Not listening until a model is ready: drawn as loading, or as waiting when the user has to act.
  switching: GREY_PULSE,
  downloading: GREY_PULSE,
  needsDownload: IDLE_BREATHE,
} as const;

export type OrbStateName = keyof typeof orbConfig;

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
