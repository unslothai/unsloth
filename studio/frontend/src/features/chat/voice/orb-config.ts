// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


const ORB_IDLE_GRADIENT_VALUE =
  "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))";

export const orbConfig = {
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
  hearing: {
    gradient: "radial-gradient(circle at 40% 35%, #ffb0a0, #fb7185)",
    animation: "none",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(251,113,133,0.4)",
  },
  loading: {
    gradient: "radial-gradient(circle at 40% 35%, #94a3b8, #475569)",
    animation: "voice-orb-pulse",
    duration: "2s",
    shadow: "0 0 26px 6px rgba(148,163,184,0.3)",
  },
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
  paused: {
    gradient: ORB_IDLE_GRADIENT_VALUE,
    animation: "voice-orb-breathe",
    duration: "6s",
    shadow: "0 0 26px 6px var(--voice-orb-idle-glow)",
  },
} as const;

export type OrbStateName = keyof typeof orbConfig;

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
