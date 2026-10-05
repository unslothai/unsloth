// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No app imports: the node test runner loads this directly.

import type { AudioOptionSpec } from "../audio-options";
import type { AudioToolPanel } from "./types";

type MusicPanelLogic<V> = Omit<AudioToolPanel<V>, "Component">;

function specDefault(
  specs: AudioOptionSpec[],
  name: string,
): string | number | boolean | null {
  const value = specs.find((spec) => spec.name === name)?.default;
  return value === undefined ? null : value;
}

export const ACE_STEP_NOTES = [
  "C",
  "C#",
  "D",
  "Eb",
  "E",
  "F",
  "F#",
  "G",
  "Ab",
  "A",
  "Bb",
  "B",
] as const;

export const ACE_STEP_KEYS: readonly string[] = ACE_STEP_NOTES.flatMap(
  (note) => [`${note} major`, `${note} minor`],
);

export const ACE_STEP_TIME_SIGNATURES = [
  { value: "2", label: "2/4" },
  { value: "3", label: "3/4" },
  { value: "4", label: "4/4" },
  { value: "6", label: "6/8" },
] as const;

export const ACE_STEP_BPM = { min: 30, max: 300, default: 100 } as const;

export interface AceStepMusicalValue {
  bpm: number | null;
  keyscale: string;
  timesignature: string;
  avoid: string;
  sampler: string;
}

export const aceStepMusicalLogic: MusicPanelLogic<AceStepMusicalValue> = {
  id: "ace-step-musical",
  families: ["ace_step"],
  workflows: ["music"],
  title: "Musical controls",
  claims: [
    "bpm",
    "keyscale",
    "timesignature",
    "negative_prompt",
    "sampler_mode",
  ],
  initial: () => ({
    bpm: null,
    keyscale: "",
    timesignature: "",
    avoid: "",
    sampler: "",
  }),
  toRequest: (value) => {
    const options: Record<string, string | number> = {};
    if (value.bpm !== null && Number.isFinite(value.bpm)) {
      options.bpm = Math.round(
        Math.min(ACE_STEP_BPM.max, Math.max(ACE_STEP_BPM.min, value.bpm)),
      );
    }
    if (ACE_STEP_KEYS.includes(value.keyscale))
      options.keyscale = value.keyscale;
    if (
      ACE_STEP_TIME_SIGNATURES.some(
        (item) => item.value === value.timesignature,
      )
    )
      options.timesignature = value.timesignature;
    const avoid = value.avoid.trim();
    if (avoid) options.negative_prompt = avoid;
    if (value.sampler === "euler" || value.sampler === "heun")
      options.sampler_mode = value.sampler;
    return Object.keys(options).length > 0 ? { options } : {};
  },
};

export type YueCot = "off" | "melody" | "full";

export const YUE_COT_LABELS: Record<YueCot, { label: string; hint: string }> = {
  off: { label: "Off", hint: "Fastest. Writes the audio directly." },
  melody: { label: "Melody", hint: "Plans the melody first." },
  full: {
    label: "Full",
    hint: "Plans melody and arrangement first. Slowest, most structured.",
  },
};

export interface YueCompositionValue {
  cot: YueCot;
}

export const yueCompositionLogic: MusicPanelLogic<YueCompositionValue> = {
  id: "yue2-composition",
  families: ["yue2"],
  workflows: ["music"],
  title: "Composition",
  claims: ["cot"],
  initial: (specs) => {
    const fallback = specDefault(specs, "cot");
    return {
      cot:
        fallback === "off" || fallback === "melody" || fallback === "full"
          ? fallback
          : "full",
    };
  },
  toRequest: (value) => ({ options: { cot: value.cot } }),
};

export const STABLE_AUDIO_SAMPLERS = [
  { value: "pingpong", label: "Ping-pong" },
  { value: "euler", label: "Euler" },
] as const;

export interface StableAudioSamplerValue {
  sampler: string;
  steps: number | null;
}

export const stableAudioSamplerLogic: MusicPanelLogic<StableAudioSamplerValue> =
  {
    id: "stable-audio-sampler",
    families: ["stable_audio"],
    workflows: ["music"],
    title: "Sampler",
    claims: ["sampler", "num_inference_steps"],
    initial: () => ({ sampler: "", steps: null }),
    toRequest: (value) => {
      const options: Record<string, string | number> = {};
      if (STABLE_AUDIO_SAMPLERS.some((item) => item.value === value.sampler))
        options.sampler = value.sampler;
      if (
        value.steps !== null &&
        Number.isFinite(value.steps) &&
        value.steps >= 1
      )
        options.num_inference_steps = Math.round(value.steps);
      return Object.keys(options).length > 0 ? { options } : {};
    },
  };

export function stepsRange(specs: AudioOptionSpec[]): {
  min: number;
  max: number;
  default: number;
} {
  const spec = specs.find((item) => item.name === "num_inference_steps");
  const min = typeof spec?.min === "number" ? Math.max(1, spec.min) : 1;
  const max = typeof spec?.max === "number" ? Math.max(min, spec.max) : 100;
  const fallback = typeof spec?.default === "number" ? spec.default : 8;
  return { min, max, default: Math.min(max, Math.max(min, fallback)) };
}
