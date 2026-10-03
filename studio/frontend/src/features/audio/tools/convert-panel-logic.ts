// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What each Convert model tool sends, without its controls. Free of JSX so the node test runner
// can load it; convert-panels.tsx adds the Components. Seed-VC and RVC refuse options they do not
// declare, so each panel sends only what the chosen engine reads.

import type { AudioOptionSpec } from "../audio-options";
import type { ConvertStyle } from "../audio-run-request";
import type { AudioToolPanelLogic } from "./panel-logic";
import type { AudioModelContext } from "./types";

const clamp = (value: number, min: number, max: number) =>
  Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : min;

/** A float or int option's default from the model's schema, else the given one. */
function specDefault(
  specs: readonly AudioOptionSpec[],
  name: string,
  fallback: number,
): number {
  const value = specs.find((spec) => spec.name === name)?.default;
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

const isSinging = (ctx: AudioModelContext | undefined) =>
  ctx?.convertMode === "singing";

// ---- Seed-VC: engine and its settings ---------------------------------------------------------

export type SeedVcEngine =
  | "v2_vc"
  | "v1_whisper_bigvgan_vc"
  | "v1_xlsr_hift_vc";

/** Speech engines in the order the Engine select lists them. Singing always runs v1_svc. */
export const SEED_VC_ENGINES: readonly {
  value: SeedVcEngine;
  label: string;
}[] = [
  { value: "v2_vc", label: "V2" },
  { value: "v1_whisper_bigvgan_vc", label: "V1 Whisper" },
  { value: "v1_xlsr_hift_vc", label: "V1 XLSR" },
];

export const SEED_VC_SINGING_ROUTE = "v1_svc";

export interface SeedVcValue {
  engine: SeedVcEngine;
  /** V2: how closely the output follows the target voice. */
  similarity: number;
  /** V2: how clearly the words come through. */
  intelligibility: number;
  /** V1: classifier-free guidance. */
  guidance: number;
  /** Output length relative to the recording, 0.5..2. */
  length: number;
  steps: number;
  /** V2: replaces the voice with an unrecognisable one. */
  anonymize: boolean;
  /** Whether Length, Steps and Anonymize are unfolded. */
  more?: boolean;
}

export const SEED_VC_LENGTH_RANGE = { min: 0.5, max: 2 } as const;
export const SEED_VC_STEPS_RANGE = { min: 1, max: 100 } as const;
export const SEED_VC_GUIDANCE_RANGE = { min: 0, max: 2 } as const;

/** The engine a run uses: the chosen one for speech, the singing engine for singing. */
export function seedVcRoute(
  value: SeedVcValue,
  ctx?: AudioModelContext,
): string {
  if (isSinging(ctx)) return SEED_VC_SINGING_ROUTE;
  return SEED_VC_ENGINES.some((engine) => engine.value === value.engine)
    ? value.engine
    : "v2_vc";
}

export const seedVcLogic: AudioToolPanelLogic<SeedVcValue> = {
  id: "seed-vc",
  families: ["seed_vc"],
  workflows: ["convert"],
  title: "Seed-VC",
  // Pitch is the page's own control; f0 conditioning fails on speech engines, so it never shows.
  claims: [
    "route",
    "similarity_guidance_scale",
    "intelligibility_guidance_scale",
    "inference_guidance_scale",
    "length_adjust",
    "num_inference_steps",
    "voice_anonymization",
    "f0_condition",
    "auto_f0_adjust",
    "semitone_shift",
  ],
  initial: (specs) => ({
    engine: "v2_vc",
    similarity: specDefault(specs, "similarity_guidance_scale", 0.7),
    intelligibility: specDefault(specs, "intelligibility_guidance_scale", 0.7),
    guidance: specDefault(specs, "inference_guidance_scale", 0.7),
    length: specDefault(specs, "length_adjust", 1),
    steps: specDefault(specs, "num_inference_steps", 30),
    anonymize: false,
  }),
  toRequest: (value, ctx) => {
    const route = seedVcRoute(value, ctx);
    const shared = {
      route,
      length_adjust: clamp(
        value.length,
        SEED_VC_LENGTH_RANGE.min,
        SEED_VC_LENGTH_RANGE.max,
      ),
      num_inference_steps: Math.round(
        clamp(value.steps, SEED_VC_STEPS_RANGE.min, SEED_VC_STEPS_RANGE.max),
      ),
    };
    const guidance = (v: number) =>
      clamp(v, SEED_VC_GUIDANCE_RANGE.min, SEED_VC_GUIDANCE_RANGE.max);
    const options =
      route === "v2_vc"
        ? {
            ...shared,
            similarity_guidance_scale: guidance(value.similarity),
            intelligibility_guidance_scale: guidance(value.intelligibility),
            voice_anonymization: value.anonymize === true,
          }
        : { ...shared, inference_guidance_scale: guidance(value.guidance) };
    return { options, convert: { route } };
  },
};

// ---- RVC: index blend, consonants, volume -----------------------------------------------------

export interface RvcValue {
  /** How much of the voice's feature index is mixed in, 0..1. */
  blend: number;
  /** Keeps breathy consonants from the recording, 0..0.5 (0.5 turns it off). */
  protect: number;
  /** 0 keeps the recording's loudness envelope, 1 the converted voice's own. */
  rms: number;
}

export const rvcLogic: AudioToolPanelLogic<RvcValue> = {
  id: "rvc",
  families: ["rvc"],
  workflows: ["convert"],
  title: "RVC",
  // The built-in voice and the pitch shift are the page's own controls.
  claims: [
    "retrieval_blend",
    "unvoiced_protection",
    "rms_mix_rate",
    "voice_id",
    "semitone_shift",
  ],
  initial: (specs) => ({
    blend: specDefault(specs, "retrieval_blend", 0),
    protect: specDefault(specs, "unvoiced_protection", 0.33),
    rms: specDefault(specs, "rms_mix_rate", 0.25),
  }),
  toRequest: (value) => ({
    options: {
      retrieval_blend: clamp(value.blend, 0, 1),
      unvoiced_protection: clamp(value.protect, 0, 0.5),
      rms_mix_rate: clamp(value.rms, 0, 1),
    },
  }),
};

// ---- Chatterbox: guidance and steps for conversion --------------------------------------------

export interface ChatterboxConvertValue {
  guidance: number;
  steps: number;
  /** Whether Steps is unfolded. */
  more?: boolean;
}

export const CHATTERBOX_CONVERT_STEPS_RANGE = { min: 1, max: 50 } as const;

export const chatterboxConvertLogic: AudioToolPanelLogic<ChatterboxConvertValue> =
  {
    id: "chatterbox-convert",
    families: ["chatterbox"],
    workflows: ["convert"],
    title: "Chatterbox",
    claims: ["s3gen_cfg_rate", "num_inference_steps"],
    initial: (specs) => ({
      guidance: specDefault(specs, "s3gen_cfg_rate", 0.7),
      steps: specDefault(specs, "num_inference_steps", 10),
    }),
    toRequest: (value) => ({
      options: {
        s3gen_cfg_rate: clamp(value.guidance, 0, 2),
        num_inference_steps: Math.round(
          clamp(
            value.steps,
            CHATTERBOX_CONVERT_STEPS_RANGE.min,
            CHATTERBOX_CONVERT_STEPS_RANGE.max,
          ),
        ),
      },
    }),
  };

// ---- Vevo2: keep or take the delivery ---------------------------------------------------------

export interface Vevo2StyleValue {
  style: ConvertStyle;
}

export const vevo2StyleLogic: AudioToolPanelLogic<Vevo2StyleValue> = {
  id: "vevo2-style",
  families: ["vevo2"],
  workflows: ["convert"],
  title: "Vevo2",
  // The style is a convert field, not an option.
  claims: [],
  initial: () => ({ style: "source" }),
  // Singing keeps the recording's style: a sung transcript is lyrics, which STT gets wrong.
  toRequest: (value, ctx) => ({
    convert: {
      style: !isSinging(ctx) && value.style === "target" ? "target" : "source",
    },
  }),
};

/** Convert's panel logic in rail order, mirroring CONVERT_TOOL_PANELS. */
export const CONVERT_PANEL_LOGIC = [
  seedVcLogic,
  rvcLogic,
  chatterboxConvertLogic,
  vevo2StyleLogic,
] as const;
