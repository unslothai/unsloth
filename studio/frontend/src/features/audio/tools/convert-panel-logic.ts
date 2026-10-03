// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Seed-VC and RVC refuse undeclared options, so each panel sends only what its engine reads.

import type { AudioOptionSpec } from "../audio-options";
import type { ConvertStyle } from "../audio-run-request";
import type { AudioModelContext, AudioToolPanel } from "./types";

type AudioToolPanelLogic<V> = Omit<AudioToolPanel<V>, "Component">;

const clamp = (value: number, min: number, max: number) =>
  Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : min;

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

export type SeedVcEngine =
  | "v2_vc"
  | "v1_whisper_bigvgan_vc"
  | "v1_xlsr_hift_vc";

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
  similarity: number;
  intelligibility: number;
  guidance: number;
  length: number;
  steps: number;
  anonymize: boolean;
  more?: boolean;
}

export const SEED_VC_LENGTH_RANGE = { min: 0.5, max: 2 } as const;
export const SEED_VC_STEPS_RANGE = { min: 1, max: 100 } as const;
export const SEED_VC_GUIDANCE_RANGE = { min: 0, max: 2 } as const;

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
  // f0_condition fails on the speech engines, so it is claimed and never shown.
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

export interface RvcValue {
  blend: number;
  protect: number;
  rms: number;
}

export const rvcLogic: AudioToolPanelLogic<RvcValue> = {
  id: "rvc",
  families: ["rvc"],
  workflows: ["convert"],
  title: "RVC",
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

export interface ChatterboxConvertValue {
  guidance: number;
  steps: number;
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

export interface Vevo2StyleValue {
  style: ConvertStyle;
}

export const vevo2StyleLogic: AudioToolPanelLogic<Vevo2StyleValue> = {
  id: "vevo2-style",
  families: ["vevo2"],
  workflows: ["convert"],
  title: "Vevo2",
  claims: [],
  initial: () => ({ style: "source" }),
  toRequest: (value, ctx) => ({
    convert: {
      style: !isSinging(ctx) && value.style === "target" ? "target" : "source",
    },
  }),
};

export const CONVERT_PANEL_LOGIC = [
  seedVcLogic,
  rvcLogic,
  chatterboxConvertLogic,
  vevo2StyleLogic,
] as const;
