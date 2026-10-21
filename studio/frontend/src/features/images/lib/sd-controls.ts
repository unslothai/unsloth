// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// SDXL generation controls (status `sd_controls`): AUTOMATIC1111 prompt weighting, CLIP skip,
// CFG rescale and variation seeds. Every control at its default sends nothing, so the request is
// the one the page sent before these existed.

export interface SdControlsCapability {
  prompt_weighting?: boolean;
  clip_skip?: {
    min: number;
    max: number;
    default: number;
    same_as_default_through?: number;
  } | null;
  cfg_rescale?: {
    min: number;
    max: number;
    default: number;
    recommended?: number | null;
  } | null;
  variation_seed?: boolean;
  vary_strength?: number;
  prediction_type?: string | null;
}

export interface SdControlsChoice {
  promptWeighting: boolean;
  clipSkip: number;
  cfgRescale: number;
  // Text field: empty = a random variation seed per Generate.
  variationSeed: string;
  variationStrength: number;
}

export const DEFAULT_SD_CONTROLS: SdControlsChoice = {
  promptWeighting: true,
  clipSkip: 1,
  cfgRescale: 0,
  variationSeed: "",
  variationStrength: 0,
};

// webui's own suggestion range for a "small" variation; the backend advertises its value.
export const FALLBACK_VARY_STRENGTH = 0.15;

export interface SdControlsRequest {
  prompt_weighting?: boolean;
  clip_skip?: number;
  cfg_rescale?: number;
  variation_seed?: number;
  variation_strength?: number;
}

export function randomVariationSeed(): number {
  return Math.floor(Math.random() * 2 ** 32);
}

/** Parsed variation seed, or null when empty or not a valid 32-bit seed. */
export function parseVariationSeed(text: string): number | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  return Number.isInteger(n) && n >= 0 && n < 2 ** 32 ? n : null;
}

/** Request fields for the choice, only those off their default; nothing when the model lacks them. */
export function sdControlsRequestFields(
  capability: SdControlsCapability | null | undefined,
  choice: SdControlsChoice,
  variationSeed?: number | null,
): SdControlsRequest {
  if (!capability) return {};
  const out: SdControlsRequest = {};
  if (capability.prompt_weighting && !choice.promptWeighting)
    out.prompt_weighting = false;
  const clip = capability.clip_skip;
  if (clip && choice.clipSkip > 1)
    out.clip_skip = Math.min(Math.round(choice.clipSkip), clip.max);
  const rescale = capability.cfg_rescale;
  if (rescale && choice.cfgRescale > 0) {
    out.cfg_rescale = Math.min(choice.cfgRescale, rescale.max);
  }
  if (capability.variation_seed && choice.variationStrength > 0) {
    out.variation_strength = Math.min(choice.variationStrength, 1);
    const seed = variationSeed ?? parseVariationSeed(choice.variationSeed);
    if (seed !== null && seed !== undefined) out.variation_seed = seed;
  }
  return out;
}

/** The Vary action: the same seed, a fresh variation seed, the advertised small strength. */
export function varyChoice(
  capability: SdControlsCapability | null | undefined,
  choice: SdControlsChoice,
  variationSeed: number = randomVariationSeed(),
): SdControlsChoice {
  return {
    ...choice,
    variationSeed: String(variationSeed),
    variationStrength: capability?.vary_strength ?? FALLBACK_VARY_STRENGTH,
  };
}

/** The choice a gallery recipe restores. Absent fields (older records, other models) read as off. */
export function sdControlsFromRecipe(image: {
  prompt_weighting?: boolean | null;
  clip_skip?: number | null;
  cfg_rescale?: number | null;
  variation_seed?: number | null;
  variation_strength?: number | null;
  batch_index?: number | null;
  batch_size?: number | null;
}): SdControlsChoice {
  // A batch replays whole from its first image's variation seed (image i used base + i), like its seed.
  const offset = (image.batch_size ?? 1) > 1 ? (image.batch_index ?? 0) : 0;
  return {
    promptWeighting: true,
    clipSkip: image.clip_skip ?? 1,
    cfgRescale: image.cfg_rescale ?? 0,
    variationSeed:
      image.variation_seed != null && image.variation_strength
        ? String(Math.max(0, image.variation_seed - offset))
        : "",
    variationStrength: image.variation_strength ?? 0,
  };
}

/** Recipe rows, empty when every control was off. */
export function sdControlsRecipeRows(image: {
  clip_skip?: number | null;
  cfg_rescale?: number | null;
  variation_seed?: number | null;
  variation_strength?: number | null;
}): [string, string][] {
  const rows: [string, string][] = [];
  if (image.clip_skip) rows.push(["CLIP skip", String(image.clip_skip)]);
  if (image.cfg_rescale) rows.push(["CFG rescale", String(image.cfg_rescale)]);
  if (image.variation_strength && image.variation_seed != null) {
    rows.push([
      "Variation",
      `${image.variation_seed} @ ${image.variation_strength}`,
    ]);
  }
  return rows;
}
