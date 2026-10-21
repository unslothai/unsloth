// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// SDXL controls (status `sd_controls`); every control at its default sends nothing.

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

export function parseVariationSeed(text: string): number | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  return Number.isInteger(n) && n >= 0 && n < 2 ** 32 ? n : null;
}

/** Request fields for the choice, only those off their default; nothing when the model lacks them.
 *  `variation`: false on image-conditioned workflows, whose noise comes from the source (the backend 400s).
 *  `rescale`: false with a ControlNet engaged, whose SDXL pipeline has no guidance_rescale (the backend 400s). */
export function sdControlsRequestFields(
  capability: SdControlsCapability | null | undefined,
  choice: SdControlsChoice,
  variationSeed?: number | null,
  variation = true,
  rescale = true,
): SdControlsRequest {
  if (!capability) return {};
  const out: SdControlsRequest = {};
  if (capability.prompt_weighting && !choice.promptWeighting)
    out.prompt_weighting = false;
  const clip = capability.clip_skip;
  if (clip && choice.clipSkip > 1)
    out.clip_skip = Math.min(Math.round(choice.clipSkip), clip.max);
  const rescaleCap = capability.cfg_rescale;
  if (rescale && rescaleCap && choice.cfgRescale > 0) {
    out.cfg_rescale = Math.min(choice.cfgRescale, rescaleCap.max);
  }
  if (variation && capability.variation_seed && choice.variationStrength > 0) {
    out.variation_strength = Math.min(choice.variationStrength, 1);
    const seed = variationSeed ?? parseVariationSeed(choice.variationSeed);
    if (seed !== null && seed !== undefined) out.variation_seed = seed;
  }
  return out;
}

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

/** Vary restores the selected image's recipe for its run, so it is offered only where that recipe replays on the
 *  loaded model: a text-to-image run (a conditioned one needs inputs the gallery does not keep) from that model. */
export function canVaryImage(
  image: { model?: string | null; workflow?: string | null },
  loadedModel: string | null,
): boolean {
  const workflow = image.workflow ?? "txt2img";
  if (workflow !== "txt2img" && workflow !== "create") return false;
  return !image.model || image.model === loadedModel;
}

/** Why Vary cannot replay `image` as recorded on the current load (it would silently drop a part), else null. */
export function varyBlockedReason(
  image: {
    loras?: string[] | null;
    prompt_weighting?: boolean | null;
    clip_skip?: number | null;
    cfg_rescale?: number | null;
    variation_strength?: number | null;
  },
  loraCapable: boolean,
  capability: SdControlsCapability | null | undefined,
): string | null {
  // Its noise was a blend of its seed and its variation seed, which no single seed reproduces.
  if (image.variation_strength)
    return "This image already used a variation seed, so Vary cannot start from its noise. Restore it and change the variation seed instead.";
  const missing: string[] = [];
  if (image.loras?.length && !loraCapable) missing.push("its LoRAs");
  if (image.prompt_weighting === false && !capability?.prompt_weighting) missing.push("prompt weighting off");
  if ((image.clip_skip ?? 1) > (capability?.clip_skip?.max ?? 1)) missing.push(`CLIP skip ${image.clip_skip}`);
  if (image.cfg_rescale && !capability?.cfg_rescale) missing.push(`CFG rescale ${image.cfg_rescale}`);
  return missing.length
    ? `The current load cannot apply ${missing.join(", ")} from this image's recipe, so Vary would not match it.`
    : null;
}

/** The SD controls a restored gallery recipe recorded that the current load drops or clamps (the request
 *  fields omit what `capability` lacks); prompt weighting off is literalPromptNotApplied's. */
export function sdControlsNotRestored(
  image: {
    clip_skip?: number | null;
    cfg_rescale?: number | null;
    variation_seed?: number | null;
    variation_strength?: number | null;
  },
  capability: SdControlsCapability | null | undefined,
): string[] {
  const out: string[] = [];
  const clipSkip = image.clip_skip ?? 1;
  const clipMax = capability?.clip_skip?.max;
  if (clipSkip > 1) {
    if (clipMax === undefined) out.push(`CLIP skip ${clipSkip}`);
    else if (clipSkip > clipMax) out.push(`CLIP skip ${clipSkip} (clamped to ${clipMax})`);
  }
  const rescale = image.cfg_rescale ?? 0;
  if (rescale) {
    const hi = capability?.cfg_rescale ? Math.min(1, capability.cfg_rescale.max) : undefined;
    if (hi === undefined) out.push(`CFG rescale ${rescale}`);
    else if (rescale > hi) out.push(`CFG rescale ${rescale} (clamped to ${hi})`);
  }
  if (image.variation_strength && !capability?.variation_seed) {
    out.push(
      image.variation_seed != null
        ? `variation seed ${image.variation_seed} (strength ${image.variation_strength})`
        : `variation strength ${image.variation_strength}`,
    );
  }
  return out;
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
    // false = the run turned weighting off; null / absent = it was on (or never mattered).
    promptWeighting: image.prompt_weighting !== false,
    clipSkip: image.clip_skip ?? 1,
    cfgRescale: image.cfg_rescale ?? 0,
    variationSeed:
      image.variation_seed != null && image.variation_strength
        ? String(Math.max(0, image.variation_seed - offset))
        : "",
    variationStrength: image.variation_strength ?? 0,
  };
}

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
