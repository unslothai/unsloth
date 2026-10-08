// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { explicitFamily } from "../model-picker/components/model-selector/family-override.ts";

// Generation defaults when the model is unrecognised. Also seeds the Create sliders.
export const DEFAULT_GEN = { steps: 9, guidance: 0 };

const MODEL_DEFAULTS: Array<{
  match: string;
  steps: number;
  guidance: number;
}> = [
  { match: "z-image-turbo", steps: 8, guidance: 0 },
  // Krea 2 Raw is the undistilled base, so it must precede the distilled Krea key below.
  { match: "krea-2-raw", steps: 52, guidance: 3.5 },
  { match: "krea-2", steps: 8, guidance: 0 },
  { match: "flux.1-schnell", steps: 4, guidance: 0 },
  // Kontext and the Krea dev finetune must precede generic FLUX.1.
  { match: "kontext", steps: 20, guidance: 2.5 },
  { match: "flux.1-krea", steps: 20, guidance: 3.5 },
  { match: "flux.1", steps: 20, guidance: 3.5 },
  // Klein base is undistilled. The generic key below covers both distilled sizes.
  { match: "flux.2-klein-base", steps: 20, guidance: 5 },
  { match: "flux.2-klein", steps: 4, guidance: 1 },
  { match: "flux.2-dev", steps: 20, guidance: 4 },
  // Qwen-Image-2.1 and its aliases, before the generic key.
  { match: "qwen-image-2.1", steps: 25, guidance: 1 },
  { match: "qwen-image-21", steps: 25, guidance: 1 },
  { match: "qwen_image_21", steps: 25, guidance: 1 },
  { match: "qwenimage21", steps: 25, guidance: 1 },
  // ComfyUI Image to Layers template, before the generic key.
  { match: "qwen-image-layered", steps: 20, guidance: 2.5 },
  { match: "qwen_image_layered", steps: 20, guidance: 2.5 },
  { match: "qwenimagelayered", steps: 20, guidance: 2.5 },
  { match: "qwen-image-edit-2509", steps: 20, guidance: 4 },
  { match: "qwen-image-edit", steps: 40, guidance: 4 },
  { match: "qwen-image-2512", steps: 50, guidance: 4 },
  { match: "qwen-image", steps: 20, guidance: 4 },
  { match: "z-image", steps: 25, guidance: 3 },
  { match: "ideogram", steps: 20, guidance: 7 },
  { match: "lumina", steps: 50, guidance: 4 },
  { match: "hunyuanimage", steps: 50, guidance: 3.25 },
  { match: "hidream-i1-dev", steps: 28, guidance: 0 },
  { match: "hidream-i1-fast", steps: 16, guidance: 0 },
  { match: "hidream", steps: 50, guidance: 5 },
  { match: "sdxl-turbo", steps: 3, guidance: 0 },
  { match: "stable-diffusion-xl", steps: 25, guidance: 7 },
  { match: "sdxl", steps: 25, guidance: 7 },
];

export function defaultsFor(repoId: string): {
  steps: number;
  guidance: number;
} {
  const id = repoId.toLowerCase();
  const matched = MODEL_DEFAULTS.find((entry) => id.includes(entry.match));
  return matched
    ? { steps: matched.steps, guidance: matched.guidance }
    : DEFAULT_GEN;
}

/** A recognizable variant (Schnell vs Dev) keeps its own key; an explicit family keys only an opaque path. */
export function defaultsKeyFor(repoId: string, familyOverride: unknown): string {
  return defaultsFor(repoId) !== DEFAULT_GEN ? repoId : (explicitFamily(familyOverride) ?? repoId);
}

/** The loaded model's recipe for a pick that got the fallback (its name named no family), else null. */
export function loadedRecipeFor(
  pickDefaults: { steps: number; guidance: number } | null | undefined,
  residentKey: string,
  reported?: { steps?: number; guidance?: number } | null,
): { steps: number; guidance: number } | null {
  if (pickDefaults !== DEFAULT_GEN) return null;
  const resident = residentRecipeFor(residentKey, reported);
  return resident.steps === DEFAULT_GEN.steps && resident.guidance === DEFAULT_GEN.guidance ? null : resident;
}

/** The resident model's recipe: the backend's own when it reports one, else the base-repo key's. */
export function residentRecipeFor(
  residentKey: string,
  reported?: { steps?: number; guidance?: number } | null,
): { steps: number; guidance: number } {
  if (reported && typeof reported.steps === "number" && typeof reported.guidance === "number") {
    return { steps: reported.steps, guidance: reported.guidance };
  }
  return defaultsFor(residentKey);
}

export function residentDefaultsKey(
  repoId: string,
  baseRepo: string | null | undefined,
  resolvedFamily: { value?: unknown; source?: "auto" | "explicit" } | null | undefined,
): string {
  return defaultsKeyFor(baseRepo ?? repoId, resolvedFamily?.source === "explicit" ? resolvedFamily.value : null);
}

// The canvas every family is tuned for, and what an unrecognised model gets.
export const DEFAULT_RESOLUTION = { width: 1024, height: 1024 } as const;

// Families whose QUANTISED pipeline build defaults to a smaller canvas, with the schemes that
// trigger it.
//
// Quantising the denoiser shrinks the weights and leaves the activations alone, so on a family
// whose activations dominate, the canvas is what decides whether the load fits. Qwen-Image-2.1 at
// 1024 measures about 26 GB against about 19 GB at 512, which is the difference between running
// and not on a 24 GB card, and someone who picked a quantised build is telling us the card is the
// constraint. Dense bf16 keeps 1024: it was never going to fit a small card either way, so
// shrinking its canvas would cost quality and buy nothing.
//
// The GGUF route is deliberately absent. It streams the denoiser off disk, so its footprint does
// not turn on this, and it keeps 1024.
//
// Only a PICKED quant shrinks it: auto precision takes the hosted int8 on every card, so it says
// nothing about VRAM.
const QUANTISED_CANVAS: Array<{
  match: string;
  schemes: readonly string[];
  width: number;
  height: number;
}> = [
  {
    match: "qwen-image-2.1",
    schemes: ["fp8", "fp8_dynamic", "int8", "nvfp4"],
    width: 512,
    height: 512,
  },
];

export function resolutionFor(
  repoId: string,
  build: {
    modelKind?: string | null;
    transformerQuant?: string | null;
    // Absent on older backends: treated as a pick.
    transformerQuantSource?: string | null;
  },
): { width: number; height: number } {
  // A GGUF resident reports no family substring in repo_id, so callers pass base_repo; the kind is
  // what actually excludes that route, not the id.
  if ((build.modelKind ?? "").toLowerCase() === "gguf") {
    return DEFAULT_RESOLUTION;
  }
  const scheme = (build.transformerQuant ?? "")
    .toLowerCase()
    .replace(/-/g, "_");
  if (!scheme) {
    return DEFAULT_RESOLUTION;
  }
  if ((build.transformerQuantSource ?? "").toLowerCase() === "auto") {
    return DEFAULT_RESOLUTION;
  }
  const id = repoId.toLowerCase();
  const matched = QUANTISED_CANVAS.find(
    (entry) => id.includes(entry.match) && entry.schemes.includes(scheme),
  );
  return matched
    ? { width: matched.width, height: matched.height }
    : DEFAULT_RESOLUTION;
}
