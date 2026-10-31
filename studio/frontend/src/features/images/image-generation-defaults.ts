// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { explicitFamily } from "../model-picker/components/model-selector/family-override.ts";

export const DEFAULT_GEN = { steps: 9, guidance: 0 };

const MODEL_DEFAULTS: Array<{
  match: string;
  steps: number;
  guidance: number;
}> = [
  { match: "z-image-turbo", steps: 8, guidance: 0 },
  // Krea 2 Raw is undistilled and must precede the distilled Krea key.
  { match: "krea-2-raw", steps: 52, guidance: 3.5 },
  // The same Raw checkpoint as GGUFs name it (Krea-2-Base-*, krea2_raw-*).
  { match: "krea-2-base", steps: 52, guidance: 3.5 },
  { match: "krea2-raw", steps: 52, guidance: 3.5 },
  { match: "krea2-base", steps: 52, guidance: 3.5 },
  { match: "krea-2", steps: 8, guidance: 0 },
  { match: "flux.1-schnell", steps: 4, guidance: 0 },
  // Must precede generic FLUX.1.
  { match: "kontext", steps: 20, guidance: 2.5 },
  { match: "flux.1-krea", steps: 20, guidance: 3.5 },
  { match: "flux.1", steps: 20, guidance: 3.5 },
  // FLUX.3 Image Klein (ComfyUI templates, the BFL reference): base 50 steps at CFG 3.5, distilled 16 steps without CFG. Base first.
  // Every alias the backend registers for the two families, and the canonical family names (an opaque path
  // resolves through the explicit family).
  { match: "flux.3-klein-base", steps: 50, guidance: 3.5 },
  { match: "flux-3-image-klein-base", steps: 50, guidance: 3.5 },
  { match: "flux3-image-klein-base", steps: 50, guidance: 3.5 },
  { match: "flux3-klein-base", steps: 50, guidance: 3.5 },
  { match: "flux.3-klein", steps: 16, guidance: 1 },
  { match: "flux-3-image-klein", steps: 16, guidance: 1 },
  { match: "flux3-image-klein", steps: 16, guidance: 1 },
  { match: "flux3-klein", steps: 16, guidance: 1 },
  // Klein base is undistilled. The generic key below covers both distilled sizes.
  { match: "flux.2-klein-base", steps: 20, guidance: 5 },
  { match: "flux.2-klein", steps: 4, guidance: 1 },
  { match: "flux.2-dev", steps: 20, guidance: 4 },
  // Qwen-Image-2.1 and its aliases, before the generic key, Turbo before 2.1.
  { match: "qwen-image-2.1-turbo", steps: 8, guidance: 1 },
  { match: "qwen-image-21-turbo", steps: 8, guidance: 1 },
  { match: "qwenimage21-turbo", steps: 8, guidance: 1 },
  { match: "qwenimage21turbo", steps: 8, guidance: 1 },
  { match: "qwen-image-2.1", steps: 25, guidance: 1 },
  { match: "qwen-image-21", steps: 25, guidance: 1 },
  { match: "qwen_image_21", steps: 25, guidance: 1 },
  { match: "qwenimage21", steps: 25, guidance: 1 },
  { match: "qwen-image-layered", steps: 20, guidance: 2.5 },
  { match: "qwen_image_layered", steps: 20, guidance: 2.5 },
  { match: "qwenimagelayered", steps: 20, guidance: 2.5 },
  { match: "qwen-image-edit-2509", steps: 20, guidance: 4 },
  { match: "qwen-image-edit", steps: 40, guidance: 4 },
  { match: "qwen-image-2512", steps: 50, guidance: 4 },
  { match: "qwen-image", steps: 20, guidance: 4 },
  { match: "z-image", steps: 25, guidance: 3 },
  { match: "ideogram", steps: 20, guidance: 7 },
  { match: "moss-image", steps: 25, guidance: 2.5 },
  { match: "moss_image", steps: 25, guidance: 2.5 },
  { match: "mossimage", steps: 25, guidance: 2.5 },
  { match: "lumina", steps: 50, guidance: 4 },
  { match: "hunyuanimage", steps: 50, guidance: 3.25 },
  { match: "hidream-i1-dev", steps: 28, guidance: 0 },
  { match: "hidream-i1-fast", steps: 16, guidance: 0 },
  { match: "hidream", steps: 50, guidance: 5 },
  { match: "sdxl-turbo", steps: 1, guidance: 0 },
  { match: "stable-diffusion-xl", steps: 25, guidance: 7 },
  { match: "sdxl", steps: 25, guidance: 7 },
];

// The Steps slider's usual top, and the request's (DiffusionGenerateRequest.steps is 1-100).
export const STEPS_SLIDER_MAX = 50;
export const STEPS_REQUEST_MAX = 100;

/** Steps slider top: 50, widened to twice the model's default (Krea-2-raw ships 52) and to the
 *  current value (an import or restore), never past what the request accepts. */
export function stepsSliderMax(modelDefault: number, current: number): number {
  const wanted = Math.max(STEPS_SLIDER_MAX, modelDefault * 2, current);
  return Math.min(STEPS_REQUEST_MAX, Number.isFinite(wanted) ? Math.ceil(wanted) : STEPS_SLIDER_MAX);
}

export function defaultsFor(repoId: string): {
  steps: number;
  guidance: number;
} {
  // Mirrors the backend's name_key_in, so both tables resolve one spelling alike.
  const fold = (text: string) => text.replace(/[-_.\s]+/g, "-");
  const id = repoId.toLowerCase();
  const folded = fold(id);
  const matched = MODEL_DEFAULTS.find(
    (entry) => id.includes(entry.match) || folded.includes(fold(entry.match)),
  );
  return matched
    ? { steps: matched.steps, guidance: matched.guidance }
    : DEFAULT_GEN;
}

// Families whose explicit choice outranks the name: FLUX.3 base and distilled share names (BFL's ComfyUI base
// files are flux3_image_klein_*.safetensors), and the backend refuses a checkpoint of the other variant.
const FAMILY_NAMES_THE_VARIANT = new Set(["flux.3-klein", "flux.3-klein-base"]);

/** A recognizable variant (Schnell vs Dev) keeps its own key; an explicit family keys only an opaque path. A GGUF
 * filename that names a model outranks its repo, as on the backend: one repo can hold several checkpoints
 * (realrebelai/KREA-2_GGUFs ships BASE/Krea-2-Base-* beside TURBO/Krea-2-Turbo-*). */
export function defaultsKeyFor(repoId: string, familyOverride: unknown, ggufFilename?: string | null): string {
  const family = explicitFamily(familyOverride);
  if (family && FAMILY_NAMES_THE_VARIANT.has(family)) return family;
  if (ggufFilename && defaultsFor(ggufFilename) !== DEFAULT_GEN) return ggufFilename;
  return defaultsFor(repoId) !== DEFAULT_GEN ? repoId : (family ?? repoId);
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
  ggufFilename?: string | null,
): string {
  return defaultsKeyFor(
    baseRepo ?? repoId,
    resolvedFamily?.source === "explicit" ? resolvedFamily.value : null,
    ggufFilename,
  );
}

export const DEFAULT_RESOLUTION = { width: 1024, height: 1024 } as const;

// Families trained at a smaller canvas on every build (incl. GGUF); SDXL-Turbo is a 512x512 model.
const NATIVE_CANVAS: Array<{ match: string; width: number; height: number }> = [
  { match: "sdxl-turbo", width: 512, height: 512 },
];

// Quantised builds of activation-heavy families default to a smaller canvas so they fit 24 GB
// (Qwen-Image-2.1: ~26 GB at 1024 vs ~19 GB at 512). Only a picked quant shrinks it; GGUF keeps 1024.
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
    transformerQuantSource?: string | null;
  },
): { width: number; height: number } {
  const fold = (text: string) => text.toLowerCase().replace(/[-_.\s]+/g, "-");
  const native = NATIVE_CANVAS.find((entry) =>
    fold(repoId).includes(fold(entry.match)),
  );
  if (native) {
    return { width: native.width, height: native.height };
  }
  // A GGUF resident has no family substring in repo_id, so exclude it by kind.
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
