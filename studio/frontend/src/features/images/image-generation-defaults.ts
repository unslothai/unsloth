// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Generation defaults when the model is unrecognised. Also seeds the Create sliders.
export const DEFAULT_GEN = { steps: 9, guidance: 0 };

const MODEL_DEFAULTS: Array<{
  match: string;
  steps: number;
  guidance: number;
}> = [
  { match: "z-image-turbo", steps: 9, guidance: 0 },
  // Krea 2 Raw is the undistilled base, so it must precede the distilled Krea key below.
  { match: "krea-2-raw", steps: 52, guidance: 3.5 },
  { match: "krea-2", steps: 8, guidance: 0 },
  { match: "flux.1-schnell", steps: 4, guidance: 0 },
  // Kontext and the Krea dev finetune must precede generic FLUX.1.
  { match: "kontext", steps: 28, guidance: 2.5 },
  { match: "flux.1-krea", steps: 28, guidance: 4.5 },
  { match: "flux.1", steps: 28, guidance: 3.5 },
  // Klein base is undistilled. The generic key below covers both distilled sizes.
  { match: "flux.2-klein-base", steps: 50, guidance: 4 },
  { match: "flux.2-klein", steps: 4, guidance: 1 },
  { match: "flux.2-dev", steps: 28, guidance: 4 },
  { match: "qwen-image", steps: 20, guidance: 4 },
  { match: "z-image", steps: 20, guidance: 4 },
  { match: "ideogram", steps: 48, guidance: 7 },
  { match: "lumina", steps: 50, guidance: 4 },
  { match: "hunyuanimage", steps: 50, guidance: 3.25 },
  { match: "hidream-i1-dev", steps: 28, guidance: 0 },
  { match: "hidream-i1-fast", steps: 16, guidance: 0 },
  { match: "hidream", steps: 50, guidance: 5 },
  { match: "sdxl-turbo", steps: 3, guidance: 0 },
  { match: "stable-diffusion-xl", steps: 30, guidance: 7 },
  { match: "sdxl", steps: 30, guidance: 7 },
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
  build: { modelKind?: string | null; transformerQuant?: string | null },
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
  const id = repoId.toLowerCase();
  const matched = QUANTISED_CANVAS.find(
    (entry) => id.includes(entry.match) && entry.schemes.includes(scheme),
  );
  return matched
    ? { width: matched.width, height: matched.height }
    : DEFAULT_RESOLUTION;
}
