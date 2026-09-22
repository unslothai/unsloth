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

// The rule lives in the backend memory planner, the only place that knows both the weights and the
// card size. A suggestion, never a cap on what the user may ask for.
export function resolutionFor(build: {
  recommendedCanvas?: number | null;
}): { width: number; height: number } {
  const px = build.recommendedCanvas;
  // Older backend, unified memory, or an unsizable plan: no opinion must not shrink on a guess.
  if (!px || !Number.isFinite(px) || px <= 0) {
    return DEFAULT_RESOLUTION;
  }
  return { width: px, height: px };
}

// The size a load's recommendation should seed, or null when the fields no longer hold the page's
// own last seed: a size the user chose, before the pick or while its download ran, stands.
export function canvasSeedFor(
  current: { width: number; height: number },
  seeded: { width: number; height: number },
  recommendedCanvas: number | null | undefined,
): { width: number; height: number } | null {
  if (current.width !== seeded.width || current.height !== seeded.height) {
    return null;
  }
  return resolutionFor({ recommendedCanvas });
}
