// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Generation defaults when the model is unrecognised. Also seeds the Create sliders.
export const DEFAULT_GEN = { steps: 9, guidance: 0 };

const MODEL_DEFAULTS: Array<{
  match: string;
  family: string;
  steps: number;
  guidance: number;
}> = [
  { match: "z-image-turbo", family: "z-image", steps: 9, guidance: 0 },
  // Krea 2 Raw is the undistilled base, so it must precede the distilled Krea key below.
  { match: "krea-2-raw", family: "krea-2", steps: 52, guidance: 3.5 },
  { match: "krea-2", family: "krea-2", steps: 8, guidance: 0 },
  { match: "flux.1-schnell", family: "flux.1", steps: 4, guidance: 0 },
  // Kontext and the Krea dev finetune must precede generic FLUX.1.
  { match: "kontext", family: "flux.1-kontext", steps: 28, guidance: 2.5 },
  { match: "flux.1-krea", family: "flux.1", steps: 28, guidance: 4.5 },
  { match: "flux.1", family: "flux.1", steps: 28, guidance: 3.5 },
  // Klein base is undistilled. The generic key below covers both distilled sizes.
  { match: "flux.2-klein-base", family: "flux.2-klein", steps: 50, guidance: 4 },
  { match: "flux.2-klein", family: "flux.2-klein", steps: 4, guidance: 1 },
  { match: "flux.2-dev", family: "flux.2-dev", steps: 28, guidance: 4 },
  { match: "qwen-image", family: "qwen-image", steps: 20, guidance: 4 },
  { match: "z-image", family: "z-image", steps: 20, guidance: 4 },
  { match: "ideogram", family: "ideogram-4", steps: 48, guidance: 7 },
  { match: "lumina", family: "lumina-2", steps: 50, guidance: 4 },
  { match: "hunyuanimage", family: "hunyuanimage-2.1", steps: 50, guidance: 3.25 },
  { match: "hidream-i1-dev", family: "hidream-i1", steps: 28, guidance: 0 },
  { match: "hidream-i1-fast", family: "hidream-i1", steps: 16, guidance: 0 },
  { match: "hidream", family: "hidream-i1", steps: 50, guidance: 5 },
  { match: "sdxl-turbo", family: "sdxl", steps: 3, guidance: 0 },
  { match: "stable-diffusion-xl", family: "sdxl", steps: 30, guidance: 7 },
  { match: "sdxl", family: "sdxl", steps: 30, guidance: 7 },
];

export function defaultsFor(
  repoId: string,
  familyOverride?: string | null,
): {
  steps: number;
  guidance: number;
} {
  const id = repoId.toLowerCase();
  const idMatch = MODEL_DEFAULTS.find((entry) => id.includes(entry.match));
  if (familyOverride && familyOverride !== "auto") {
    const fam = familyOverride.toLowerCase();
    const famMatch = MODEL_DEFAULTS.find((entry) => fam.includes(entry.match));
    // A variant recipe is more specific only when it belongs to the resolved family. An opaque
    // merge named after another architecture must still obey the explicit family override.
    if (idMatch && famMatch && idMatch.family === famMatch.family) {
      return { steps: idMatch.steps, guidance: idMatch.guidance };
    }
    if (famMatch) return { steps: famMatch.steps, guidance: famMatch.guidance };
  }
  if (idMatch) return { steps: idMatch.steps, guidance: idMatch.guidance };
  return DEFAULT_GEN;
}
