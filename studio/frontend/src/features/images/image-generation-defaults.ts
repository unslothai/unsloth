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

export function defaultsFor(
  repoId: string,
  familyOverride?: string | null,
): {
  steps: number;
  guidance: number;
} {
  const id = repoId.toLowerCase();
  const matched = MODEL_DEFAULTS.find((entry) => id.includes(entry.match));
  if (matched) return { steps: matched.steps, guidance: matched.guidance };
  if (familyOverride && familyOverride !== "auto") {
    const fam = familyOverride.toLowerCase();
    const famMatch = MODEL_DEFAULTS.find((entry) => fam.includes(entry.match));
    if (famMatch) return { steps: famMatch.steps, guidance: famMatch.guidance };
  }
  return DEFAULT_GEN;
}
