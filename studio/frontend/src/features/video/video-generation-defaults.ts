// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const VIDEO_DEFAULT_GEN = { steps: 8, guidance: 1 };

// Per-model generation defaults, matched by repository substring from most specific to broadest.
const MODEL_DEFAULTS: Array<{
  match: string;
  family: string;
  steps: number;
  guidance: number;
}> = [
  { match: "minimax-h3", family: "minimax-h3", steps: 30, guidance: 1 },
  { match: "minimax_h3", family: "minimax-h3", steps: 30, guidance: 1 },
  // Distilled must precede generic LTX: it runs at 8 steps/CFG 1 rather than 40/4.
  { match: "distilled", family: "ltx-2", steps: 8, guidance: 1 },
  { match: "ltx", family: "ltx-2", steps: 40, guidance: 4 },
  { match: "wan", family: "wan2.2", steps: 50, guidance: 5 },
  {
    match: "hunyuanvideo-1.5-720p",
    family: "hunyuanvideo-1.5",
    steps: 50,
    guidance: 6,
  },
  { match: "hunyuanvideo", family: "hunyuanvideo-1.5", steps: 50, guidance: 6 },
];

export function videoDefaultsFor(
  repoId: string,
  familyOverride?: string | null,
): { steps: number; guidance: number } {
  const idMatch = MODEL_DEFAULTS.find((entry) =>
    repoId.toLowerCase().includes(entry.match),
  );
  if (familyOverride && familyOverride !== "auto") {
    const familyMatch = MODEL_DEFAULTS.find((entry) =>
      familyOverride.toLowerCase().includes(entry.match),
    );
    // A variant recipe is authoritative only inside the explicitly resolved architecture.
    if (idMatch && familyMatch && idMatch.family === familyMatch.family) {
      return { steps: idMatch.steps, guidance: idMatch.guidance };
    }
    if (familyMatch) {
      return { steps: familyMatch.steps, guidance: familyMatch.guidance };
    }
  }
  return idMatch
    ? { steps: idMatch.steps, guidance: idMatch.guidance }
    : VIDEO_DEFAULT_GEN;
}
