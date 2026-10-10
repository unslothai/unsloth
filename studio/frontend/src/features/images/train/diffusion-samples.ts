// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionSampleImage } from "../api";

/** The backend refuses more than this many sample prompts. */
export const MAX_SAMPLE_PROMPTS = 4;

const LINE_BREAK = /\r?\n/;

/** One prompt per line; blank lines dropped, capped at the backend's limit. */
export function parseSamplePrompts(text: string): string[] {
  return text
    .split(LINE_BREAK)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .slice(0, MAX_SAMPLE_PROMPTS);
}

export interface SampleRound {
  step: number;
  images: DiffusionSampleImage[];
}

/** Preview images grouped into rounds by step, oldest first; within a round in prompt order. */
export function groupSamplesByStep(
  samples: DiffusionSampleImage[] | null | undefined,
): SampleRound[] {
  const byStep = new Map<number, DiffusionSampleImage[]>();
  for (const s of samples ?? []) {
    if (!s || typeof s.path !== "string" || !Number.isFinite(s.step)) {
      continue;
    }
    const list = byStep.get(s.step) ?? [];
    list.push(s);
    byStep.set(s.step, list);
  }
  return Array.from(byStep.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([step, images]) => ({ step, images }));
}
