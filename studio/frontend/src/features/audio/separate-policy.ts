// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it directly.

import type { AudioGenerationPresentation } from "./audio-page-policy";
import type { AudioSourceSelection } from "./audio-run-request";
import type { AudioToolPanel } from "./tools/types";

/** Mirrors SEPARATE_MAX_SECONDS in studio/backend/routes/inference.py. */
export const SEPARATE_MAX_SECONDS = 600;

export const ROFORMER_FAMILIES: ReadonlySet<string> = new Set([
  "bs_roformer",
  "mel_band_roformer",
]);

export type SeparateBlockerKind =
  | "source"
  | "source-busy"
  | "source-expired"
  | "source-error"
  | "too-long";

export interface SeparateBlocker {
  kind: SeparateBlockerKind;
  reason: string;
}

function clock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export function separateBlocker({
  source,
  sourceBusy,
  sourceExpired,
  sourceError,
}: {
  source: AudioSourceSelection | null;
  sourceBusy: boolean;
  sourceExpired: boolean;
  sourceError: string | null;
}): SeparateBlocker | null {
  if (sourceBusy) {
    return {
      kind: "source-busy",
      reason: "Waiting for the track to finish uploading.",
    };
  }
  if (sourceError) return { kind: "source-error", reason: sourceError };
  if (!source) return { kind: "source", reason: "Add a track to separate." };
  if (sourceExpired) {
    return {
      kind: "source-expired",
      reason: "This track expired. Add it again.",
    };
  }
  if (
    source.durationS !== null &&
    source.durationS > SEPARATE_MAX_SECONDS + 0.5
  ) {
    return {
      kind: "too-long",
      reason: `Separate tracks up to 10 minutes long. This one is ${clock(source.durationS)}.`,
    };
  }
  return null;
}

// GPU seconds per second of audio, measured on the pinned runtime (180 s track).
const GPU_SECONDS_PER_SECOND: Record<string, number> = {
  htdemucs: 0.032,
  htdemucs_6stems: 0.036,
  bs_roformer: 0.056,
  mel_band_roformer: 0.056,
};
const ROFORMER_NO_OVERLAP_SECONDS_PER_SECOND = 0.016;

export function estimateSeparateSeconds(
  family: string | null | undefined,
  durationS: number | null | undefined,
  overlap: boolean,
): number | null {
  if (!family || !durationS || durationS <= 0) return null;
  const rate =
    ROFORMER_FAMILIES.has(family) && !overlap
      ? ROFORMER_NO_OVERLAP_SECONDS_PER_SECOND
      : GPU_SECONDS_PER_SECOND[family];
  if (rate === undefined) return null;
  return Math.max(1, Math.round(rate * durationS + durationS / 60));
}

/** RoFormers on the CPU take minutes for seconds of audio (142 s for an 8 s clip). */
export function separateCpuWarning(
  family: string | null | undefined,
  device: string | null | undefined,
): string | null {
  if (device !== "cpu" || !family) return null;
  return ROFORMER_FAMILIES.has(family)
    ? "RoFormer models are very slow on the CPU: expect minutes for a short clip. Load into GPU when you can."
    : "Separating on the CPU is slower; expect about a second per second of audio.";
}

export interface OverlapValue {
  overlap: boolean;
}

export function overlapRequest(value: OverlapValue | undefined): {
  options?: { num_overlap: number };
} {
  return value?.overlap === false ? { options: { num_overlap: 1 } } : {};
}

export function overlapReloads(
  lastOverlap: boolean | undefined,
  overlap: boolean,
): boolean {
  return (lastOverlap ?? true) !== overlap;
}

export function separatePresentation(
  presentation: AudioGenerationPresentation | null,
  phase: string | null,
  reloading: boolean,
): AudioGenerationPresentation | null {
  if (!presentation) return null;
  switch (phase) {
    case "preparing":
      return { ...presentation, status: "Preparing the track…" };
    case "generating":
      return {
        ...presentation,
        status: reloading
          ? "Reloading the model with the new overlap, then separating…"
          : "Separating…",
      };
    case "finishing":
      return { ...presentation, status: "Saving stems…" };
    case "stopping":
      return { ...presentation, status: "Stopping…" };
    default:
      return presentation;
  }
}

/** Overlap is a load-time setting, so changing it reloads the model. */
export const roformerOverlapLogic: Omit<
  AudioToolPanel<OverlapValue>,
  "Component"
> = {
  id: "roformer-overlap",
  families: [...ROFORMER_FAMILIES],
  workflows: ["separate"],
  title: "Overlap",
  claims: [],
  initial: () => ({ overlap: true }),
  toRequest: (value) => overlapRequest(value),
};
