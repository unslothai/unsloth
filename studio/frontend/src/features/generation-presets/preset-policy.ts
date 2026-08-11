// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MediaGenerationPreset, MediaPresetSource } from "./types";

export const DEFAULT_PRESET_NAME = "Default";

export type ResidentMediaLoadTarget = {
  repoId: string;
  kind: "gguf" | "single_file" | "pipeline";
  filename?: string;
};

type ResidentMediaStatus = {
  loaded?: boolean;
  repo_id?: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
};

export function reapplyTargetFromStatus(
  status: ResidentMediaStatus | null | undefined,
): ResidentMediaLoadTarget | null {
  if (!(status?.loaded && status.repo_id)) {
    return null;
  }
  const kind = status.model_kind;
  if (kind !== "gguf" && kind !== "single_file" && kind !== "pipeline") {
    return null;
  }
  if (kind === "pipeline") {
    return { repoId: status.repo_id, kind };
  }
  if (!status.gguf_filename) {
    return null;
  }
  return {
    repoId: status.repo_id,
    kind,
    filename: status.gguf_filename,
  };
}

export type DynamicDefaultRollback<Params> = (
  restore: (current: Params) => Params,
) => boolean;

export function chainDynamicDefaultRollback<Params>(
  previous: DynamicDefaultRollback<Params> | undefined,
  next: DynamicDefaultRollback<Params>,
): DynamicDefaultRollback<Params> {
  if (!previous) {
    return next;
  }
  return (restore) => next(restore) && previous(restore);
}

export function presetSource(
  name: string,
): Exclude<MediaPresetSource, "modified"> {
  return name === DEFAULT_PRESET_NAME ? "builtin-default" : "custom";
}

export function configKey(value: unknown): string {
  return JSON.stringify(value ?? null);
}

export function mergeUntouchedParams<Params extends object>(
  baseline: Params,
  current: Params,
  next: Params,
): Params {
  const merged = { ...next };
  for (const key of Object.keys(merged) as Array<keyof Params>) {
    if (configKey(current[key]) !== configKey(baseline[key])) {
      merged[key] = current[key];
    }
  }
  return merged;
}

export function getBuiltinVariantName(
  usedNames: Set<string>,
  baseName = DEFAULT_PRESET_NAME,
): string {
  let suffix = 1;
  let candidate = `${baseName} ${suffix}`;
  while (usedNames.has(candidate)) {
    suffix += 1;
    candidate = `${baseName} ${suffix}`;
  }
  return candidate;
}

export function normalizeCustomPresets<Params, LoadConfig>(
  presets: MediaGenerationPreset<Params, LoadConfig>[],
): MediaGenerationPreset<Params, LoadConfig>[] {
  const usedNames = new Set([DEFAULT_PRESET_NAME]);
  const normalized: MediaGenerationPreset<Params, LoadConfig>[] = [];
  for (const preset of presets) {
    const trimmed = preset.name.trim();
    if (!trimmed) {
      continue;
    }
    const name = usedNames.has(trimmed)
      ? getBuiltinVariantName(usedNames, trimmed)
      : trimmed;
    usedNames.add(name);
    normalized.push({ ...preset, name });
  }
  return normalized;
}

export function closestResolutionIndex(
  presets: [number, number][],
  targetWidth: number,
  targetHeight: number,
): number {
  if (presets.length === 0) {
    return 0;
  }
  const targetAspect = targetWidth / targetHeight;
  const targetArea = targetWidth * targetHeight;
  let bestIndex = 0;
  let bestScore = Number.POSITIVE_INFINITY;
  presets.forEach(([width, height], index) => {
    const aspectDistance = Math.abs(Math.log(width / height / targetAspect));
    const areaDistance = Math.abs(Math.log((width * height) / targetArea));
    const score = aspectDistance * 4 + areaDistance;
    if (score < bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });
  return bestIndex;
}

export function closestDurationIndex(
  durations: Array<{ seconds: number }>,
  targetSeconds: number,
): number {
  if (durations.length === 0) {
    return 0;
  }
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  durations.forEach(({ seconds }, index) => {
    const distance = Math.abs(seconds - targetSeconds);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}
