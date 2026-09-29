// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const DEFAULT_PRESET_NAME = "Default";

export function configKey(value: unknown): string {
  return JSON.stringify(value ?? null);
}

export function getBuiltinVariantName(usedNames: Set<string>): string {
  let suffix = 1;
  let candidate = `${DEFAULT_PRESET_NAME} ${suffix}`;
  while (usedNames.has(candidate)) {
    suffix += 1;
    candidate = `${DEFAULT_PRESET_NAME} ${suffix}`;
  }
  return candidate;
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

/**
 * A stored recipe owns the first status seed. Later model changes apply their own defaults, unless
 * the user claimed the form after the pick that is being confirmed: their choice is the newer one.
 */
export function shouldApplyModelDefaults(
  alreadySeeded: boolean,
  storedRecipe: boolean,
  supersededByUser = false,
): boolean {
  return !supersededByUser && (alreadySeeded || !storedRecipe);
}
