// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const CONTEXT_LENGTH_FINE_STEP = 1;
export const CONTEXT_LENGTH_FIXED_STEPS = [4096, 8192] as const;

export const CONTEXT_LENGTH_SLIDER_STEPS = [
  { value: CONTEXT_LENGTH_FINE_STEP, label: "Fine / 1 token" },
  { value: CONTEXT_LENGTH_FIXED_STEPS[0], label: "4,096 tokens" },
  { value: CONTEXT_LENGTH_FIXED_STEPS[1], label: "8,192 tokens" },
] as const;

export type ContextLengthSliderStep =
  (typeof CONTEXT_LENGTH_SLIDER_STEPS)[number]["value"];

export function getContextLengthSliderBounds(
  min: number,
  max: number,
  step: number,
): { min: number; max: number } {
  const normalizedStep =
    Number.isFinite(step) && step > 0 ? Math.floor(step) : 1;
  const normalizedMin = Math.ceil(Math.min(min, max));
  const normalizedMax = Math.floor(Math.max(min, max));

  if (normalizedStep === 1) {
    return { min: normalizedMin, max: normalizedMax };
  }

  const alignedMin = Math.ceil(normalizedMin / normalizedStep) * normalizedStep;
  const alignedMax =
    Math.floor(normalizedMax / normalizedStep) * normalizedStep;
  return {
    min: alignedMin,
    max: Math.max(alignedMin, alignedMax),
  };
}

export function snapContextLengthToStep(
  value: number,
  min: number,
  max: number,
  step: number,
): number {
  const normalizedStep =
    Number.isFinite(step) && step > 0 ? Math.floor(step) : 1;
  const bounds = getContextLengthSliderBounds(min, max, normalizedStep);
  const candidate = Number.isFinite(value) ? value : bounds.min;
  const snapped = Math.round(candidate / normalizedStep) * normalizedStep;
  return Math.min(Math.max(snapped, bounds.min), bounds.max);
}
