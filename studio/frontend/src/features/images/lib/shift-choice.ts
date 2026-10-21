// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Per-generation Shift override; backend: core/inference/diffusion_flow_shift.py. "sd3" takes one static shift
 * (ComfyUI ModelSamplingSD3 / AuraFlow), "flux" takes base / max shift (ComfyUI ModelSamplingFlux). */
export interface ShiftControl {
  node: "sd3" | "flux";
  // sd3: the loaded value, null when the model's schedule depends on resolution.
  shift?: number | null;
  base_shift?: number;
  max_shift?: number;
}

/** Text fields as typed; empty = the model's own schedule. */
export interface ShiftValue {
  shift: string;
  baseShift: string;
  maxShift: string;
}

export const EMPTY_SHIFT: ShiftValue = { shift: "", baseShift: "", maxShift: "" };

export const SHIFT_MAX = 100;

function parse(text: string, min: number, exclusiveMin: boolean): number | undefined {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value > SHIFT_MAX) return undefined;
  if (exclusiveMin ? value <= min : value < min) return undefined;
  return value;
}

/** Request fields for the typed values; nothing when the model offers no Shift or the boxes are empty/invalid. */
export function shiftRequestFields(
  control: ShiftControl | null | undefined,
  value: ShiftValue,
): { shift?: number; base_shift?: number; max_shift?: number } {
  if (!control) return {};
  if (control.node === "sd3") {
    const shift = parse(value.shift, 0, true);
    return shift === undefined ? {} : { shift };
  }
  const base = parse(value.baseShift, 0, false);
  const max = parse(value.maxShift, 0, false);
  return {
    ...(base !== undefined ? { base_shift: base } : {}),
    ...(max !== undefined ? { max_shift: max } : {}),
  };
}

/** Whether a typed value is out of range (shown as an error state). Empty is valid. */
export function shiftFieldInvalid(text: string, exclusiveMin: boolean): boolean {
  return text.trim() !== "" && parse(text, 0, exclusiveMin) === undefined;
}

/** Append ``trigger`` to a prompt as A1111 users write embeddings: comma separated. */
export function withEmbeddingTrigger(prompt: string, trigger: string): string {
  const trimmed = prompt.trimEnd();
  if (!trimmed) return trigger;
  return /[,(]$/.test(trimmed) ? `${trimmed} ${trigger}` : `${trimmed}, ${trigger}`;
}
