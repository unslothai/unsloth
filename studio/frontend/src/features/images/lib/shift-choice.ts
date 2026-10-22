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

export function shiftFieldInvalid(text: string, exclusiveMin: boolean): boolean {
  return text.trim() !== "" && parse(text, 0, exclusiveMin) === undefined;
}

export function withEmbeddingTrigger(prompt: string, trigger: string): string {
  const trimmed = prompt.trimEnd();
  if (!trimmed) return trigger;
  return /[,(]$/.test(trimmed) ? `${trimmed} ${trigger}` : `${trimmed}, ${trigger}`;
}

/** The Shift boxes a gallery record or imported recipe restores; absent values leave the model's schedule. */
export function shiftFromRecipe(r: {
  shift?: number | null;
  base_shift?: number | null;
  max_shift?: number | null;
}): ShiftValue {
  const text = (n: number | null | undefined) => (typeof n === "number" && Number.isFinite(n) ? String(n) : "");
  return { shift: text(r.shift), baseShift: text(r.base_shift), maxShift: text(r.max_shift) };
}

/** Imported Shift values the loaded model cannot take (no Shift control, or the other ComfyUI node's fields). */
export function shiftNotApplied(
  r: Readonly<Record<string, unknown>>,
  control: ShiftControl | null | undefined,
): string[] {
  const out: string[] = [];
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
  const shift = num(r.shift);
  const base = num(r.base_shift);
  const max = num(r.max_shift);
  // The request's bounds: shift in (0, SHIFT_MAX], base / max shift in [0, SHIFT_MAX].
  const name = (label: string, v: number, ok: boolean) =>
    out.push(ok ? `${label} ${v}` : `${label} ${v} (out of range)`);
  if (shift !== undefined && !(control?.node === "sd3" && shift > 0 && shift <= SHIFT_MAX))
    name("shift", shift, control?.node !== "sd3");
  if (base !== undefined && !(control?.node === "flux" && base >= 0 && base <= SHIFT_MAX))
    name("base shift", base, control?.node !== "flux");
  if (max !== undefined && !(control?.node === "flux" && max >= 0 && max <= SHIFT_MAX))
    name("max shift", max, control?.node !== "flux");
  return out;
}

/** Split imported textual-inversion names by whether an installed embedding (its trigger, any case) matches. */
export function splitInstalledEmbeddings(
  names: readonly string[],
  triggers: readonly string[],
): { installed: string[]; missing: string[] } {
  const known = new Set(triggers.map((t) => t.toLowerCase()));
  const installed: string[] = [];
  const missing: string[] = [];
  for (const name of names) (known.has(name.toLowerCase()) ? installed : missing).push(name);
  return { installed, missing };
}
