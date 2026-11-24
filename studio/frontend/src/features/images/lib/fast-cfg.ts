// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Measured preset (FLUX.3 Klein base, 50 steps): ~10% faster, LPIPS 0.016; 0.7 visibly drifted. */
export const FAST_CFG_INTERVAL_END = 0.8;

export const FAST_CFG_STORAGE_KEY = "unsloth_images_fast_cfg";

export const FAST_CFG_HINT =
  "Lossy speed-up: skips the unconditional (negative prompt) pass for the last 20% of the steps, about 10% faster " +
  "on FLUX.3 Klein base. Images change slightly; off keeps full CFG.";

export function fastCfgAvailable(
  status: { loaded?: boolean; supports_cfg_interval?: boolean } | null | undefined,
  guidance: number,
  external = false,
): boolean {
  return !external && status?.loaded === true && status.supports_cfg_interval === true && guidance > 1;
}

function intervalOf(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value < 1 ? value : undefined;
}

/** ``end``: a restored recipe's own stop (e.g. a 0.7 compare cell); otherwise the preset. */
export function fastCfgRequestFields(
  available: boolean,
  on: boolean,
  end?: number | null,
): { cfg_interval_end?: number } {
  return available && on ? { cfg_interval_end: intervalOf(end) ?? FAST_CFG_INTERVAL_END } : {};
}

export function fastCfgFromRecipe(r: { cfg_interval_end?: unknown }): boolean {
  return intervalOf(r.cfg_interval_end) !== undefined;
}

/** The recorded stop restore keeps (null: off, or the preset when the switch is turned on by hand). */
export function fastCfgEndFromRecipe(r: { cfg_interval_end?: unknown }): number | null {
  return intervalOf(r.cfg_interval_end) ?? null;
}

export function fastCfgNotApplied(value: unknown, supported: boolean): string[] {
  const end = intervalOf(value);
  return end !== undefined && !supported ? [`Fast CFG ${end} (not available for this model)`] : [];
}

export function fastCfgLabel(end: number | null | undefined): string | null {
  const value = intervalOf(end);
  return value === undefined ? null : `On (CFG for the first ${Math.round(value * 100)}% of steps)`;
}
