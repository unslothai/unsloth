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

export function fastCfgRequestFields(available: boolean, on: boolean): { cfg_interval_end?: number } {
  return available && on ? { cfg_interval_end: FAST_CFG_INTERVAL_END } : {};
}

function intervalOf(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value < 1 ? value : undefined;
}

export function fastCfgFromRecipe(r: { cfg_interval_end?: unknown }): boolean {
  return intervalOf(r.cfg_interval_end) !== undefined;
}

export function fastCfgNotApplied(value: unknown, supported: boolean): string[] {
  const end = intervalOf(value);
  if (end === undefined) return [];
  if (!supported) return [`Fast CFG ${end} (not available for this model)`];
  return end !== FAST_CFG_INTERVAL_END ? [`Fast CFG ${end} (restored as ${FAST_CFG_INTERVAL_END})`] : [];
}

export function fastCfgLabel(end: number | null | undefined): string | null {
  const value = intervalOf(end);
  return value === undefined ? null : `On (CFG for the first ${Math.round(value * 100)}% of steps)`;
}
