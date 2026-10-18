// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MoeInfo, ReapOptions } from "../api/export-api";

export const REAP_SPARSITIES = [0.25, 0.5] as const;
export const REAP_SAMPLES = [
  { value: 128, label: "Quick", hint: "128 samples" },
  { value: 512, label: "Standard", hint: "512 samples, recommended" },
] as const;

export interface ReapSettings {
  enabled: boolean;
  sparsity: number;
  samples: number;
  dataset: string;
}

export const DEFAULT_REAP_SETTINGS: ReapSettings = {
  enabled: false,
  sparsity: 0.25,
  samples: 512,
  dataset: "",
};

/** Experts kept per MoE layer, as Core computes it (int(E * sparsity) dropped). */
export function reapKeptExperts(moe: MoeInfo, sparsity: number): number {
  return moe.num_experts - Math.floor(moe.num_experts * sparsity);
}

/** Core refuses a sparsity that drops no expert or leaves fewer experts than the router's top_k. */
export function reapSparsityAllowed(moe: MoeInfo, sparsity: number): boolean {
  const kept = reapKeptExperts(moe, sparsity);
  return kept < moe.num_experts && kept >= moe.top_k;
}

export function reapAllowedSparsities(moe: MoeInfo): number[] {
  return REAP_SPARSITIES.filter((s) => reapSparsityAllowed(moe, s));
}

/** The chosen sparsity, or the first allowed one when the model changed under it. */
export function reapEffectiveSparsity(
  settings: ReapSettings,
  moe: MoeInfo,
): number | null {
  const allowed = reapAllowedSparsities(moe);
  return allowed.includes(settings.sparsity)
    ? settings.sparsity
    : (allowed[0] ?? null);
}

/** The request block for an enabled setting, else null (REAP stays off). */
export function reapRequest(
  settings: ReapSettings,
  moe: MoeInfo,
): ReapOptions | null {
  const sparsity = reapEffectiveSparsity(settings, moe);
  if (!settings.enabled || sparsity === null) return null;
  const dataset = settings.dataset.trim();
  return {
    sparsity,
    num_calibration_samples: settings.samples,
    dataset: dataset || null,
  };
}

export function reapSummaryLabel(
  settings: ReapSettings,
  moe: MoeInfo,
): string | null {
  const sparsity = reapEffectiveSparsity(settings, moe);
  if (!settings.enabled || sparsity === null) return null;
  return `${Math.round(sparsity * 100)}% of experts removed (${reapKeptExperts(moe, sparsity)} of ${moe.num_experts} kept)`;
}
