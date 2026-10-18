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

/** Python's round(): halves go to the even neighbour. */
function roundHalfEven(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff !== 0.5) return Math.round(x);
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Core's plan for one sparsity (unsloth/models/reap.py): experts dropped, and whether it runs. */
function reapPlan(
  moe: MoeInfo,
  sparsity: number,
): { dropped: number; allowed: boolean } {
  const experts = moe.num_experts;
  let dropped = Math.floor(experts * sparsity);
  if (dropped === 0) return { dropped, allowed: false };
  let reachable = experts - dropped;
  const groups = moe.n_group ?? 0;
  const topkGroup = moe.topk_group ?? 0;
  if (groups > 0 && topkGroup > 0) {
    // Group-limited routing drops the same number from every group.
    if (experts % groups !== 0) return { dropped, allowed: false };
    const perGroup = roundHalfEven(dropped / groups);
    const groupSize = experts / groups;
    dropped = perGroup * groups;
    if (perGroup === 0 || groupSize - perGroup < 2) {
      return { dropped, allowed: false };
    }
    reachable = topkGroup * (groupSize - perGroup);
  }
  return { dropped, allowed: moe.top_k <= reachable };
}

/** Experts kept per MoE layer, as Core computes it. */
export function reapKeptExperts(moe: MoeInfo, sparsity: number): number {
  return moe.num_experts - reapPlan(moe, sparsity).dropped;
}

/** Core refuses a sparsity that drops no expert or leaves fewer experts than the router reaches. */
export function reapSparsityAllowed(moe: MoeInfo, sparsity: number): boolean {
  return reapPlan(moe, sparsity).allowed;
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
