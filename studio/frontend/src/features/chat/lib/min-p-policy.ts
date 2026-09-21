// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MinPMode, PersistedInferenceParams } from "../types/runtime";

export function isMinPMode(value: unknown): value is MinPMode {
  return value === "server-default" || value === "custom";
}

/** Only call on saved records, before merging runtime defaults or partial edits. */
export function normalizeSavedMinP<T extends PersistedInferenceParams>(
  params: T,
): T {
  if (
    isMinPMode(params.minPMode) ||
    typeof params.minP !== "number" ||
    !Number.isFinite(params.minP)
  ) {
    return params;
  }
  return { ...params, minPMode: "custom" };
}

export function effectiveMinPMode(
  params: Pick<PersistedInferenceParams, "minPMode">,
): MinPMode {
  return params.minPMode ?? "server-default";
}

export function minPSamplingPayload(
  providerType: string | null | undefined,
  params: { minP: number; minPMode?: MinPMode },
): { min_p?: number } {
  return providerType === "vllm" &&
    effectiveMinPMode(params) === "server-default"
    ? {}
    : { min_p: params.minP };
}
