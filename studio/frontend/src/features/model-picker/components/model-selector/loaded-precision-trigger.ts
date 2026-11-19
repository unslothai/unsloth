// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The picker trigger names the LOADED build: a catalog row reads "BF16" (its stored precision) and a dense-quant host
// qualifies it "(Fast FP8)" for every 8-bit scheme, while the resident build may run INT8, FP8 or BF16. The row's speed
// qualifier is dropped and the suffix becomes the precision the backend reports.

const FAST_QUALIFIER = /\s*\(Fast(?: [A-Z0-9_]+)?\)$/;

export function withLoadedPrecision<T extends { name: string; descriptionSuffix?: string }>(
  model: T,
  precision: string | null | undefined,
): T {
  if (!precision) return model;
  return { ...model, name: model.name.replace(FAST_QUALIFIER, ""), descriptionSuffix: precision };
}
