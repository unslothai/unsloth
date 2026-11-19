// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Names the LOADED build's precision, not the catalog row's ("BF16", "(Fast FP8)").

const FAST_QUALIFIER = /\s*\(Fast(?: [A-Z0-9_]+)?\)$/;

export function withLoadedPrecision<T extends { name: string; descriptionSuffix?: string }>(
  model: T,
  precision: string | null | undefined,
): T {
  if (!precision) return model;
  return { ...model, name: model.name.replace(FAST_QUALIFIER, ""), descriptionSuffix: precision };
}
