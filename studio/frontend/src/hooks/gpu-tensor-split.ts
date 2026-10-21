// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A positional ratio is only reusable with the ordered GPU IDs it was written for. */
export function normalizeTensorSplit(
  value: unknown,
  gpuIds: number[] | null | undefined,
): number[] | null {
  if (
    !Array.isArray(value) ||
    !gpuIds ||
    gpuIds.length < 2 ||
    value.length !== gpuIds.length
  )
    return null;
  if (
    !gpuIds.every((id) => Number.isInteger(id) && id >= 0) ||
    new Set(gpuIds).size !== gpuIds.length
  )
    return null;
  if (
    !value.every(
      (weight) =>
        typeof weight === "number" && Number.isFinite(weight) && weight >= 0,
    )
  )
    return null;
  const total = value.reduce((sum, weight) => sum + weight, 0);
  return Number.isFinite(total) && total > 0 ? [...value] : null;
}

export function reconcileTensorSplit(
  value: unknown,
  savedIds: number[] | null | undefined,
  resolvedIds: number[] | null | undefined,
): number[] | null {
  // Unpinned on both sides: the ratio spans every visible GPU, as a resident unpinned manual load reports it.
  if (savedIds == null && resolvedIds == null) {
    return Array.isArray(value)
      ? normalizeTensorSplit(value, value.map((_, index) => index))
      : null;
  }
  if (
    !savedIds ||
    !resolvedIds ||
    savedIds.length !== resolvedIds.length ||
    !savedIds.every((id, index) => id === resolvedIds[index])
  )
    return null;
  return normalizeTensorSplit(value, savedIds);
}
