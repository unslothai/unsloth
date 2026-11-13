// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const planPrintTokenization = (
  sizes: readonly number[],
  budget: number,
): number[] => {
  const order = sizes
    .map((_, index) => index)
    .sort((a, b) => sizes[a] - sizes[b]);
  const planned: number[] = [];
  let left = budget;
  for (const index of order) {
    if (sizes[index] > left) break;
    left -= sizes[index];
    planned.push(index);
  }
  return planned;
};
