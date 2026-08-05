// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Retry budget for the cached dataset format re-check.
 *
 * When a format check finishes against a cache generation that has since advanced, the
 * answer may be stale and the store re-runs it. The rejection tracker advances that
 * generation on any inventory-fingerprint change, and the fingerprint includes
 * sizeBytes, so a dataset that is still downloading invalidates every in-flight check.
 * Unbounded, that pair never converges (unslothai/unsloth#7853).
 *
 * The budget is keyed on the selection so switching dataset or split starts fresh.
 */

export const DATASET_CACHE_RECHECK_LIMIT = 3;

let currentKey: string | null = null;
let attempts = 0;

export function datasetCacheRecheckKey(datasetName: string, split: string): string {
  // "::" cannot appear in a split name, so keys cannot collide across selections.
  return `${datasetName}::${split}`;
}

/** True while the selection still has re-checks left; consumes one. */
export function claimDatasetCacheRecheck(key: string): boolean {
  if (currentKey !== key) {
    currentKey = key;
    attempts = 0;
  }
  if (attempts >= DATASET_CACHE_RECHECK_LIMIT) {
    return false;
  }
  attempts += 1;
  return true;
}

export function resetDatasetCacheRecheckBudget(): void {
  currentKey = null;
  attempts = 0;
}
