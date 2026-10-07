// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The cache generation advances on any fingerprint change, including sizeBytes while downloading,
 * so unbounded re-checks never converge. Keyed on the selection. */

export const DATASET_CACHE_RECHECK_LIMIT = 3;

let currentKey: string | null = null;
let attempts = 0;

/** Mirrors the four user-chosen fields of DatasetCacheUsabilityIdentity. cachePath is excluded:
 * it moves during a download and would re-arm the loop. */
export interface DatasetRecheckSelection {
  dataset: string;
  subset: string | null;
  split: string;
  streaming: boolean;
}

export function datasetCacheRecheckKey(selection: DatasetRecheckSelection): string {
  // JSON rather than a separator: no delimiter collisions, and null differs from "null".
  return JSON.stringify([
    selection.dataset,
    selection.subset,
    selection.split,
    selection.streaming,
  ]);
}

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
