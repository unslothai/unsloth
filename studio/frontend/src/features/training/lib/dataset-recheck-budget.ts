


/**
* Retry budget for the cached dataset format re-check.
*
* When a format check finishes against a cache generation that has since advanced, the answer
* may be stale and the store re-runs it. The rejection tracker advances that generation on any
* inventory-fingerprint change, and the fingerprint includes sizeBytes, so a dataset that is
* still downloading invalidates every in-flight check. Unbounded, that pair never converges
* (unslothai/unsloth#7853). The budget is keyed on the selection so a new one starts fresh.
*/

export const DATASET_CACHE_RECHECK_LIMIT = 3;

let currentKey: string | null = null;
let attempts = 0;

/**
* Identity of a dataset *selection*, mirroring the four user-chosen dimensions of
* DatasetCacheUsabilityIdentity. Keying on fewer fields makes a genuinely different selection
* inherit an exhausted budget and lose its local-cache preference.
*
* cachePath is deliberately excluded even though the usability identity carries it: it is
* derived state that moves as a download populates the cache, so keying on it would re-arm the
* non-terminating loop this module exists to bound (unslothai/unsloth#7853).
*/
export interface DatasetRecheckSelection {
  dataset: string;
  subset: string | null;
  split: string;
  streaming: boolean;
}

export function datasetCacheRecheckKey(selection: DatasetRecheckSelection): string {
  // JSON encoding rather than a separator: no delimiter can collide with a name that
  // happens to contain it, and null is distinguishable from the string "null".
  return JSON.stringify([
    selection.dataset,
    selection.subset,
    selection.split,
    selection.streaming,
  ]);
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
